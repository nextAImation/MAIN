"""
Institutional-grade position management with MAE/MFE tracking.
Zero lookahead, deterministic position handling.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from decimal import Decimal, ROUND_DOWN


@dataclass
class PositionLot:
    """Individual lot for MAE/MFE tracking"""
    entry_price: float
    quantity: float
    entry_timestamp: int
    entry_bar: int
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    current_mfe: float = 0.0
    current_mae: float = 0.0


@dataclass
class PositionState:
    """Complete position state"""
    symbol: str
    direction: str  # 'long', 'short', 'flat'
    quantity: float
    avg_entry_price: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    entry_timestamp: Optional[int]
    entry_bar: Optional[int]
    lots: List[PositionLot]
    cumulative_commission: float


class PositionManager:
    """
    Manages position state with lot-level MAE/MFE tracking.
    Supports reverse position logic and proper PnL calculation.
    """
    
    def __init__(self, symbol: str, tick_size: float = 0.01, contract_type: str = 'linear'):
        self.symbol = symbol
        self.tick_size = tick_size
        self.contract_type = contract_type
        self.state = PositionState(
            symbol=symbol,
            direction='flat',
            quantity=0.0,
            avg_entry_price=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_pnl=0.0,
            entry_timestamp=None,
            entry_bar=None,
            lots=[],
            cumulative_commission=0.0
        )
        
        # Tracking
        self._equity_curve = []
        self._max_drawdown = 0.0
        self._peak_equity = 0.0
        
    def update_unrealized_pnl(self, current_price: float, timestamp: int, bar: int) -> None:
        """Update unrealized PnL and track MAE/MFE for all lots"""
        if self.state.direction == 'flat':
            self.state.unrealized_pnl = 0.0
            return
            
        # Calculate current position PnL
        if self.state.direction == 'long':
            price_diff = current_price - self.state.avg_entry_price
        else:  # short
            price_diff = self.state.avg_entry_price - current_price
            
        self.state.unrealized_pnl = price_diff * self.state.quantity
        
        # Update MAE/MFE for each lot
        for lot in self.state.lots:
            if self.state.direction == 'long':
                excursion = current_price - lot.entry_price
            else:  # short
                excursion = lot.entry_price - current_price
                
            # Update current excursion
            if not np.isnan(excursion):
                lot.current_mfe = max(lot.current_mfe, excursion) if excursion > 0 else lot.current_mfe
                lot.current_mae = min(lot.current_mae, excursion) if excursion < 0 else lot.current_mae
                
                # Update max excursions
                lot.max_favorable_excursion = max(lot.max_favorable_excursion, excursion)
                lot.max_adverse_excursion = min(lot.max_adverse_excursion, excursion)
        
        self.state.total_pnl = self.state.realized_pnl + self.state.unrealized_pnl
        
        # Update equity curve and drawdown
        current_equity = self.state.total_pnl
        self._equity_curve.append((timestamp, bar, current_equity))
        
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            
        drawdown = (self._peak_equity - current_equity) / abs(self._peak_equity) if self._peak_equity != 0 else 0.0
        self._max_drawdown = max(self._max_drawdown, drawdown)
    
    def add_to_position(self, direction: str, quantity: float, price: float, 
                       timestamp: int, bar: int, commission: float = 0.0) -> Tuple[bool, str]:
        """
        Add to existing position or open new position.
        Returns (success, message)
        """
        # Validate inputs
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if price <= 0:
            return False, "Price must be positive"
        
        # Round quantity to tick size
        quantity = self._round_to_tick(quantity)
        
        # Handle reverse position logic
        if self.state.direction != 'flat' and self.state.direction != direction:
            return self._reverse_position(direction, quantity, price, timestamp, bar, commission)
        
        # Add to existing position or open new
        if self.state.direction == 'flat':
            self._open_position(direction, quantity, price, timestamp, bar)
        else:
            self._add_to_existing_position(quantity, price, timestamp, bar)
        
        # Add commission
        self.state.cumulative_commission += commission
        
        return True, f"Added {quantity} to {direction} position at {price}"
    
    def reduce_position(self, quantity: float, price: float, timestamp: int, 
                       bar: int, commission: float = 0.0) -> Tuple[bool, str]:
        """
        Reduce position size.
        Returns (success, message)
        """
        if self.state.direction == 'flat':
            return False, "No position to reduce"
        
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if quantity > self.state.quantity:
            return False, "Reduce quantity exceeds position size"
        
        # Round quantity to tick size
        quantity = self._round_to_tick(quantity)
        
        # Calculate PnL for reduction
        if self.state.direction == 'long':
            pnl = (price - self.state.avg_entry_price) * quantity
        else:  # short
            pnl = (self.state.avg_entry_price - price) * quantity
            
        self.state.realized_pnl += pnl
        self.state.quantity -= quantity
        
        # Remove lots using FIFO
        remaining_reduce = quantity
        new_lots = []
        
        for lot in self.state.lots:
            if remaining_reduce <= 0:
                new_lots.append(lot)
                continue
                
            if lot.quantity <= remaining_reduce:
                remaining_reduce -= lot.quantity
            else:
                # Partial reduction of this lot
                new_lot = PositionLot(
                    entry_price=lot.entry_price,
                    quantity=lot.quantity - remaining_reduce,
                    entry_timestamp=lot.entry_timestamp,
                    entry_bar=lot.entry_bar,
                    max_favorable_excursion=lot.max_favorable_excursion,
                    max_adverse_excursion=lot.max_adverse_excursion
                )
                new_lots.append(new_lot)
                remaining_reduce = 0
        
        self.state.lots = new_lots
        
        # Check if position is fully closed
        if abs(self.state.quantity) < 1e-10:  # Floating point tolerance
            self.state.direction = 'flat'
            self.state.avg_entry_price = 0.0
            self.state.entry_timestamp = None
            self.state.entry_bar = None
            self.state.lots = []
        
        # Add commission
        self.state.cumulative_commission += commission
        
        return True, f"Reduced position by {quantity} at {price}, PnL: {pnl:.2f}"
    
    def close_position(self, price: float, timestamp: int, bar: int, 
                      commission: float = 0.0) -> Tuple[bool, str]:
        """Close entire position"""
        if self.state.direction == 'flat':
            return False, "No position to close"
        
        success, message = self.reduce_position(
            self.state.quantity, price, timestamp, bar, commission
        )
        
        if success:
            message = f"Closed {self.state.direction} position at {price}"
            
        return success, message
    
    def _open_position(self, direction: str, quantity: float, price: float, 
                      timestamp: int, bar: int) -> None:
        """Open new position"""
        self.state.direction = direction
        self.state.quantity = quantity
        self.state.avg_entry_price = price
        self.state.entry_timestamp = timestamp
        self.state.entry_bar = bar
        
        # Create initial lot
        lot = PositionLot(
            entry_price=price,
            quantity=quantity,
            entry_timestamp=timestamp,
            entry_bar=bar
        )
        self.state.lots = [lot]
    
    def _add_to_existing_position(self, quantity: float, price: float, 
                                timestamp: int, bar: int) -> None:
        """Add to existing position with average price calculation"""
        total_value = (self.state.avg_entry_price * self.state.quantity) + (price * quantity)
        self.state.quantity += quantity
        self.state.avg_entry_price = total_value / self.state.quantity
        
        # Add new lot
        new_lot = PositionLot(
            entry_price=price,
            quantity=quantity,
            entry_timestamp=timestamp,
            entry_bar=bar
        )
        self.state.lots.append(new_lot)
    
    def _reverse_position(self, direction: str, quantity: float, price: float,
                         timestamp: int, bar: int, commission: float) -> Tuple[bool, str]:
        """Close current position and open opposite position"""
        # First close current position
        close_pnl = self.state.realized_pnl
        close_quantity = self.state.quantity
        
        success, message = self.close_position(price, timestamp, bar, commission)
        if not success:
            return False, f"Failed to close position for reversal: {message}"
        
        # Now open new position in opposite direction
        self._open_position(direction, quantity, price, timestamp, bar)
        
        return True, f"Reversed {close_quantity} to {direction} at {price}, Close PnL: {close_pnl:.2f}"
    
    def _round_to_tick(self, value: float) -> float:
        """Round value to nearest tick size"""
        if self.tick_size <= 0:
            return value
            
        ticks = round(value / self.tick_size)
        return ticks * self.tick_size
    
    def get_position_summary(self) -> Dict:
        """Get complete position summary"""
        return {
            'symbol': self.state.symbol,
            'direction': self.state.direction,
            'quantity': self.state.quantity,
            'avg_entry_price': self.state.avg_entry_price,
            'realized_pnl': self.state.realized_pnl,
            'unrealized_pnl': self.state.unrealized_pnl,
            'total_pnl': self.state.total_pnl,
            'entry_timestamp': self.state.entry_timestamp,
            'entry_bar': self.state.entry_bar,
            'num_lots': len(self.state.lots),
            'cumulative_commission': self.state.cumulative_commission,
            'max_drawdown': self._max_drawdown,
            'peak_equity': self._peak_equity
        }
    
    def get_mae_mfe_summary(self) -> Dict:
        """Get MAE/MFE summary across all lots"""
        if not self.state.lots:
            return {'mae': 0.0, 'mfe': 0.0, 'avg_mae': 0.0, 'avg_mfe': 0.0}
        
        total_mae = sum(lot.max_adverse_excursion for lot in self.state.lots if not np.isnan(lot.max_adverse_excursion))
        total_mfe = sum(lot.max_favorable_excursion for lot in self.state.lots if not np.isnan(lot.max_favorable_excursion))
        
        return {
            'mae': total_mae,
            'mfe': total_mfe,
            'avg_mae': total_mae / len(self.state.lots),
            'avg_mfe': total_mfe / len(self.state.lots)
        }
    
    def reset(self) -> None:
        """Reset position manager to initial state"""
        self.__init__(self.symbol, self.tick_size, self.contract_type)