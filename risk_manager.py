"""
Institutional risk management with comprehensive controls.
Deterministic risk validation and enforcement.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta


@dataclass
class RiskParameters:
    """Complete risk configuration"""
    # Position limits
    max_position_size: float = 100.0
    max_open_exposure: float = 10000.0
    
    # Loss limits
    max_loss_per_trade: float = 500.0
    max_daily_loss: float = 2000.0
    max_drawdown: float = 0.2  # 20% as decimal
    
    # Concentration limits
    max_symbol_concentration: float = 0.3  # 30% of portfolio
    max_sector_concentration: float = 0.5  # 50% of portfolio
    
    # Leverage limits
    max_leverage: float = 10.0
    
    # Trading limits
    max_trades_per_day: int = 100
    max_order_value: float = 50000.0
    
    # Volatility limits
    max_volatility_exposure: float = 0.1  # 10% of portfolio
    atr_multiplier_limit: float = 3.0


@dataclass
class RiskState:
    """Current risk state across all symbols"""
    daily_pnl: float
    daily_trades: int
    daily_start_time: int
    current_drawdown: float
    peak_equity: float
    symbol_exposures: Dict[str, float]
    sector_exposures: Dict[str, float]
    trade_history: List[Dict]


class RiskManager:
    """
    Comprehensive risk management with real-time validation.
    Enforces all risk limits deterministically.
    """
    
    def __init__(self, initial_equity: float = 10000.0, params: Optional[RiskParameters] = None):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.params = params or RiskParameters()
        
        self.state = RiskState(
            daily_pnl=0.0,
            daily_trades=0,
            daily_start_time=self._get_daily_timestamp(),
            current_drawdown=0.0,
            peak_equity=initial_equity,
            symbol_exposures={},
            sector_exposures={},
            trade_history=[]
        )
        
    def validate_order(self, symbol: str, order_type: str, quantity: float, 
                      price: float, position_manager, atr_value: float = 0.0,
                      sector: str = "default") -> Tuple[bool, str]:
        """
        Validate order against all risk limits.
        Returns (is_valid, rejection_reason)
        """
        # Basic validation
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if price <= 0:
            return False, "Price must be positive"
        
        order_value = abs(quantity * price)
        
        # 1. Check position size limits
        if not self._check_position_size(symbol, quantity, position_manager):
            return False, "Position size limit exceeded"
        
        # 2. Check order value limit
        if order_value > self.params.max_order_value:
            return False, f"Order value {order_value} exceeds limit {self.params.max_order_value}"
        
        # 3. Check open exposure
        if not self._check_open_exposure(symbol, order_value, position_manager):
            return False, "Open exposure limit exceeded"
        
        # 4. Check per-trade loss limit
        if not self._check_trade_loss_limit(quantity, price, position_manager, atr_value):
            return False, "Potential trade loss exceeds limit"
        
        # 5. Check daily loss limit
        if not self._check_daily_loss_limit():
            return False, "Daily loss limit exceeded"
        
        # 6. Check drawdown limit
        if not self._check_drawdown_limit():
            return False, "Drawdown limit exceeded"
        
        # 7. Check concentration limits
        if not self._check_concentration_limits(symbol, order_value, sector):
            return False, "Concentration limit exceeded"
        
        # 8. Check leverage limit
        if not self._check_leverage_limit(order_value):
            return False, "Leverage limit exceeded"
        
        # 9. Check trading frequency
        if not self._check_trading_frequency():
            return False, "Trading frequency limit exceeded"
        
        # 10. Check volatility exposure
        if not self._check_volatility_exposure(atr_value, quantity):
            return False, "Volatility exposure limit exceeded"
        
        return True, "Order validated"
    
    def update_equity(self, equity: float, timestamp: int) -> None:
        """Update current equity and risk state"""
        self.current_equity = equity
        
        # Update peak equity and drawdown
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        
        self.state.current_drawdown = (self.state.peak_equity - equity) / abs(self.state.peak_equity) if self.state.peak_equity != 0 else 0.0
        
        # Reset daily metrics if new day
        if self._is_new_day(timestamp):
            self._reset_daily_metrics(timestamp)
    
    def record_trade(self, symbol: str, quantity: float, price: float, 
                    pnl: float, commission: float, timestamp: int) -> None:
        """Record completed trade for risk tracking"""
        trade_value = abs(quantity * price)
        
        # Update daily metrics
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1
        
        # Update symbol exposure
        self.state.symbol_exposures[symbol] = self.state.symbol_exposures.get(symbol, 0.0) + trade_value
        
        # Record trade history
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'pnl': pnl,
            'commission': commission,
            'trade_value': trade_value
        }
        self.state.trade_history.append(trade_record)
        
        # Clean old trade history (keep 30 days)
        self._clean_trade_history(timestamp)
    
    def adjust_quantity_for_risk(self, symbol: str, desired_quantity: float, 
                                price: float, position_manager, atr_value: float = 0.0,
                                sector: str = "default") -> float:
        """
        Adjust order quantity to comply with risk limits.
        Returns maximum allowed quantity.
        """
        if desired_quantity <= 0:
            return 0.0
        
        # Start with desired quantity and reduce until it passes all checks
        current_quantity = desired_quantity
        min_quantity = 0.0
        
        # Binary search for maximum allowed quantity
        for _ in range(20):  # Max 20 iterations for better convergence
            is_valid, _ = self.validate_order(
                symbol, 'limit', current_quantity, price, position_manager, atr_value, sector
            )
            
            if is_valid:
                min_quantity = current_quantity
                # Try larger quantity but never exceed desired
                if current_quantity >= desired_quantity:
                    break
                current_quantity = min(desired_quantity, current_quantity * 1.2)
            else:
                # Try smaller quantity
                current_quantity = (min_quantity + current_quantity) / 2
            
            # Convergence check
            if abs(current_quantity - min_quantity) < 0.001 or current_quantity < 0.001:
                break
        
        return max(0.0, min_quantity)
    
    def _check_position_size(self, symbol: str, quantity: float, position_manager) -> bool:
        """Check position size limits"""
        if position_manager.state.direction != 'flat':
            new_quantity = position_manager.state.quantity + quantity
        else:
            new_quantity = quantity
            
        return abs(new_quantity) <= self.params.max_position_size
    
    def _check_open_exposure(self, symbol: str, order_value: float, position_manager) -> bool:
        """Check total open exposure limits"""
        current_exposure = self._calculate_current_exposure()
        new_exposure = current_exposure + order_value
        
        return new_exposure <= self.params.max_open_exposure
    
    def _check_trade_loss_limit(self, quantity: float, price: float, 
                               position_manager, atr_value: float) -> bool:
        """Check per-trade loss limit using ATR-based stops"""
        if atr_value <= 0:
            return True
            
        # Calculate potential loss using ATR stop
        if position_manager.state.direction == 'long':
            potential_loss = quantity * atr_value * self.params.atr_multiplier_limit
        elif position_manager.state.direction == 'short':
            potential_loss = quantity * atr_value * self.params.atr_multiplier_limit
        else:
            # For new positions, use conservative estimate
            potential_loss = quantity * atr_value * 2.0
            
        return abs(potential_loss) <= self.params.max_loss_per_trade
    
    def _check_daily_loss_limit(self) -> bool:
        """Check daily loss limit"""
        return self.state.daily_pnl >= -self.params.max_daily_loss
    
    def _check_drawdown_limit(self) -> bool:
        """Check maximum drawdown limit"""
        return self.state.current_drawdown <= self.params.max_drawdown
    
    def _check_concentration_limits(self, symbol: str, order_value: float, sector: str) -> bool:
        """Check symbol and sector concentration limits"""
        portfolio_value = max(self.current_equity, 1.0)  # Avoid division by zero
        
        # Symbol concentration
        symbol_exposure = self.state.symbol_exposures.get(symbol, 0.0) + order_value
        symbol_concentration = symbol_exposure / portfolio_value
        
        if symbol_concentration > self.params.max_symbol_concentration:
            return False
        
        # Sector concentration
        sector_exposure = self.state.sector_exposures.get(sector, 0.0) + order_value
        sector_concentration = sector_exposure / portfolio_value
        
        if sector_concentration > self.params.max_sector_concentration:
            return False
            
        return True
    
    def _check_leverage_limit(self, order_value: float) -> bool:
        """Check leverage limits"""
        total_exposure = self._calculate_current_exposure() + order_value
        leverage = total_exposure / max(self.current_equity, 1.0)
        
        return leverage <= self.params.max_leverage
    
    def _check_trading_frequency(self) -> bool:
        """Check trading frequency limits"""
        return self.state.daily_trades < self.params.max_trades_per_day
    
    def _check_volatility_exposure(self, atr_value: float, quantity: float) -> bool:
        """Check volatility exposure limits"""
        if atr_value <= 0:
            return True
            
        volatility_exposure = atr_value * quantity
        max_allowed_volatility = self.current_equity * self.params.max_volatility_exposure
        
        return volatility_exposure <= max_allowed_volatility
    
    def _calculate_current_exposure(self) -> float:
        """Calculate current total exposure across all symbols"""
        return sum(self.state.symbol_exposures.values())
    
    def _is_new_day(self, timestamp: int) -> bool:
        """Check if timestamp represents a new trading day"""
        current_dt = datetime.fromtimestamp(timestamp / 1000)  # Assuming ms timestamp
        last_reset_dt = datetime.fromtimestamp(self.state.daily_start_time / 1000)
        
        return current_dt.date() != last_reset_dt.date()
    
    def _reset_daily_metrics(self, timestamp: int) -> None:
        """Reset daily risk metrics"""
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.daily_start_time = timestamp
    
    def _clean_trade_history(self, current_timestamp: int) -> None:
        """Remove trades older than 30 days"""
        thirty_days_ago = current_timestamp - (30 * 24 * 60 * 60 * 1000)  # 30 days in ms
        self.state.trade_history = [
            trade for trade in self.state.trade_history 
            if trade['timestamp'] >= thirty_days_ago
        ]
    
    def _get_daily_timestamp(self) -> int:
        """Get timestamp for start of current day"""
        now = datetime.now()
        start_of_day = datetime(now.year, now.month, now.day)
        return int(start_of_day.timestamp() * 1000)
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        total_exposure = self._calculate_current_exposure()
        leverage = total_exposure / max(self.current_equity, 1.0)
        
        return {
            'current_equity': self.current_equity,
            'daily_pnl': self.state.daily_pnl,
            'daily_trades': self.state.daily_trades,
            'current_drawdown': self.state.current_drawdown,
            'peak_equity': self.state.peak_equity,
            'total_exposure': total_exposure,
            'leverage': leverage,
            'symbol_exposures': self.state.symbol_exposures,
            'sector_exposures': self.state.sector_exposures,
            'remaining_daily_trades': self.params.max_trades_per_day - self.state.daily_trades,
            'remaining_daily_loss': self.params.max_daily_loss + self.state.daily_pnl
        }
    
    def reset(self) -> None:
        """Reset risk manager to initial state"""
        self.__init__(self.initial_equity, self.params)