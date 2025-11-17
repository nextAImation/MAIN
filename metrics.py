"""
Institutional-grade metrics engine with MAE/MFE tracking.
Deterministic performance calculation with full trade analytics.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime
import json


@dataclass
class TradeRecord:
    """Complete trade record for analytics"""
    trade_id: str
    symbol: str
    entry_timestamp: int
    exit_timestamp: int
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    pnl: float
    pnl_pct: float
    commission: float
    net_pnl: float
    duration_bars: int
    duration_seconds: int
    entry_reason: str
    exit_reason: str
    tags: List[str]
    mae: float  # Maximum Adverse Excursion
    mfe: float  # Maximum Favorable Excursion
    avg_entry_price: float
    avg_exit_price: float
    max_drawdown_trade: float
    runup: float


@dataclass 
class EquityPoint:
    """Equity curve data point"""
    timestamp: int
    bar_index: int
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown: float
    drawdown_pct: float


class MetricsEngine:
    """
    Comprehensive performance metrics with institutional-grade analytics.
    Tracks MAE/MFE, drawdown, risk-adjusted returns, and trade quality.
    """
    
    def __init__(self, initial_equity: float = 10000.0):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        
        # Core tracking
        self.equity_curve: List[EquityPoint] = []
        self.trades: List[TradeRecord] = []
        self.open_trades: Dict[str, TradeRecord] = {}
        
        # Performance metrics
        self.peak_equity = initial_equity
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.total_commission = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Risk metrics
        self.volatility = 0.0
        self.sharpe_ratio = 0.0
        self.calmar_ratio = 0.0
        self.sortino_ratio = 0.0
        
        # Trade analytics
        self.avg_trade_pnl = 0.0
        self.avg_winning_trade = 0.0
        self.avg_losing_trade = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.expectancy = 0.0
        
        # MAE/MFE tracking
        self.avg_mae = 0.0
        self.avg_mfe = 0.0
        self.mfe_mae_ratio = 0.0
        
    def update(self, timestamp: int, equity: float, position_manager, bar_index: int) -> None:
        """
        Update metrics with current state.
        Zero lookahead - only uses completed data.
        """
        self.current_equity = equity
        
        # Update peak equity and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        drawdown = self.peak_equity - equity
        drawdown_pct = drawdown / self.peak_equity if self.peak_equity > 0 else 0.0
        
        self.max_drawdown = max(self.max_drawdown, drawdown)
        self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown_pct)
        
        # Record equity point
        equity_point = EquityPoint(
            timestamp=timestamp,
            bar_index=bar_index,
            equity=equity,
            unrealized_pnl=position_manager.state.unrealized_pnl,
            realized_pnl=position_manager.state.realized_pnl,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct
        )
        self.equity_curve.append(equity_point)
        
        # Update open trades MAE/MFE
        self._update_open_trades_mae_mfe(position_manager)
        
        # Close completed trades
        self._close_completed_trades(position_manager, timestamp, bar_index)
        
        # Update performance metrics periodically
        if len(self.trades) > 0 and len(self.trades) % 10 == 0:
            self._update_performance_metrics()
    
    def record_trade_entry(self, symbol: str, timestamp: int, bar_index: int, 
                          entry_price: float, quantity: float, side: str, 
                          entry_reason: str, tags: List[str] = None) -> str:
        """
        Record new trade entry.
        Returns trade_id for future reference.
        """
        trade_id = f"{symbol}_{timestamp}_{len(self.open_trades)}"
        
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            entry_timestamp=timestamp,
            exit_timestamp=0,
            entry_bar=bar_index,
            exit_bar=0,
            entry_price=entry_price,
            exit_price=0.0,
            quantity=quantity,
            side=side,
            pnl=0.0,
            pnl_pct=0.0,
            commission=0.0,
            net_pnl=0.0,
            duration_bars=0,
            duration_seconds=0,
            entry_reason=entry_reason,
            exit_reason="",
            tags=tags or [],
            mae=0.0,
            mfe=0.0,
            avg_entry_price=entry_price,
            avg_exit_price=0.0,
            max_drawdown_trade=0.0,
            runup=0.0
        )
        
        self.open_trades[trade_id] = trade
        return trade_id
    
    def record_trade_exit(self, trade_id: str, timestamp: int, bar_index: int,
                         exit_price: float, quantity: float, pnl: float,
                         commission: float, exit_reason: str) -> bool:
        """
        Record trade exit and calculate final metrics.
        Returns success status.
        """
        if trade_id not in self.open_trades:
            return False
            
        trade = self.open_trades[trade_id]
        
        # Update trade with exit data
        trade.exit_timestamp = timestamp
        trade.exit_bar = bar_index
        trade.exit_price = exit_price
        trade.quantity = quantity
        trade.pnl = pnl
        trade.commission = commission
        trade.net_pnl = pnl - commission
        trade.exit_reason = exit_reason
        
        # Calculate percentages and durations
        trade.pnl_pct = (pnl / (trade.entry_price * quantity)) * 100 if trade.entry_price > 0 else 0.0
        trade.duration_bars = bar_index - trade.entry_bar
        trade.duration_seconds = timestamp - trade.entry_timestamp
        
        # Move to completed trades
        self.trades.append(trade)
        del self.open_trades[trade_id]
        
        # Update counters
        self.total_trades += 1
        self.total_commission += commission
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        return True
    
    def _update_open_trades_mae_mfe(self, position_manager):
        """Update MAE/MFE for open trades based on position manager lots"""
        for trade_id, trade in self.open_trades.items():
            if trade.side == 'long':
                # For long positions, calculate current PnL from entry
                if position_manager.state.direction == 'long' and position_manager.state.lots:
                    # Use average entry price from position manager
                    current_price = position_manager.state.avg_entry_price + position_manager.state.unrealized_pnl / position_manager.state.quantity
                    current_pnl_pct = (current_price - trade.entry_price) / trade.entry_price * 100
                    
                    trade.mfe = max(trade.mfe, current_pnl_pct)
                    trade.mae = min(trade.mae, current_pnl_pct)
                    trade.runup = max(trade.runup, current_pnl_pct)
                    trade.max_drawdown_trade = min(trade.max_drawdown_trade, current_pnl_pct - trade.runup)
                    
            elif trade.side == 'short':
                # For short positions
                if position_manager.state.direction == 'short' and position_manager.state.lots:
                    current_price = position_manager.state.avg_entry_price - position_manager.state.unrealized_pnl / position_manager.state.quantity
                    current_pnl_pct = (trade.entry_price - current_price) / trade.entry_price * 100
                    
                    trade.mfe = max(trade.mfe, current_pnl_pct)
                    trade.mae = min(trade.mae, current_pnl_pct)
                    trade.runup = max(trade.runup, current_pnl_pct)
                    trade.max_drawdown_trade = min(trade.max_drawdown_trade, current_pnl_pct - trade.runup)
    
    def _close_completed_trades(self, position_manager, timestamp: int, bar_index: int):
        """Detect and close completed trades based on position state"""
        if position_manager.state.direction == 'flat' and self.open_trades:
            # Position closed, close all open trades
            for trade_id in list(self.open_trades.keys()):
                trade = self.open_trades[trade_id]
                # Use position manager's realized PnL for this trade
                pnl = position_manager.state.realized_pnl / len(self.open_trades)  # Approximate
                self.record_trade_exit(
                    trade_id, timestamp, bar_index,
                    position_manager.state.avg_entry_price, trade.quantity,
                    pnl, 0.0, "position_closed"
                )
    
    def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        if not self.trades:
            return
            
        # Basic trade statistics
        winning_pnls = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losing_pnls = [t.net_pnl for t in self.trades if t.net_pnl <= 0]
        
        self.avg_trade_pnl = np.mean([t.net_pnl for t in self.trades])
        self.avg_winning_trade = np.mean(winning_pnls) if winning_pnls else 0.0
        self.avg_losing_trade = np.mean(losing_pnls) if losing_pnls else 0.0
        self.largest_win = max(winning_pnls) if winning_pnls else 0.0
        self.largest_loss = min(losing_pnls) if losing_pnls else 0.0
        self.win_rate = len(winning_pnls) / len(self.trades)
        
        # Risk-adjusted metrics
        total_gross_profit = sum(winning_pnls)
        total_gross_loss = abs(sum(losing_pnls))
        self.profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf')
        
        self.expectancy = (self.win_rate * self.avg_winning_trade + 
                          (1 - self.win_rate) * self.avg_losing_trade)
        
        # MAE/MFE statistics
        maes = [t.mae for t in self.trades if t.mae != 0.0]
        mfes = [t.mfe for t in self.trades if t.mfe != 0.0]
        
        self.avg_mae = np.mean(maes) if maes else 0.0
        self.avg_mfe = np.mean(mfes) if mfes else 0.0
        self.mfe_mae_ratio = abs(self.avg_mfe / self.avg_mae) if self.avg_mae != 0 else 0.0
        
        # Calculate volatility and ratios (simplified)
        returns = [t.pnl_pct / 100 for t in self.trades]  # Convert to decimal
        if returns:
            self.volatility = np.std(returns)
            avg_return = np.mean(returns)
            self.sharpe_ratio = avg_return / self.volatility if self.volatility > 0 else 0.0
            self.calmar_ratio = avg_return / self.max_drawdown_pct if self.max_drawdown_pct > 0 else 0.0
            
            # Sortino ratio (downside deviation only)
            downside_returns = [r for r in returns if r < 0]
            downside_dev = np.std(downside_returns) if downside_returns else 0.0
            self.sortino_ratio = avg_return / downside_dev if downside_dev > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        total_net_pnl = sum(t.net_pnl for t in self.trades)
        total_return_pct = (total_net_pnl / self.initial_equity) * 100
        
        return {
            'initial_equity': self.initial_equity,
            'final_equity': self.current_equity,
            'total_net_pnl': total_net_pnl,
            'total_return_pct': total_return_pct,
            'total_commission': self.total_commission,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_trade_pnl': self.avg_trade_pnl,
            'avg_winning_trade': self.avg_winning_trade,
            'avg_losing_trade': self.avg_losing_trade,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'avg_mae': self.avg_mae,
            'avg_mfe': self.avg_mfe,
            'mfe_mae_ratio': self.mfe_mae_ratio,
            'open_trades': len(self.open_trades)
        }
    
    def get_trades(self) -> List[Dict]:
        """Get all trades as dictionaries"""
        return [asdict(trade) for trade in self.trades]
    
    def get_equity_curve(self) -> List[Dict]:
        """Get equity curve as dictionaries"""
        return [asdict(point) for point in self.equity_curve]
    
    def get_trade_analytics(self) -> Dict[str, Any]:
        """Get detailed trade analytics"""
        if not self.trades:
            return {}
            
        # Trade duration analysis
        durations = [t.duration_bars for t in self.trades]
        durations_seconds = [t.duration_seconds for t in self.trades]
        
        # Reason analysis
        entry_reasons = {}
        exit_reasons = {}
        for trade in self.trades:
            entry_reasons[trade.entry_reason] = entry_reasons.get(trade.entry_reason, 0) + 1
            exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1
        
        # Time-based analysis
        hourly_pnl = {}
        daily_pnl = {}
        for trade in self.trades:
            hour = datetime.fromtimestamp(trade.exit_timestamp/1000).hour
            day = datetime.fromtimestamp(trade.exit_timestamp/1000).strftime('%Y-%m-%d')
            hourly_pnl[hour] = hourly_pnl.get(hour, 0) + trade.net_pnl
            daily_pnl[day] = daily_pnl.get(day, 0) + trade.net_pnl
        
        return {
            'durations': {
                'avg_bars': np.mean(durations),
                'avg_seconds': np.mean(durations_seconds),
                'min_bars': np.min(durations) if durations else 0,
                'max_bars': np.max(durations) if durations else 0
            },
            'entry_reasons': entry_reasons,
            'exit_reasons': exit_reasons,
            'hourly_performance': hourly_pnl,
            'daily_performance': daily_pnl,
            'consecutive_wins': self._calculate_consecutive_wins(),
            'consecutive_losses': self._calculate_consecutive_losses()
        }
    
    def _calculate_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning trades"""
        if not self.trades:
            return 0
            
        max_streak = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade.net_pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                
        return max_streak
    
    def _calculate_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades"""
        if not self.trades:
            return 0
            
        max_streak = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade.net_pnl <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                
        return max_streak
    
    def reset(self):
        """Reset metrics engine to initial state"""
        self.__init__(self.initial_equity)


# Unit Tests
def test_metrics_engine():
    """Unit tests for MetricsEngine"""
    print("Testing MetricsEngine...")
    
    # Test 1: Initialization
    metrics = MetricsEngine(10000.0)
    assert metrics.initial_equity == 10000.0
    assert metrics.current_equity == 10000.0
    assert len(metrics.trades) == 0
    print("✓ Test 1 passed: Initialization")
    
    # Test 2: Equity Tracking
    class MockPositionManager:
        def __init__(self):
            self.state = type('State', (), {
                'unrealized_pnl': 100.0,
                'realized_pnl': 500.0,
                'direction': 'long',
                'quantity': 1.0,
                'avg_entry_price': 50000.0,
                'lots': [type('Lot', (), {'entry_price': 50000.0})]
            })()
    
    pm = MockPositionManager()
    metrics.update(1000, 10600.0, pm, 1)
    
    assert len(metrics.equity_curve) == 1
    assert metrics.equity_curve[0].equity == 10600.0
    print("✓ Test 2 passed: Equity tracking")
    
    # Test 3: Trade Recording
    trade_id = metrics.record_trade_entry(
        symbol="BTCUSDT",
        timestamp=1000,
        bar_index=1,
        entry_price=50000.0,
        quantity=1.0,
        side="long",
        entry_reason="breakout",
        tags=["momentum"]
    )
    
    assert trade_id in metrics.open_trades
    assert metrics.open_trades[trade_id].symbol == "BTCUSDT"
    print("✓ Test 3 passed: Trade entry recording")
    
    # Test 4: Trade Exit
    success = metrics.record_trade_exit(
        trade_id=trade_id,
        timestamp=2000,
        bar_index=10,
        exit_price=51000.0,
        quantity=1.0,
        pnl=1000.0,
        commission=5.0,
        exit_reason="target"
    )
    
    assert success
    assert len(metrics.trades) == 1
    assert metrics.trades[0].net_pnl == 995.0  # 1000 - 5
    assert metrics.total_trades == 1
    assert metrics.winning_trades == 1
    print("✓ Test 4 passed: Trade exit recording")
    
    # Test 5: Performance Summary
    summary = metrics.get_summary()
    assert summary['total_trades'] == 1
    assert summary['winning_trades'] == 1
    assert summary['win_rate'] == 1.0
    print("✓ Test 5 passed: Performance summary")
    
    print("All MetricsEngine tests passed! ✅")

if __name__ == "__main__":
    test_metrics_engine()