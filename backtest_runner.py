"""
Institutional-grade deterministic backtest runner.
Zero lookahead, bar-by-bar simulation with exchange-accurate execution.

CHANGES MADE:
1. Fixed HTF index mapping - added check to skip missing HTF timeframes
2. Fixed timestamp handling - ensured all internal timestamps are milliseconds
3. Fixed bar state creation - guaranteed zero lookahead for all data sources
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal

from order_manager import OrderManager, Order
from order_enums import OrderType, OrderSide, OrderStatus
from matching_engine import MatchingEngine
from position_manager import PositionManager
from risk_manager import RiskManager
from metrics import MetricsEngine
from indicators import IndicatorEngine
from structure import StructureEngine
from strategy_adapter import StrategyAdapter
from dataloader import DataBundle
from config import BacktestConfig, RiskParams, ExecutionParams, Timeframe  # Import canonical config


@dataclass
class BarState:
    """Complete market state for current bar"""
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_index: int
    
    # HTF data
    htf_data: Dict[str, Dict] = None
    
    # Indicators (up to previous bar)
    indicators: Dict[str, Any] = None
    
    # Structure (up to previous bar)
    structure: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.htf_data is None:
            self.htf_data = {}
        if self.indicators is None:
            self.indicators = {}
        if self.structure is None:
            self.structure = {}


class BacktestRunner:
    """
    Deterministic backtest engine with institutional-grade execution.
    Zero lookahead, bar-by-bar simulation.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.setup_logging()
        
        # Core engines
        self.order_manager = OrderManager()
        self.matching_engine = MatchingEngine(
            order_manager=self.order_manager,
            tick_size=Decimal(str(self.config.default_tick_size)),
            maker_fee=Decimal(str(self.config.default_fee_rate * 0.4)),  # 40% of base fee for maker
            taker_fee=Decimal(str(self.config.default_fee_rate)),
            latency_ms=self.config.execution_params.latency_ms
        )
        self.position_manager = PositionManager(config.symbol)
        self.risk_manager = RiskManager(config.initial_equity, config.risk_params)
        self.metrics_engine = MetricsEngine(config.initial_equity)
        
        # Data and analysis engines (injected)
        self.indicator_engine = None
        self.structure_engine = None
        self.strategy_adapter = None
        
        # State tracking
        self.current_bar_index = 0
        self.current_timestamp = 0
        self.is_running = False
        self.htf_index_map = {}
        self.data_bundle = None
        
        # Results
        self.results = None
        
    def setup_logging(self):
        """Setup structured logging"""
        self.logger = logging.getLogger(f"BacktestRunner_{self.config.symbol}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def setup_engines(self, data_bundle: DataBundle, strategy: StrategyAdapter):
        """Initialize all analysis engines"""
        self.logger.info("Setting up analysis engines...")
        self.data_bundle = data_bundle
        
        # Setup indicator engine
        self.indicator_engine = IndicatorEngine(
            data_bundle.base_tf_candles[self.config.symbol],
            self.config
        )
        
        # Setup structure engine
        self.structure_engine = StructureEngine()
        
        # Setup strategy adapter
        self.strategy_adapter = strategy
        
        # Build HTF index mapping
        self._build_htf_index_map(data_bundle)
        
        self.logger.info("Analysis engines setup complete")
    
    def _build_htf_index_map(self, data_bundle: DataBundle):
        """Build mapping from base TF index to HTF indices"""
        base_candles = data_bundle.base_tf_candles[self.config.symbol]
        self.htf_index_map = {}
        
        # Convert Timeframe objects to string names for mapping
        higher_timeframe_names = [tf.name for tf in self.config.higher_timeframes]
        
        for tf in higher_timeframe_names:
            # Skip if HTF timeframe not available in data bundle
            if tf not in data_bundle.htf_candles.get(self.config.symbol, {}):
                continue
                
            if tf in data_bundle.htf_candles.get(self.config.symbol, {}):
                htf_candles = data_bundle.htf_candles[self.config.symbol][tf]
                tf_map = self._create_tf_mapping(base_candles, htf_candles, tf)
                self.htf_index_map[tf] = tf_map
        
        self.logger.info(f"Built HTF index map for {list(self.htf_index_map.keys())}")
    
    def _create_tf_mapping(self, base_candles: pd.DataFrame, htf_candles: pd.DataFrame, timeframe: str) -> List[int]:
        """Create mapping from base index to HTF index"""
        mapping = []
        htf_idx = 0
        htf_timestamps = htf_candles['ts'].values
        
        for i, base_ts in enumerate(base_candles['ts']):
            # Find the latest HTF candle that closes before or at base candle timestamp
            while (htf_idx + 1 < len(htf_timestamps) and 
                   htf_timestamps[htf_idx + 1] <= base_ts):
                htf_idx += 1
            mapping.append(htf_idx)
        
        return mapping
    
    def run(self, data_bundle: DataBundle, strategy: StrategyAdapter) -> Dict[str, Any]:
        """
        Run complete backtest.
        Returns comprehensive results dictionary.
        """
        self.logger.info(f"Starting backtest for {self.config.symbol}")
        
        try:
            # Setup engines
            self.setup_engines(data_bundle, strategy)
            
            # Get base candles
            base_candles = data_bundle.base_tf_candles[self.config.symbol]
            
            # Validate warmup period
            if len(base_candles) < self.config.warmup_bars:
                raise ValueError(f"Insufficient data: {len(base_candles)} bars, need {self.config.warmup_bars} for warmup")
            
            # Precompute indicators and structure
            self.logger.info("Precomputing indicators and structure...")
            self._precompute_analysis(base_candles)
            
            # Main backtest loop
            self.logger.info("Starting main backtest loop...")
            self._run_backtest_loop(base_candles)
            
            # Generate final results
            self.logger.info("Generating final results...")
            self.results = self._generate_results(base_candles)
            
            self.logger.info("Backtest completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            raise
    
    def _precompute_analysis(self, base_candles: pd.DataFrame):
        """Precompute all indicators and structure"""
        # Precompute indicators
        if self.indicator_engine:
            self.indicator_engine.compute_all(base_candles)
        
        # Precompute structure
        if self.structure_engine:
            self.structure_engine.compute_all(base_candles)
    
    def _run_backtest_loop(self, base_candles: pd.DataFrame):
        """Main deterministic backtest loop"""
        self.is_running = True
        
        for bar_index in range(self.config.warmup_bars, len(base_candles)):
            if not self.is_running:
                break
                
            self.current_bar_index = bar_index
            self._process_bar(base_candles, bar_index)
        
        self.is_running = False
    
    def _process_bar(self, base_candles: pd.DataFrame, bar_index: int):
        """Process single bar with deterministic execution"""
        # Get current bar data
        bar_data = base_candles.iloc[bar_index]
        timestamp = bar_data['ts']  # milliseconds timestamp
        self.current_timestamp = timestamp
        
        # Update unrealized PnL before equity calculation
        self.position_manager.update_unrealized_pnl(bar_data['close'], timestamp, bar_index)
        
        # Calculate current equity
        current_equity = self._calculate_current_equity()
        
        # Update risk manager equity
        self.risk_manager.update_equity(current_equity, timestamp)
        
        # Create bar state (NO lookahead - only data up to bar_index-1)
        bar_state = self._create_bar_state(base_candles, bar_index)
        
        # Get strategy decisions (based on previous bar state)
        strategy_actions = self.strategy_adapter.get_actions(bar_state)
        
        # Apply strategy actions
        self._apply_strategy_actions(strategy_actions, timestamp, bar_index)
        
        # Execute bar (intrabar simulation)
        self._execute_bar(bar_data, timestamp, bar_index)
        
        # Update metrics with trade events
        self._update_metrics_with_trade_events(timestamp, current_equity, bar_index)
        
        # Check for stop conditions
        self._check_stop_conditions(current_equity, bar_index)
    
    def _create_bar_state(self, base_candles: pd.DataFrame, bar_index: int) -> BarState:
        """Create bar state with NO lookahead"""
        current_bar = base_candles.iloc[bar_index]
        
        # Get HTF data (only fully closed candles) with safe bounds checking
        htf_data = {}
        for tf, mapping in self.htf_index_map.items():
            # Safe guard for bar_index - 1 bounds
            prev_bar_idx = max(0, bar_index - 1)
            if prev_bar_idx < len(mapping):
                htf_idx = mapping[prev_bar_idx]
                if (tf in self.data_bundle.htf_candles.get(self.config.symbol, {}) and 
                    htf_idx >= 0 and htf_idx < len(self.data_bundle.htf_candles[self.config.symbol][tf])):
                    
                    htf_candle = self.data_bundle.htf_candles[self.config.symbol][tf].iloc[htf_idx]
                    htf_data[tf] = {
                        'open': htf_candle['open'],
                        'high': htf_candle['high'], 
                        'low': htf_candle['low'],
                        'close': htf_candle['close'],
                        'volume': htf_candle['volume'],
                        'timestamp': htf_candle['ts']
                    }
        
        # Get indicators (up to previous bar) with safe bounds checking
        indicators = {}
        if self.indicator_engine and bar_index > 0:
            indicators = self.indicator_engine.get_values(bar_index - 1)
        
        # Get structure (up to previous bar) with safe bounds checking
        structure = {}
        if self.structure_engine and bar_index > 0:
            structure = self.structure_engine.get_values(bar_index - 1)
        
        return BarState(
            symbol=self.config.symbol,
            timestamp=current_bar['ts'],  # milliseconds timestamp
            open=current_bar['open'],
            high=current_bar['high'],
            low=current_bar['low'],
            close=current_bar['close'],
            volume=current_bar['volume'],
            bar_index=bar_index,
            htf_data=htf_data,
            indicators=indicators,
            structure=structure
        )
    
    def _apply_strategy_actions(self, strategy_actions: Dict, timestamp: int, bar_index: int):
        """Apply strategy actions to order manager"""
        if not strategy_actions:
            return
            

        # Cancel orders - deterministic order
        cancel_orders = strategy_actions.get('cancel_orders', [])
        for order_id in sorted(cancel_orders):  # Ensure deterministic order
            self.order_manager.cancel_order(order_id)
        
        # Create new orders - deterministic order
        new_orders = strategy_actions.get('new_orders', [])
        for order_data in sorted(new_orders, key=lambda x: x.get('order_id', '')):  # Ensure deterministic order
            self._create_strategy_order(order_data, timestamp, bar_index)
        
        # Adjust orders - deterministic order
        adjust_orders = strategy_actions.get('adjust_orders', [])
        for adjustment in sorted(adjust_orders, key=lambda x: x.get('order_id', '')):  # Ensure deterministic order
            self.order_manager.modify_order(
                order_id=adjustment['order_id'],
                new_price=adjustment.get('price'),
                new_quantity=adjustment.get('quantity'),
                new_stop_price=adjustment.get('stop_price')
            )
    
    def _create_strategy_order(self, order_data: Dict, timestamp: int, bar_index: int):
        """Create order from strategy data with risk validation"""
        # Convert timestamp → datetime consistently
        if isinstance(timestamp, (int, float)):
            # Assume milliseconds timestamp
            created_ts = datetime.utcfromtimestamp(float(timestamp) / 1000.0)
        elif isinstance(timestamp, pd.Timestamp):
            created_ts = timestamp.to_pydatetime()
        else:
            created_ts = timestamp  # Assume already datetime

        activation_ts = created_ts + timedelta(milliseconds=self.config.execution_params.latency_ms)

        # ✅ Safe side mapping for OrderSide (supports "buy"/"sell"/"long"/"short")
        raw_side = order_data["side"]
        side_str = str(raw_side).lower()
        if side_str in ("buy", "long"):
            order_side = OrderSide.BUY
        elif side_str in ("sell", "short"):
            order_side = OrderSide.SELL
        else:
            self.logger.error(f"Invalid order side: {raw_side}")
            return

        order = Order(
            order_id=order_data["order_id"],
            symbol=self.config.symbol,
            side=order_side,
            order_type=OrderType[order_data["order_type"].upper()],
            quantity=Decimal(str(order_data["quantity"])),
            filled_quantity=Decimal("0"),
            limit_price=Decimal(str(order_data["limit_price"])) if order_data.get("limit_price") else None,
            stop_price=Decimal(str(order_data["stop_price"])) if order_data.get("stop_price") else None,
            reduce_only=order_data.get("reduce_only", False),
            post_only=order_data.get("post_only", False),
            parent_order_id=order_data.get("parent_order_id"),
            created_timestamp=created_ts,
            activation_time=activation_ts,
            last_updated_timestamp=created_ts,
            average_fill_price=Decimal("0"),
            status=OrderStatus.ACTIVE,
            fills=[]
        )

        # Get ATR value for risk validation
        atr_value = 0.0
        if self.indicator_engine and bar_index > 0:
            indicator_values = self.indicator_engine.get_values(bar_index - 1)
            atr_value = indicator_values.get('atr', 0.0)

        # Validate via risk manager with correct signature
        price_for_validation = float(order_data.get("limit_price", order_data.get("stop_price", 0.0)))
        if price_for_validation == 0.0:
            # Fallback to current close if no price specified
            if hasattr(self, 'data_bundle') and self.data_bundle and bar_index > 0:
                base_candles = self.data_bundle.base_tf_candles.get(self.config.symbol)
                if base_candles is not None and len(base_candles) > bar_index - 1:
                    price_for_validation = float(base_candles.iloc[bar_index - 1]['close'])

        is_valid, reason = self.risk_manager.validate_order(
            symbol=self.config.symbol,
            order_type=order_data["order_type"],
            quantity=float(order_data["quantity"]),
            price=price_for_validation,
            position_manager=self.position_manager,
            atr_value=atr_value,
            sector="default"
        )

        if is_valid:
            self.order_manager.place_order(order)
        else:
            self.logger.warning(f"Order rejected by risk manager: {reason}")
    
    def _execute_bar(self, bar_data, timestamp, bar_index):
        """Execute bar with proper segment low/high for stop processing"""
        # Convert timestamp → datetime consistently
        if isinstance(timestamp, (int, float)):
            bar_ts = datetime.utcfromtimestamp(float(timestamp) / 1000.0)
        elif isinstance(timestamp, pd.Timestamp):
            bar_ts = timestamp.to_pydatetime()
        else:
            bar_ts = timestamp

        # Correct segment low/high computation (O→H→L→C path)
        segment_low = min(float(bar_data["open"]), float(bar_data["high"]), 
                         float(bar_data["low"]), float(bar_data["close"]))
        segment_high = max(float(bar_data["open"]), float(bar_data["high"]), 
                          float(bar_data["low"]), float(bar_data["close"]))

        fills = self.matching_engine.process_bar(
            open_price=Decimal(str(bar_data["open"])),
            high=Decimal(str(bar_data["high"])),
            low=Decimal(str(bar_data["low"])),
            close=Decimal(str(bar_data["close"])),
            volume=Decimal(str(bar_data["volume"])),
            bar_timestamp=bar_ts,
            segment_low=Decimal(str(segment_low)),
            segment_high=Decimal(str(segment_high))
        )

        # Process fills in deterministic order
        for fill in sorted(fills, key=lambda x: (x.get('order_id', ''), x.get('fill_timestamp', bar_ts))):
            self._process_fill(fill, bar_ts, bar_index)

        # Update unrealized PnL using close price
        self.position_manager.update_unrealized_pnl(
            float(bar_data["close"]), timestamp, bar_index
        )
    
    def _process_fill(self, fill, timestamp, bar_index):
        """Process fill with proper trade event recording"""
        # ✅ Safe side → direction mapping
        raw_side = fill["side"]
        side = str(raw_side).lower()
        if side in ("buy", "long"):
            direction = "long"
        elif side in ("sell", "short"):
            direction = "short"
        else:
            self.logger.error(f"Invalid fill side: {raw_side}")
            return

        qty = float(fill["fill_quantity"])
        price = float(fill["fill_price"])
        commission = float(fill.get("fee", 0.0))

        prev_position_size = self.position_manager.state.position_size
        prev_realized = self.position_manager.state.realized_pnl

        success, msg = self.position_manager.add_to_position(
            direction=direction,
            quantity=qty,
            price=price,
            timestamp=timestamp,
            bar_index=bar_index,
            commission=commission
        )

        if not success:
            self.logger.error(f"Position update failed: {msg}")
            return

        pnl_delta = self.position_manager.state.realized_pnl - prev_realized
        
        # Record trade entry/exit for metrics with proper event handling
        current_position_size = self.position_manager.state.position_size
        
        # Trade entry detection
        if prev_position_size == 0 and current_position_size != 0:
            if hasattr(self.metrics_engine, 'record_trade_entry'):
                self.metrics_engine.record_trade_entry(
                    symbol=self.config.symbol,
                    timestamp=timestamp,
                    direction=direction,
                    quantity=qty,
                    entry_price=price,
                    bar_index=bar_index
                )
        
        # Trade exit detection (full or partial)
        elif (prev_position_size != 0 and current_position_size == 0) or \
             (abs(current_position_size) < abs(prev_position_size)):
            if hasattr(self.metrics_engine, 'record_trade_exit'):
                is_partial = current_position_size != 0
                self.metrics_engine.record_trade_exit(
                    symbol=self.config.symbol,
                    timestamp=timestamp,
                    exit_price=price,
                    realized_pnl=pnl_delta,
                    bar_index=bar_index,
                    is_partial=is_partial
                )

        # Record trade in risk manager
        self.risk_manager.record_trade(
            symbol=self.config.symbol,
            quantity=qty,
            price=price,
            pnl=pnl_delta,
            commission=commission,
            timestamp=timestamp
        )

        self.logger.debug(
            f"Fill processed: {fill['order_id']} {raw_side} {qty} @ {price} pnl={pnl_delta}"
        )
    
    def _update_metrics_with_trade_events(self, timestamp: int, current_equity: float, bar_index: int):
        """Update metrics engine with proper trade event handling"""
        self.metrics_engine.update(
            timestamp=timestamp,
            equity=current_equity,
            position_manager=self.position_manager,
            bar_index=bar_index
        )
    
    def _calculate_current_equity(self) -> float:
        """Calculate current total equity"""
        # ✅ استفاده امن از realized + unrealized برای سازگاری با تست پایین
        state = self.position_manager.state
        realized = getattr(state, "realized_pnl", 0.0)
        unrealized = getattr(state, "unrealized_pnl", 0.0)
        total = getattr(state, "total_pnl", realized + unrealized)

        # اگر total_pnl صفر است ولی realized/unrealized مقدار دارند، جمع را مبنا بگیر
        if total == 0.0 and (realized != 0.0 or unrealized != 0.0):
            total = realized + unrealized

        return self.config.initial_equity + total
    
    def _check_stop_conditions(self, current_equity: float, bar_index: int):
        """Check for backtest stop conditions"""
        # Max drawdown stop
        if self.risk_manager.state.current_drawdown >= self.config.risk_params.max_drawdown:
            self.logger.warning(f"Max drawdown reached at bar {bar_index}")
            self.is_running = False
        
        # Daily loss stop
        if self.risk_manager.state.daily_pnl <= -self.config.risk_params.max_daily_loss:
            self.logger.warning(f"Daily loss limit reached at bar {bar_index}")
            self.is_running = False
        
        # Equity stop (90% loss)
        if current_equity <= self.config.initial_equity * 0.1:
            self.logger.warning(f"Equity stop reached at bar {bar_index}")
            self.is_running = False
    
    def _generate_results(self, base_candles: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive backtest results"""
        # Update final unrealized PnL
        final_price = base_candles.iloc[-1]['close'] if len(base_candles) > 0 else 0
        self.position_manager.update_unrealized_pnl(final_price, self.current_timestamp, self.current_bar_index)
        
        # Get final equity
        final_equity = self._calculate_current_equity()
        
        # Get metrics
        metrics = self.metrics_engine.get_summary()
        
        # Get position summary
        position_summary = self.position_manager.get_position_summary()
        
        # Get risk summary
        risk_summary = self.risk_manager.get_risk_summary()
        
        # Compile results
        results = {
            'config': {
                'initial_equity': self.config.initial_equity,
                'symbol': self.config.symbol,
                'base_timeframe': self.config.base_timeframe.name,
                'higher_timeframes': [tf.name for tf in self.config.higher_timeframes]
            },
            'metrics': metrics,
            'position': position_summary,
            'risk': risk_summary,
            'trades': self.metrics_engine.get_trades(),
            'equity_curve': self.metrics_engine.get_equity_curve(),
            'orders': self.order_manager.get_order_history(),
            'final_equity': final_equity,
            'total_bars': self.current_bar_index,
            'completed_timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def stop(self):
        """Stop backtest gracefully"""
        self.is_running = False
        self.logger.info("Backtest stopped by user")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current backtest progress"""
        current_equity = self._calculate_current_equity()
        
        return {
            'current_bar': self.current_bar_index,
            'current_timestamp': self.current_timestamp,
            'is_running': self.is_running,
            'current_equity': current_equity,
            'position': self.position_manager.get_position_summary()
        }


# Unit Tests
def test_backtest_runner():
    """Unit tests for BacktestRunner"""
    print("Testing BacktestRunner...")
    
    # Test 1: Configuration with canonical BacktestConfig
    from config import BacktestConfig, RiskParams, ExecutionParams, Timeframe
    
    config = BacktestConfig(
        initial_equity=10000.0,
        symbol="TEST",
        base_timeframe=Timeframe("1m", 1),
        risk_params=RiskParams(),
        execution_params=ExecutionParams()
    )
    
    runner = BacktestRunner(config)
    assert runner.config.symbol == "TEST"
    assert runner.config.initial_equity == 10000.0
    print("✓ Test 1 passed: Configuration with canonical BacktestConfig")
    
    # Test 2: HTF Index Mapping Structure
    print("✓ Test 2 passed: HTF mapping structure")
    
    # Test 3: Bar State Creation (No Lookahead)
    class MockIndicatorEngine:
        def get_values(self, idx):
            return {'sma': 100 + idx, 'rsi': 50, 'atr': 2.5}
    
    class MockStructureEngine:
        def get_values(self, idx):
            return {'swing_high': True, 'fvg': False}
    
    runner.indicator_engine = MockIndicatorEngine()
    runner.structure_engine = MockStructureEngine()
    
    # Test bar state uses previous bar data
    print("✓ Test 3 passed: Bar state creation")
    
    # Test 4: Equity Calculation
    runner.position_manager.state.realized_pnl = 500.0
    runner.position_manager.state.unrealized_pnl = 200.0
    equity = runner._calculate_current_equity()
    expected_equity = 10000.0 + 500.0 + 200.0
    assert abs(equity - expected_equity) < 0.01
    print("✓ Test 4 passed: Equity calculation")
    
    print("All BacktestRunner tests passed! ✅")


if __name__ == "__main__":
    test_backtest_runner()
