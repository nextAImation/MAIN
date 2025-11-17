"""
Configuration module for the backtesting engine.
Canonical version – compatible with BacktestRunner.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import enum


# ------------------------------------------------------------
# ENUMS
# ------------------------------------------------------------

class ContractType(enum.Enum):
    LINEAR = "linear"
    INVERSE = "inverse"


class ValidationMode(enum.Enum):
    STRICT = "strict"
    SOFT = "soft"
    NONE = "none"


class OrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


# ------------------------------------------------------------
# DATA SCHEMA
# ------------------------------------------------------------

@dataclass
class DataSchema:
    TIMESTAMP: str = "ts"
    OPEN: str = "open"
    HIGH: str = "high"
    LOW: str = "low"
    CLOSE: str = "close"
    VOLUME: str = "volume"

    # Optional bid/ask columns used by DataLoader
    BID_OPEN: str = "bid_open"
    BID_HIGH: str = "bid_high"
    BID_LOW: str = "bid_low"
    BID_CLOSE: str = "bid_close"

    ASK_OPEN: str = "ask_open"
    ASK_HIGH: str = "ask_high"
    ASK_LOW: str = "ask_low"
    ASK_CLOSE: str = "ask_close"


# ------------------------------------------------------------
# TIMEFRAME
# ------------------------------------------------------------

@dataclass
class Timeframe:
    name: str
    minutes: int
    resample_method: str = "last"

    def to_seconds(self):
        return self.minutes * 60


# ------------------------------------------------------------
# RISK PARAMETERS
# ------------------------------------------------------------

@dataclass
class RiskParams:
    """Risk management parameters (aligned with RiskManager expectations)"""

    # --- Position / exposure limits ---
    max_position_size: float = 100.0                # per-position size (units)
    max_open_exposure: float = 10000.0              # notional exposure cap
    max_portfolio_exposure: float = 0.25            # 25% of equity as fraction

    # --- Loss / drawdown limits ---
    max_loss_per_trade: float = 500.0               # per-trade monetary loss limit
    max_daily_loss: float = 0.05                    # 5% of equity as fraction
    max_drawdown: float = 0.15                      # 15% of equity as fraction

    # --- Concentration limits ---
    max_symbol_concentration: float = 0.3           # 30% of portfolio in one symbol
    max_sector_concentration: float = 0.5           # 50% in one sector
    sector_exposure_limits: Dict[str, float] = field(
        default_factory=lambda: {"default": 0.5}
    )

    # --- Leverage / trade count / order size ---
    max_leverage: float = 10.0
    max_trades_per_day: int = 100
    max_order_value: float = 50000.0                # notional per order

    # --- Volatility / risk factor limits ---
    max_volatility_exposure: float = 1.0            # dummy cap, used by RiskManager if needed

    # --- Strategy-specific ATR-based exits ---
    atr_multiplier_stop: float = 2.0
    atr_multiplier_take_profit: float = 3.0


# ------------------------------------------------------------
# EXECUTION PARAMETERS
# ------------------------------------------------------------

@dataclass
class ExecutionParams:
    latency_ms: int = 10
    slippage_model: str = "proportional"
    slippage_bps: float = 1.0
    volume_participation_limit: float = 0.1
    enable_stop_priority: bool = True
    stop_loss_priority: bool = True


# ------------------------------------------------------------
# BACKTEST CONFIG – FINAL CANONICAL VERSION
# ------------------------------------------------------------

@dataclass
class BacktestConfig:

    # Basic settings
    symbol: str = "TEST"
    initial_equity: float = 10000.0

    # Timeframes
    base_timeframe: Timeframe = field(default_factory=lambda: Timeframe("1m", 1))
    htf_timeframes: List[Timeframe] = field(default_factory=list)

    # Runner compatibility
    higher_timeframes: List[Timeframe] = field(default_factory=list)

    # Other configs
    risk_params: RiskParams = field(default_factory=RiskParams)
    execution_params: ExecutionParams = field(default_factory=ExecutionParams)
    data_schema: DataSchema = field(default_factory=DataSchema)
    validation_mode: ValidationMode = ValidationMode.STRICT

    expected_time_delta: int = 60000
    warmup_bars: int = 10

    # Defaults required by Runner + DataLoader
    default_tick_size: float = 0.01
    default_contract_type: ContractType = ContractType.LINEAR
    default_fee_rate: float = 0.0004

    def __post_init__(self):
        """
        Fix compatibility issues after object creation.
        """
        # Sync htf_timeframes → higher_timeframes for BacktestRunner
        if not self.higher_timeframes:
            self.higher_timeframes = self.htf_timeframes


# ------------------------------------------------------------
# SUPPORTED TIMEFRAMES
# ------------------------------------------------------------

SUPPORTED_TIMEFRAMES = {
    "1m": Timeframe("1m", 1),
    "5m": Timeframe("5m", 5),
    "15m": Timeframe("15m", 15),
    "1H": Timeframe("1H", 60),
    "4H": Timeframe("4H", 240),
    "1D": Timeframe("1D", 1440),
}


# Do NOT add alias here — aliasing is handled safely inside __post_init__()
