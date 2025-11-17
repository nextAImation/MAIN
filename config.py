"""
Configuration module for the backtesting engine.
Centralizes all constants, schemas, and settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import enum


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


@dataclass
class DataSchema:
    """Defines the required data schema for input candles"""
    TIMESTAMP: str = "ts"
    OPEN: str = "open"
    HIGH: str = "high"
    LOW: str = "low"
    CLOSE: str = "close"
    VOLUME: str = "volume"
    # Optional bid/ask columns
    BID_OPEN: str = "bid_open"
    BID_HIGH: str = "bid_high"
    BID_LOW: str = "bid_low"
    BID_CLOSE: str = "bid_close"
    ASK_OPEN: str = "ask_open"
    ASK_HIGH: str = "ask_high"
    ASK_LOW: str = "ask_low"
    ASK_CLOSE: str = "ask_close"


@dataclass
class Timeframe:
    """Timeframe configuration"""
    name: str
    minutes: int
    resample_method: str = "last"  # 'last', 'first', 'ohlc'


@dataclass
class RiskParams:
    """Risk management parameters"""
    max_position_size: float = 100.0
    max_portfolio_exposure: float = 0.25  # 25% of equity
    max_daily_loss: float = 0.05  # 5% of equity
    max_drawdown: float = 0.15  # 15% of equity
    sector_exposure_limits: Dict[str, float] = field(default_factory=lambda: {"default": 0.5})
    atr_multiplier_stop: float = 2.0
    atr_multiplier_take_profit: float = 3.0


@dataclass
class ExecutionParams:
    """Execution and matching engine parameters"""
    latency_ms: int = 10
    slippage_model: str = "proportional"  # 'proportional', 'fixed', 'none'
    slippage_bps: float = 1.0  # 1 basis point
    volume_participation_limit: float = 0.1  # 10% of segment volume
    enable_stop_priority: bool = True
    stop_loss_priority: bool = True


from dataclasses import dataclass, field

@dataclass
class BacktestConfig:
    symbol: str = "TEST"
    initial_equity: float = 10000.0

    base_timeframe: Timeframe = field(default_factory=lambda: Timeframe("1m", 1))
    htf_timeframes: List[Timeframe] = field(default_factory=list)

    risk_params: RiskParams = field(default_factory=RiskParams)
    execution_params: ExecutionParams = field(default_factory=ExecutionParams)

    data_schema: DataSchema = field(default_factory=DataSchema)
    validation_mode: ValidationMode = ValidationMode.STRICT

    expected_time_delta: int = 60  # seconds


# Global constants
SUPPORTED_TIMEFRAMES = {
    "1m": Timeframe("1m", 1),
    "5m": Timeframe("5m", 5),
    "15m": Timeframe("15m", 15),
    "1H": Timeframe("1H", 60),
    "4H": Timeframe("4H", 240),
    "1D": Timeframe("1D", 1440),
}

DEFAULT_CONFIG = BacktestConfig()