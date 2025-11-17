"""
Order Enums Module - Centralized enumeration definitions for order management.
Compatible with order_manager.py, matching_engine.py, and BacktestRunner.
"""

from enum import Enum

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class TimeInForce(Enum):
    """Time in force enumeration"""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill

class TriggerType(Enum):
    """Trigger type enumeration"""
    LAST = "LAST"      # Last price trigger
    INDEX = "INDEX"    # Index price trigger
    MARK = "MARK"      # Mark price trigger

class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"

class FillType(Enum):
    """Fill type enumeration"""
    MAKER = "MAKER"
    TAKER = "TAKER"
    LIQUIDATION = "LIQUIDATION"

class ExecutionClass(Enum):
    """Execution class enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"