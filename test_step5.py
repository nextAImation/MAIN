"""
Combined unit tests for PositionManager and RiskManager
"""

import numpy as np
from position_manager import PositionManager, PositionState, PositionLot
from risk_manager import RiskManager, RiskParameters, RiskState


def test_position_manager():
    """Unit tests for PositionManager"""
    print("Testing PositionManager...")
    
    # Test 1: Open long position
    pm = PositionManager("BTCUSDT", 0.01)
    success, msg = pm.add_to_position('long', 1.0, 50000.0, 1000, 1)
    assert success, f"Failed to open long: {msg}"
    assert pm.state.direction == 'long'
    assert pm.state.quantity == 1.0
    assert pm.state.avg_entry_price == 50000.0
    print("✓ Test 1 passed: Open long position")
    
    # Test 2: Add to long position
    success, msg = pm.add_to_position('long', 0.5, 51000.0, 1001, 2)
    assert success, f"Failed to add to long: {msg}"
    assert pm.state.quantity == 1.5
    assert abs(pm.state.avg_entry_price - 50333.33) < 0.1
    print("✓ Test 2 passed: Add to long position")
    
    # Test 3: Reduce long position
    pm.update_unrealized_pnl(52000.0, 1002, 3)
    success, msg = pm.reduce_position(0.5, 52000.0, 1002, 3)
    assert success, f"Failed to reduce long: {msg}"
    assert pm.state.quantity == 1.0
    assert pm.state.realized_pnl > 0
    print("✓ Test 3 passed: Reduce long position")
    
    # Test 4: Close position
    success, msg = pm.close_position(53000.0, 1003, 4)
    assert success, f"Failed to close: {msg}"
    assert pm.state.direction == 'flat'
    assert pm.state.quantity == 0.0
    print("✓ Test 4 passed: Close position")
    
    # Test 5: Reverse position (long to short)
    pm.add_to_position('long', 1.0, 50000.0, 1004, 5)
    success, msg = pm.add_to_position('short', 1.0, 49000.0, 1005, 6)
    assert success, f"Failed to reverse: {msg}"
    assert pm.state.direction == 'short'
    print("✓ Test 5 passed: Reverse position")
    
    # Test 6: MAE/MFE tracking
    pm = PositionManager("BTCUSDT", 0.01)
    pm.add_to_position('long', 1.0, 50000.0, 1000, 1)
    
    # Simulate price movements
    pm.update_unrealized_pnl(48000.0, 1001, 2)  # Adverse move
    pm.update_unrealized_pnl(52000.0, 1002, 3)  # Favorable move
    pm.update_unrealized_pnl(49000.0, 1003, 4)  # Mixed
    
    mfe_summary = pm.get_mae_mfe_summary()
    assert mfe_summary['mae'] < 0, "MAE should be negative"
    assert mfe_summary['mfe'] > 0, "MFE should be positive"
    print("✓ Test 6 passed: MAE/MFE tracking")
    
    print("All PositionManager tests passed! ✅")


def test_risk_manager():
    """Unit tests for RiskManager"""
    print("Testing RiskManager...")
    
    # Test setup
    risk_params = RiskParameters(
        max_position_size=10.0,
        max_order_value=50000.0,  # Increased to allow BTC test
        max_open_exposure=10000.0,
        max_loss_per_trade=500.0,
        max_daily_loss=1000.0,
        max_trades_per_day=10
    )
    
    rm = RiskManager(initial_equity=10000.0, params=risk_params)
    pm = PositionManager("BTCUSDT")
    
    # Test 1: Valid order
    is_valid, reason = rm.validate_order("BTCUSDT", "limit", 1.0, 50000.0, pm)
    assert is_valid, f"Valid order rejected: {reason}"
    print("✓ Test 1 passed: Valid order acceptance")
    
    # Test 2: Position size limit
    is_valid, reason = rm.validate_order("BTCUSDT", "limit", 15.0, 50000.0, pm)
    assert not is_valid, "Position size limit not enforced"
    print("✓ Test 2 passed: Position size limit")
    
    # Test 3: Order value limit
    is_valid, reason = rm.validate_order("BTCUSDT", "limit", 2.0, 60000.0, pm)
    assert not is_valid, "Order value limit not enforced"
    print("✓ Test 3 passed: Order value limit")
    
    # Test 4: Daily loss limit
    rm.state.daily_pnl = -1500.0  # Exceed daily loss
    is_valid, reason = rm.validate_order("BTCUSDT", "limit", 1.0, 50000.0, pm)
    assert not is_valid, "Daily loss limit not enforced"
    rm.state.daily_pnl = 0.0  # Reset
    print("✓ Test 4 passed: Daily loss limit")
    
    # Test 5: Trading frequency limit
    rm.state.daily_trades = 15  # Exceed daily trades
    is_valid, reason = rm.validate_order("BTCUSDT", "limit", 1.0, 50000.0, pm)
    assert not is_valid, "Trading frequency limit not enforced"
    rm.state.daily_trades = 0  # Reset
    print("✓ Test 5 passed: Trading frequency limit")
    
    # Test 6: Quantity adjustment
    desired_quantity = 15.0  # Would exceed position size
    adjusted_quantity = rm.adjust_quantity_for_risk(
        "BTCUSDT", desired_quantity, 50000.0, pm
    )
    assert adjusted_quantity <= risk_params.max_position_size, "Quantity not properly adjusted"
    assert adjusted_quantity >= 0, "Adjusted quantity should not be negative"
    print("✓ Test 6 passed: Quantity adjustment")
    
    # Test 7: Trade recording
    rm.record_trade("BTCUSDT", 1.0, 50000.0, 100.0, 5.0, 1000)
    assert rm.state.daily_trades == 1, "Trade not recorded"
    assert rm.state.daily_pnl == 100.0, "PNL not recorded"
    print("✓ Test 7 passed: Trade recording")
    
    # Test 8: Drawdown limit (percentage-based)
    rm.update_equity(8000.0, 1000)  # 20% drawdown from 10000
    assert rm.state.current_drawdown == 0.2, "Drawdown calculation incorrect"
    print("✓ Test 8 passed: Drawdown calculation")
    
    print("All RiskManager tests passed! ✅")


if __name__ == "__main__":
    test_position_manager()
    test_risk_manager()
    print("All Step 5 tests passed! 🎉")