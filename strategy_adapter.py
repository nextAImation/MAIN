"""
Strategy Adapter Module - Final Stable Version
"""

from typing import Dict, List, Any, Optional
from decimal import Decimal


class StrategyAdapter:
    """
    Minimal, deterministic strategy interface.
    Compatible with BacktestRunner and all engine modules.
    """
    
    def __init__(self, config, indicator_engine, risk_manager, position_manager):
        self.config = config
        self.indicator_engine = indicator_engine
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        
        self._last_signal = None
        self._signal_count = 0

    def generate_signals(self, bar_state) -> List[Dict[str, Any]]:
        """
        Input:
            bar_state: OHLCV + indicators + structure
        Output:
            List of signals (BUY / SELL / EXIT)
        """

        signals = []

        # === Safe RSI retrieval ===
        rsi = None
        if hasattr(bar_state, "indicators"):
            rsi = bar_state.indicators.get("rsi_14", None)

        # === Position Info ===
        pos_qty = self.position_manager.state.quantity
        pos_side = self.position_manager.state.direction

        # === Exit Logic (generic + safe) ===
        if pos_qty != 0:
            exit_action = "SELL" if pos_side == "long" else "BUY"

            signals.append({
                "action": exit_action,
                "quantity": Decimal(abs(pos_qty)),
                "reason": "position_exit",
                "order_type": "MARKET",
                "limit_price": None,
                "stop_price": None
            })

            return signals

        # === Buy Logic Example ===
        if rsi is not None and rsi < 30:
            signals.append({
                "action": "BUY",
                "quantity": Decimal('1'),
                "reason": "rsi_oversold",
                "order_type": "LIMIT",
                "limit_price": Decimal(str(bar_state.close * 0.99)),
                "stop_price": None
            })

        # === Sell Logic Example ===
        elif rsi is not None and rsi > 70:
            signals.append({
                "action": "SELL",
                "quantity": Decimal('1'),
                "reason": "rsi_overbought",
                "order_type": "LIMIT",
                "limit_price": Decimal(str(bar_state.close * 1.01)),
                "stop_price": None
            })

        return signals

    def get_actions(self, bar_state) -> Dict[str, Any]:
        """
        Convert strategy signals → format acceptable by BacktestRunner
        """
        signals = self.generate_signals(bar_state)

        actions = {
            "cancel_orders": [],
            "new_orders": [],
            "adjust_orders": []
        }

        for i, signal in enumerate(signals):
            order_data = {
                "order_id": f"sig_{bar_state.bar_index}_{i}",
                "symbol": self.config.symbol,
                "side": signal["action"],
                "order_type": signal["order_type"],
                "quantity": str(signal["quantity"]),
                "limit_price": float(signal["limit_price"]) if signal["limit_price"] is not None else None,
                "stop_price": float(signal["stop_price"]) if signal["stop_price"] is not None else None,
                "reduce_only": False,
                "post_only": signal["order_type"] == "LIMIT",
                "parent_order_id": None
            }
            actions["new_orders"].append(order_data)

        self._signal_count += len(signals)
        if signals:
            self._last_signal = signals[-1]

        return actions

    def reset(self):
        self._last_signal = None
        self._signal_count = 0