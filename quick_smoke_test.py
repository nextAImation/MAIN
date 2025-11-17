# quick_smoke_test.py

import logging
from datetime import datetime

import pandas as pd

from backtest_runner import BacktestRunner
from config import BacktestConfig, RiskParams, ExecutionParams, Timeframe


class DummyStrategy:
    """
    استراتژی تست خیلی ساده که هیچ سفارشی نمی‌دهد.
    فقط برای این است که BacktestRunner بدون کرش تا آخر loop اجرا شود.
    """
    def get_actions(self, bar_state):
        # هیچ سفارش جدید / کنسل / ادجاست نمی‌دهیم
        return {
            "cancel_orders": [],
            "new_orders": [],
            "adjust_orders": [],
        }


class SimpleDataBundle:
    """
    مینیمال‌ترین DataBundle ممکن برای اینکه با BacktestRunner فعلی سازگار باشد.

    فقط همین دو فیلد استفاده می‌شود:
      - base_tf_candles[symbol]  → df با ستون‌های ts, open, high, low, close, volume
      - htf_candles[symbol]      → دیکشنری خالی (چون higher_timeframes را خالی می‌گذاریم)
    """
    def __init__(self, symbol: str, df: pd.DataFrame):
        self.base_tf_candles = {
            symbol: df
        }
        # HTF خالی، چون در config.higher_timeframes چیزی نمی‌گذاریم
        self.htf_candles = {
            symbol: {}
        }


def main() -> None:
    print("[SMOKE] Starting...")

    # 1) ساخت کانفیگ بر اساس تست داخل backtest_runner.py
    config = BacktestConfig(
        initial_equity=10_000.0,
        symbol="BTCUSDT",
        base_timeframe=Timeframe("15m", 15),
        risk_params=RiskParams(),
        execution_params=ExecutionParams(),
    )

    # 🔹 برای اسموک‌تست warmup را صفر می‌کنیم تا با ۶ بار هم اجرا شود
    config.warmup_bars = 0

    # اگر این فیلدها در BacktestConfig نباشند، برای اطمینان ست می‌کنیم
    if not hasattr(config, "higher_timeframes"):
        config.higher_timeframes = []
    if not hasattr(config, "default_tick_size"):
        config.default_tick_size = 0.1
    if not hasattr(config, "default_fee_rate"):
        config.default_fee_rate = 0.0004

    # 2) لود CSV
    csv_path = "data/BTCUSDT_15m.csv"
    print(f"[SMOKE] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = ["ts", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    print(f"[SMOKE] Loaded {len(df)} rows")

    # 3) ساخت SimpleDataBundle مینیمال
    bundle = SimpleDataBundle(symbol=config.symbol, df=df)

    # 4) استراتژی تستی
    strategy = DummyStrategy()

    # 5) BacktestRunner
    logging.basicConfig(level=logging.INFO)
    runner = BacktestRunner(config)

    # 6) اجرای بک‌تست
    print("[SMOKE] Running backtest...")
    results = runner.run(bundle, strategy)
    print("[SMOKE] Finished without crash.")

    # 7) کمی خروجی برای چک
    final_equity = results.get("final_equity")
    total_bars = results.get("total_bars")
    print(f"[SMOKE] final_equity = {final_equity}")
    print(f"[SMOKE] total_bars   = {total_bars}")


if __name__ == "__main__":
    main()
