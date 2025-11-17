import os
import pandas as pd

from dataloader import DataLoader
from config import BacktestConfig, Timeframe
from strategy_adapter import StrategyAdapter
from backtest_runner import BacktestRunner


class DummyStrategy:
    def on_bar(self, bar_state):
        return {
            "entries": [],
            "exits": [],
            "modifications": []
        }


def run_smoke_test():

    print("[SMOKE] Starting...")

    # -------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------
    csv_path = "data/BTCUSDT_15m.csv"
    print("[SMOKE] Loading CSV:", csv_path)

    df = pd.read_csv(csv_path)
    print(f"[SMOKE] Loaded {len(df)} rows")

    # -------------------------------------------------------
    # Build config + loader
    # -------------------------------------------------------
    cfg = BacktestConfig(
        symbol="BTCUSDT",
        base_timeframe=Timeframe("15m", 15),
        htf_timeframes=[],
        warmup_bars=0
    )

    loader = DataLoader(cfg)

    tick = cfg.default_tick_size
    contract = cfg.default_contract_type.value
    fee = cfg.default_fee_rate

    bundle = loader._process_dataframe(
        df,
        "BTCUSDT",
        tick,
        contract,
        fee
    )

    print("[SMOKE] DataBundle built successfully")

    # -------------------------------------------------------
    # Create runner
    # -------------------------------------------------------
    runner = BacktestRunner(cfg)

    # -------------------------------------------------------
    # First setup engines WITHOUT strategy adapter
    # -------------------------------------------------------
    # Give a dummy temporary adapter to allow setup
    temp_adapter = StrategyAdapter(DummyStrategy(), None, None, None)

    runner.setup_engines(bundle, temp_adapter)

    print("[SMOKE] setup_engines OK")

    # -------------------------------------------------------
    # Now create REAL adapter with engines from runner
    # -------------------------------------------------------
    adapter = StrategyAdapter(
        DummyStrategy(),
        runner.indicator_engine,
        runner.risk_manager,
        runner.position_manager
    )

    # -------------------------------------------------------
    # Run
    # -------------------------------------------------------
    try:
        runner.run(bundle, adapter)
        print("[SMOKE] RUN OK")
    except Exception as e:
        print("[SMOKE][ERROR] RUN failed")
        raise


if __name__ == "__main__":
    run_smoke_test()
