import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StructureType(Enum):
    """Types of market structure patterns."""
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    FVG_BULLISH = "fvg_bullish"
    FVG_BEARISH = "fvg_bearish"
    ORDER_BLOCK_BULLISH = "ob_bullish"
    ORDER_BLOCK_BEARISH = "ob_bearish"
    BOS_BULLISH = "bos_bullish"
    BOS_BEARISH = "bos_bearish"
    CHOCH_BULLISH = "choch_bullish"
    CHOCH_BEARISH = "choch_bearish"


@dataclass
class StructurePoint:
    """A detected market structure point."""
    type: StructureType
    timestamp: pd.Timestamp
    price: float
    confirmation_index: int  # Bar index when structure was confirmed
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StructureEngine:
    """
    Zero-repainting market structure detection engine.
    All structures are confirmed only after full formation with no forward references.
    """

    def __init__(self, swing_lookback: int = 3, swing_lookforward: int = 2):
        """
        Initialize Structure Engine.

        Args:
            swing_lookback: Number of bars to look back for swing confirmation
            swing_lookforward: Number of bars to look forward for swing confirmation
        """
        self.swing_lookback = swing_lookback
        self.swing_lookforward = swing_lookforward
        self.min_bars_for_confirmation = swing_lookback + swing_lookforward + 1

        # Storage for detected structures
        self.swing_highs: List[StructurePoint] = []
        self.swing_lows: List[StructurePoint] = []
        self.fvgs: List[StructurePoint] = []
        self.order_blocks: List[StructurePoint] = []
        self.bos_events: List[StructurePoint] = []
        self.choch_events: List[StructurePoint] = []

        # Cache for performance (reserved – فعلاً استفاده نمی‌شود)
        self._cached_swings: Dict = {}

    # ───────────────────────── API اصلی برای BacktestRunner ─────────────────────────

    def compute_all(self, df: pd.DataFrame) -> None:
        """
        Compute all market structures for the given DataFrame.
        This is the new API method for integration with BacktestRunner.
        """
        self.compute_structures(df)

    def get_values(self, bar_index: int) -> Dict[str, Any]:
        """
        Get structure values summary for a specific bar index.
        Returns only structures confirmed by this index (zero lookahead).
        """
        structures = self.get_structures_at_index(bar_index)

        # Last confirmed of each type
        last_swing_high = structures["swing_highs"][-1] if structures["swing_highs"] else None
        last_swing_low = structures["swing_lows"][-1] if structures["swing_lows"] else None
        last_fvg = structures["fvgs"][-1] if structures["fvgs"] else None
        last_ob = structures["order_blocks"][-1] if structures["order_blocks"] else None
        last_bos = structures["bos_events"][-1] if structures["bos_events"] else None
        last_choch = structures["choch_events"][-1] if structures["choch_events"] else None

        trend_state = self._determine_trend_state(structures)

        active_structures = {
            "swing_high_count": len(structures["swing_highs"]),
            "swing_low_count": len(structures["swing_lows"]),
            "fvg_count": len(structures["fvgs"]),
            "order_block_count": len(structures["order_blocks"]),
            "bos_count": len(structures["bos_events"]),
            "choch_count": len(structures["choch_events"]),
        }

        price_levels = {
            "last_swing_high": last_swing_high.price if last_swing_high else 0.0,
            "last_swing_low": last_swing_low.price if last_swing_low else 0.0,
            "last_fvg_price": last_fvg.price if last_fvg else 0.0,
            "last_ob_price": last_ob.price if last_ob else 0.0,
            "last_bos_price": last_bos.price if last_bos else 0.0,
            "last_choch_price": last_choch.price if last_choch else 0.0,
        }

        structure_types = {
            "last_swing_high_type": last_swing_high.type.value if last_swing_high else "none",
            "last_swing_low_type": last_swing_low.type.value if last_swing_low else "none",
            "last_fvg_type": last_fvg.type.value if last_fvg else "none",
            "last_ob_type": last_ob.type.value if last_ob else "none",
            "last_bos_type": last_bos.type.value if last_bos else "none",
            "last_choch_type": last_choch.type.value if last_choch else "none",
        }

        bos_state = None
        if last_bos:
            bos_state = "up" if last_bos.type == StructureType.BOS_BULLISH else "down"

        choch_state = None
        if last_choch:
            choch_state = "up" if last_choch.type == StructureType.CHOCH_BULLISH else "down"

        return {
            **active_structures,
            **price_levels,
            **structure_types,
            "has_swing_high": last_swing_high is not None,
            "has_swing_low": last_swing_low is not None,
            "has_fvg": last_fvg is not None,
            "has_order_block": last_ob is not None,
            "has_bos": last_bos is not None,
            "has_choch": last_choch is not None,
            "swing_high": last_swing_high.price if last_swing_high else None,
            "swing_low": last_swing_low.price if last_swing_low else None,
            "bos": bos_state,
            "choch": choch_state,
            "trend": trend_state,
        }

    # ───────────────────────── منطق ترند ─────────────────────────

    def _determine_trend_state(self, structures: Dict[str, List["StructurePoint"]]) -> str:
        """
        Determine the current trend state based on confirmed swing points.
        """
        swing_highs = structures["swing_highs"]
        swing_lows = structures["swing_lows"]

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "neutral"

        sorted_swing_highs = sorted(swing_highs, key=lambda x: x.timestamp)
        sorted_swing_lows = sorted(swing_lows, key=lambda x: x.timestamp)

        last_swing_high = sorted_swing_highs[-1]
        prev_swing_high = sorted_swing_highs[-2]
        last_swing_low = sorted_swing_lows[-1]
        prev_swing_low = sorted_swing_lows[-2]

        # Higher highs + higher lows → bullish
        if (
            last_swing_high.price > prev_swing_high.price
            and last_swing_low.price > prev_swing_low.price
        ):
            return "bullish"

        # Lower highs + lower lows → bearish
        if (
            last_swing_high.price < prev_swing_high.price
            and last_swing_low.price < prev_swing_low.price
        ):
            return "bearish"

        return "neutral"

    # ───────────────────────── اصلی: محاسبه ساختار ─────────────────────────

    def compute_structures(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Compute all market structures for the given DataFrame.
        """
        logger.info("Computing market structures...")

        self.swing_highs.clear()
        self.swing_lows.clear()
        self.fvgs.clear()
        self.order_blocks.clear()
        self.bos_events.clear()
        self.choch_events.clear()
        self._cached_swings.clear()

        if len(df) < self.min_bars_for_confirmation:
            logger.warning("Insufficient data for structure detection")
            return self._create_structure_dataframes(df)

        self._detect_swing_points(df)
        self._detect_fvgs(df)
        self._detect_order_blocks(df)
        self._detect_bos_choch(df)

        return self._create_structure_dataframes(df)

    # ───────────────────────── سوئینگ‌ها ─────────────────────────

    def _detect_swing_points(self, df: pd.DataFrame):
        """
        Detect swing highs and swing lows with confirmation.
        A swing high is confirmed when we have lookback+lookforward bars around it.
        """
        high = df["high"].values
        low = df["low"].values

        for i in range(self.swing_lookback, len(df) - self.swing_lookforward):
            # Swing High
            current_high = high[i]
            is_swing_high = True

            # Left side (lookback) – all highs ≤ current_high
            for j in range(1, self.swing_lookback + 1):
                if high[i - j] > current_high:
                    is_swing_high = False
                    break

            # Right side (lookforward) – all highs ≤ current_high
            if is_swing_high:
                for j in range(1, self.swing_lookforward + 1):
                    if high[i + j] > current_high:
                        is_swing_high = False
                        break

            if is_swing_high:
                existing_swing = any(
                    sh.metadata.get("index", -1) == i for sh in self.swing_highs
                )
                if not existing_swing:
                    swing_high = StructurePoint(
                        type=StructureType.SWING_HIGH,
                        timestamp=df.index[i],
                        price=current_high,
                        confirmation_index=i + self.swing_lookforward,
                        metadata={
                            "index": i,
                            "lookback": self.swing_lookback,
                            "lookforward": self.swing_lookforward,
                        },
                    )
                    self.swing_highs.append(swing_high)

            # Swing Low
            current_low = low[i]
            is_swing_low = True

            # Left side – all lows ≥ current_low
            for j in range(1, self.swing_lookback + 1):
                if low[i - j] < current_low:
                    is_swing_low = False
                    break

            # Right side – all lows ≥ current_low
            if is_swing_low:
                for j in range(1, self.swing_lookforward + 1):
                    if low[i + j] < current_low:
                        is_swing_low = False
                        break

            if is_swing_low:
                existing_swing = any(
                    sl.metadata.get("index", -1) == i for sl in self.swing_lows
                )
                if not existing_swing:
                    swing_low = StructurePoint(
                        type=StructureType.SWING_LOW,
                        timestamp=df.index[i],
                        price=current_low,
                        confirmation_index=i + self.swing_lookforward,
                        metadata={
                            "index": i,
                            "lookback": self.swing_lookback,
                            "lookforward": self.swing_lookforward,
                        },
                    )
                    self.swing_lows.append(swing_low)

    # ───────────────────────── FVG ─────────────────────────

    def _detect_fvgs(self, df: pd.DataFrame):
        """
        Detect Fair Value Gaps (FVGs).

        Bullish FVG: low[i] > high[i-2]  (current low > two bars ago high)
        Bearish FVG: high[i] < low[i-2] (current high < two bars ago low)

        Confirmed at bar i. (بدون نگاه به آینده)
        """
        high = df["high"].values
        low = df["low"].values

        for i in range(2, len(df)):
            # Bullish FVG (gap up)
            if low[i] > high[i - 2]:
                fvg_high = high[i - 2]
                fvg_low = low[i]
                fvg_mid = (fvg_high + fvg_low) / 2

                existing_fvg = any(
                    fvg.metadata.get("index", -1) == i
                    and fvg.type == StructureType.FVG_BULLISH
                    for fvg in self.fvgs
                )
                if not existing_fvg:
                    self.fvgs.append(
                        StructurePoint(
                            type=StructureType.FVG_BULLISH,
                            timestamp=df.index[i],
                            price=fvg_mid,
                            confirmation_index=i,
                            metadata={
                                "index": i,
                                "fvg_high": fvg_high,
                                "fvg_low": fvg_low,
                                "gap_size": fvg_low - fvg_high,
                            },
                        )
                    )

            # Bearish FVG (gap down)
            if high[i] < low[i - 2]:
                fvg_high = low[i - 2]
                fvg_low = high[i]
                fvg_mid = (fvg_high + fvg_low) / 2

                existing_fvg = any(
                    fvg.metadata.get("index", -1) == i
                    and fvg.type == StructureType.FVG_BEARISH
                    for fvg in self.fvgs
                )
                if not existing_fvg:
                    self.fvgs.append(
                        StructurePoint(
                            type=StructureType.FVG_BEARISH,
                            timestamp=df.index[i],
                            price=fvg_mid,
                            confirmation_index=i,
                            metadata={
                                "index": i,
                                "fvg_high": fvg_high,
                                "fvg_low": fvg_low,
                                "gap_size": fvg_high - fvg_low,
                            },
                        )
                    )

    # ───────────────────────── Order Blocks ─────────────────────────

    def _detect_order_blocks(self, df: pd.DataFrame):
        """
        Detect Order Blocks (OB).

        Bullish OB:
            Bearish candle (close < open)
            followed by Bullish candle (close > open)
            AND bullish close > previous candle high

        Bearish OB:
            Bullish candle followed by Bearish candle
            AND bearish close < previous candle low
        """
        open_price = df["open"].values
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        for i in range(1, len(df)):
            # Bullish OB
            prev_bearish = close[i - 1] < open_price[i - 1]
            curr_bullish = close[i] > open_price[i]
            close_above_prev_high = close[i] > high[i - 1]

            if prev_bearish and curr_bullish and close_above_prev_high:
                ob_price = low[i - 1]
                existing_ob = any(
                    ob.timestamp == df.index[i]
                    and ob.type == StructureType.ORDER_BLOCK_BULLISH
                    for ob in self.order_blocks
                )
                if not existing_ob:
                    self.order_blocks.append(
                        StructurePoint(
                            type=StructureType.ORDER_BLOCK_BULLISH,
                            timestamp=df.index[i],
                            price=ob_price,
                            confirmation_index=i,
                            metadata={
                                "index": i,
                                "candle_high": high[i - 1],
                                "candle_low": low[i - 1],
                                "candle_range": high[i - 1] - low[i - 1],
                            },
                        )
                    )

            # Bearish OB
            prev_bullish = close[i - 1] > open_price[i - 1]
            curr_bearish = close[i] < open_price[i]
            close_below_prev_low = close[i] < low[i - 1]

            if prev_bullish and curr_bearish and close_below_prev_low:
                ob_price = high[i - 1]
                existing_ob = any(
                    ob.timestamp == df.index[i]
                    and ob.type == StructureType.ORDER_BLOCK_BEARISH
                    for ob in self.order_blocks
                )
                if not existing_ob:
                    self.order_blocks.append(
                        StructurePoint(
                            type=StructureType.ORDER_BLOCK_BEARISH,
                            timestamp=df.index[i],
                            price=ob_price,
                            confirmation_index=i,
                            metadata={
                                "index": i,
                                "candle_high": high[i - 1],
                                "candle_low": low[i - 1],
                                "candle_range": high[i - 1] - low[i - 1],
                            },
                        )
                    )

    # ───────────────────────── BOS / CHOCH ─────────────────────────

    def _detect_bos_choch(self, df: pd.DataFrame):
        """
        Detect Break of Structure (BOS) and Change of Character (CHOCH).
        Uses confirmed swing points for detection.
        """
        if not self.swing_highs or not self.swing_lows:
            logger.warning("No swing points available for BOS/CHOCH detection")
            return

        close = df["close"].values

        for i in range(len(df)):
            current_close = close[i]
            current_timestamp = df.index[i]

            confirmed_structures = self.get_structures_at_index(i)
            confirmed_swing_highs = confirmed_structures["swing_highs"]
            confirmed_swing_lows = confirmed_structures["swing_lows"]

            if len(confirmed_swing_highs) < 2 or len(confirmed_swing_lows) < 2:
                continue

            sorted_swing_highs = sorted(confirmed_swing_highs, key=lambda x: x.timestamp)
            sorted_swing_lows = sorted(confirmed_swing_lows, key=lambda x: x.timestamp)

            last_swing_high = sorted_swing_highs[-1]
            prev_swing_high = sorted_swing_highs[-2]
            last_swing_low = sorted_swing_lows[-1]
            prev_swing_low = sorted_swing_lows[-2]

            uptrend = (
                last_swing_high.timestamp > prev_swing_high.timestamp
                and last_swing_low.timestamp > prev_swing_low.timestamp
                and last_swing_high.price > prev_swing_high.price
                and last_swing_low.price > prev_swing_low.price
            )

            downtrend = (
                last_swing_high.timestamp > prev_swing_high.timestamp
                and last_swing_low.timestamp > prev_swing_low.timestamp
                and last_swing_high.price < prev_swing_high.price
                and last_swing_low.price < prev_swing_low.price
            )

            # Bullish BOS: close > last swing high در آپ‌ترند
            if uptrend and current_close > last_swing_high.price:
                existing_bos = any(
                    bos.confirmation_index == i
                    and bos.type == StructureType.BOS_BULLISH
                    for bos in self.bos_events
                )
                if not existing_bos:
                    self.bos_events.append(
                        StructurePoint(
                            type=StructureType.BOS_BULLISH,
                            timestamp=current_timestamp,
                            price=current_close,
                            confirmation_index=i,
                            metadata={
                                "broken_swing_high": last_swing_high.price,
                                "previous_swing_high": prev_swing_high.price,
                            },
                        )
                    )

            # Bearish BOS: close < last swing low در داون‌ترند
            if downtrend and current_close < last_swing_low.price:
                existing_bos = any(
                    bos.confirmation_index == i
                    and bos.type == StructureType.BOS_BEARISH
                    for bos in self.bos_events
                )
                if not existing_bos:
                    self.bos_events.append(
                        StructurePoint(
                            type=StructureType.BOS_BEARISH,
                            timestamp=current_timestamp,
                            price=current_close,
                            confirmation_index=i,
                            metadata={
                                "broken_swing_low": last_swing_low.price,
                                "previous_swing_low": prev_swing_low.price,
                            },
                        )
                    )

            # Bullish CHOCH: در داون‌ترند، close > previous swing high
            if downtrend and current_close > prev_swing_high.price:
                existing_choch = any(
                    choch.confirmation_index == i
                    and choch.type == StructureType.CHOCH_BULLISH
                    for choch in self.choch_events
                )
                if not existing_choch:
                    self.choch_events.append(
                        StructurePoint(
                            type=StructureType.CHOCH_BULLISH,
                            timestamp=current_timestamp,
                            price=current_close,
                            confirmation_index=i,
                            metadata={
                                "broken_swing_high": prev_swing_high.price,
                                "current_swing_high": last_swing_high.price,
                            },
                        )
                    )

            # Bearish CHOCH: در آپ‌ترند، close < previous swing low
            if uptrend and current_close < prev_swing_low.price:
                existing_choch = any(
                    choch.confirmation_index == i
                    and choch.type == StructureType.CHOCH_BEARISH
                    for choch in self.choch_events
                )
                if not existing_choch:
                    self.choch_events.append(
                        StructurePoint(
                            type=StructureType.CHOCH_BEARISH,
                            timestamp=current_timestamp,
                            price=current_close,
                            confirmation_index=i,
                            metadata={
                                "broken_swing_low": prev_swing_low.price,
                                "current_swing_low": last_swing_low.price,
                            },
                        )
                    )

    # ───────────────────────── خروجی DataFrame ─────────────────────────

    def _create_structure_dataframes(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create DataFrames with structure information."""
        result: Dict[str, pd.DataFrame] = {}

        swing_data = [
            {
                "timestamp": swing.timestamp,
                "type": swing.type.value,
                "price": swing.price,
                "confirmation_index": swing.confirmation_index,
                **swing.metadata,
            }
            for swing in (self.swing_highs + self.swing_lows)
        ]
        result["swings"] = pd.DataFrame(swing_data) if swing_data else pd.DataFrame()

        fvg_data = [
            {
                "timestamp": fvg.timestamp,
                "type": fvg.type.value,
                "price": fvg.price,
                "confirmation_index": fvg.confirmation_index,
                **fvg.metadata,
            }
            for fvg in self.fvgs
        ]
        result["fvgs"] = pd.DataFrame(fvg_data) if fvg_data else pd.DataFrame()

        ob_data = [
            {
                "timestamp": ob.timestamp,
                "type": ob.type.value,
                "price": ob.price,
                "confirmation_index": ob.confirmation_index,
                **ob.metadata,
            }
            for ob in self.order_blocks
        ]
        result["order_blocks"] = pd.DataFrame(ob_data) if ob_data else pd.DataFrame()

        bos_data = [
            {
                "timestamp": bos.timestamp,
                "type": bos.type.value,
                "price": bos.price,
                "confirmation_index": bos.confirmation_index,
                **bos.metadata,
            }
            for bos in (self.bos_events + self.choch_events)
        ]
        result["bos_choch"] = pd.DataFrame(bos_data) if bos_data else pd.DataFrame()

        return result

    # ───────────────────────── Helperهای دسترسی ─────────────────────────

    def get_structures_at_index(self, index: int) -> Dict[str, List[StructurePoint]]:
        """
        Get all structures that are confirmed and available at a specific bar index.
        Only structures with confirmation_index <= index are returned.
        """
        return {
            "swing_highs": [sh for sh in self.swing_highs if sh.confirmation_index <= index],
            "swing_lows": [sl for sl in self.swing_lows if sl.confirmation_index <= index],
            "fvgs": [fvg for fvg in self.fvgs if fvg.confirmation_index <= index],
            "order_blocks": [ob for ob in self.order_blocks if ob.confirmation_index <= index],
            "bos_events": [bos for bos in self.bos_events if bos.confirmation_index <= index],
            "choch_events": [c for c in self.choch_events if c.confirmation_index <= index],
        }

    def get_last_structures(self, count: int = 5) -> Dict[str, List[StructurePoint]]:
        """Get the last N structures of each type."""
        return {
            "swing_highs": self.swing_highs[-count:] if self.swing_highs else [],
            "swing_lows": self.swing_lows[-count:] if self.swing_lows else [],
            "fvgs": self.fvgs[-count:] if self.fvgs else [],
            "order_blocks": self.order_blocks[-count:] if self.order_blocks else [],
            "bos_events": self.bos_events[-count:] if self.bos_events else [],
            "choch_events": self.choch_events[-count:] if self.choch_events else [],
        }


class MultiTimeframeStructureEngine:
    """
    Structure engine that works across multiple timeframes.
    Ensures structure detection is consistent and non-repainting across all timeframes.
    """

    def __init__(self, base_lookback: int = 3, base_lookforward: int = 2):
        self.base_engine = StructureEngine(base_lookback, base_lookforward)
        self.htf_engines: Dict[str, StructureEngine] = {}

        htf_params = {
            "1H": (5, 3),
            "4H": (8, 4),
            "1D": (10, 5),
        }
        for tf, (lb, lf) in htf_params.items():
            self.htf_engines[tf] = StructureEngine(lb, lf)

    def compute_all(
        self,
        base_data: pd.DataFrame,
        htf_data: Dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Compute structures across all timeframes (API برای BacktestRunner)."""
        if htf_data is None:
            htf_data = {}

        self.base_engine.compute_all(base_data)
        for tf_name, tf_df in htf_data.items():
            if tf_name in self.htf_engines:
                self.htf_engines[tf_name].compute_all(tf_df)

    def get_values(self, bar_index: int) -> Dict[str, Any]:
        """Return only base timeframe structures (همون چیزی که استراتژی لازم داره)."""
        return self.base_engine.get_values(bar_index)

    def compute_multi_timeframe_structures(
        self,
        base_data: pd.DataFrame,
        htf_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Compute structures across all timeframes.
        Returns nested dict of structure DataFrames.
        """
        result: Dict[str, Dict[str, pd.DataFrame]] = {}
        result["base"] = self.base_engine.compute_structures(base_data)

        for tf_name, tf_df in htf_data.items():
            if tf_name in self.htf_engines:
                result[tf_name] = self.htf_engines[tf_name].compute_structures(tf_df)

        return result

    def get_consensus_structures(
        self,
        current_index: int,
        min_timeframes: int = 2,
    ) -> Dict[str, List[StructurePoint]]:
        """
        Get structures that appear across multiple timeframes (consensus).
        فعلاً نسخه ساده: همون ساختارهای تایم‌فریم پایه را برمی‌گرداند.
        """
        return self.base_engine.get_structures_at_index(current_index)
