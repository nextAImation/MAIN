"""
RADAR v2.0 — Full Strategy Adapter for BacktestRunner

Stage 2.3:
- اتصال هسته‌ی استراتژی (RadarV2Core) به موتور بک‌تست.
- این کلاس هیچ وابستگی سخت به نوع BacktestRunner ندارد؛
  فقط چند نقطه‌ی Adapter دارد که با TODO مشخص شده‌اند.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from strategy_radar_v2_core import (
    RadarV2Core,
    RadarV2Config,
    CoreContext,
    CoreSignal,
)


# ======================================================================
# سطح میانی: Intent برای موتور (Entry / Exit / SLTP Update)
# ======================================================================

@dataclass
class EntryIntent:
    side: str            # "long" or "short"
    qty: float
    tag: str
    reason: str


@dataclass
class ExitIntent:
    side: str            # "long" or "short"
    reason: str
    weak_exit: bool = False


@dataclass
class SLTPUpdate:
    side: str            # "long" or "short"
    tp1: Optional[float] = None
    trail_stop: Optional[float] = None


@dataclass
class ActionPlan:
    """
    خروجی استراتژی برای یک بار.
    موتور بک‌تست باید این را به اوردر واقعی تبدیل کند.
    """
    entries: List[EntryIntent]
    exits: List[ExitIntent]
    sltp_updates: List[SLTPUpdate]
    debug: Dict[str, Any]


# ======================================================================
# استراتژی High-Level برای BacktestRunner
# ======================================================================

class RadarV2Strategy:
    """
    این کلاس همان چیزی است که BacktestRunner باید با آن کار کند.

    استفاده:
        strat = RadarV2Strategy(RadarV2Config())
        for each bar:
            plan = strat.on_bar(engine_state)
            engine.apply_action_plan(plan)

    - engine_state: هر آبجکتی از موتور که بتوانی از آن:
        • ohlcv
        • equity
        • position
        • indicators
        • structure
      را بخوانی.

    من در on_bar چند بخش را با TODO علامت زدم که فقط باید
    به API موتور خودت مپ‌شان کنی.
    """

    def __init__(self, config: Optional[RadarV2Config] = None):
        self.config = config or RadarV2Config()
        self.core = RadarV2Core(self.config)

    # ------------------------------------------------------------------
    # نقطه‌ی ورودی اصلی برای هر بار
    # ------------------------------------------------------------------
    def on_bar(self, engine_state: Any) -> ActionPlan:
        """
        این متد باید در هر بار (bar) توسط BacktestRunner صدا زده شود.

        engine_state:
            آبجکتی از موتور که اطلاعات بار جاری و پوزیشن را دارد.
            (ساخت دقیقش بستگی به موتور تو دارد.)
        """

        # 1) ساخت CoreContext از روی engine_state
        ctx = self._build_core_context(engine_state)

        # 2) گرفتن CoreSignal از هسته
        core_sig: CoreSignal = self.core.step(ctx)

        # 3) تبدیل CoreSignal → ActionPlan
        plan = self._build_action_plan(ctx, core_sig)

        return plan

    # ------------------------------------------------------------------
    # ساخت CoreContext از روی موتور
    # ------------------------------------------------------------------
    def _build_core_context(self, engine_state: Any) -> CoreContext:
        """
        اینجا فقط دیتا را از موتور برداشته و در قالب CoreContext می‌ریزیم.

        ❗ فقط این بخش را باید با ساختار دیتای موتور خودت هماهنگ کنی.
        """
        # ===================== TODO: مپ‌کردن به دیتاهای واقعی =====================

        # مثال فرضی: اگر موتور تو چیزی شبیه به این داشته باشد:
        #   engine_state.bar.open, .high, .low, .close, .volume
        #   engine_state.index
        #   engine_state.timestamp
        #   engine_state.account.equity
        #   engine_state.position.size, .avg_price
        #   engine_state.indicators: dict
        #   engine_state.structures: dict
        #
        # اگر نام‌ها متفاوت است، اینجا را تغییر بده.

        index = engine_state.index
        timestamp = engine_state.timestamp

        o = engine_state.open
        h = engine_state.high
        l = engine_state.low
        c = engine_state.close
        v = engine_state.volume

        equity = engine_state.equity

        pos_size = engine_state.position_size
        pos_avg_price = engine_state.position_avg_price

        indicators: Dict[str, float] = engine_state.indicators    # dict
        htf: Dict[str, float] = getattr(engine_state, "htf", {})  # dict یا خالی
        structure: Dict[str, Any] = getattr(engine_state, "structure", {})

        # ===================================================================

        ctx = CoreContext(
            index=index,
            timestamp=timestamp,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            equity=equity,
            position_size=pos_size,
            position_avg_price=pos_avg_price,
            indicators=indicators,
            htf=htf,
            structure=structure,
        )
        return ctx

    # ------------------------------------------------------------------
    # تبدیل CoreSignal → ActionPlan (ورود/خروج/SL/TP)
    # ------------------------------------------------------------------
    def _build_action_plan(self, ctx: CoreContext, sig: CoreSignal) -> ActionPlan:
        entries: List[EntryIntent] = []
        exits: List[ExitIntent] = []
        sltp_updates: List[SLTPUpdate] = []

        pos_size = ctx.position_size

        # ========== ENTRY LOGIC ==========
        # چون مشابه Pine pyramiding=0 داریم:
        # - فقط وقتی پوزیشن صفر است اجازه ورود جدید می‌دهیم.
        if pos_size == 0:
            if sig.should_long and sig.qty_core > 0:
                entries.append(
                    EntryIntent(
                        side="long",
                        qty=sig.qty_core,
                        tag="RADAR_Long",
                        reason="core_signal_long",
                    )
                )
            elif sig.should_short and sig.qty_core > 0:
                entries.append(
                    EntryIntent(
                        side="short",
                        qty=sig.qty_core,
                        tag="RADAR_Short",
                        reason="core_signal_short",
                    )
                )

        # اگر موتور تو قرار است flipping (بستن و باز کردن در جهت مخالف) را پشتیبانی کند،
        # می‌توانی اینجا منطق اضافه‌تری بگذاری (فعلاً مثل Pine عمل می‌کنیم = بدون flip مستقیم).

        # ========== EXIT LOGIC (Weak Exit) ==========
        if pos_size > 0 and sig.exit_weak_long:
            exits.append(
                ExitIntent(
                    side="long",
                    reason="Weak_Exit_L",
                    weak_exit=True,
                )
            )

        if pos_size < 0 and sig.exit_weak_short:
            exits.append(
                ExitIntent(
                    side="short",
                    reason="Weak_Exit_S",
                    weak_exit=True,
                )
            )

        # ========== SL / TP UPDATES ==========
        # اینجا فقط سطح TP1 و Trail Stop را به موتور اطلاع می‌دهیم.
        # خود موتور می‌تواند:
        # - یا با limit/stop order شبیه‌سازی کند
        # - یا مستقیم در matching engine از روی این سطح‌ها رفتار کند.

        if pos_size > 0:
            sltp_updates.append(
                SLTPUpdate(
                    side="long",
                    tp1=sig.tp1_long,
                    trail_stop=sig.trail_stop_long,
                )
            )
        elif pos_size < 0:
            sltp_updates.append(
                SLTPUpdate(
                    side="short",
                    tp1=sig.tp1_short,
                    trail_stop=sig.trail_stop_short,
                )
            )

        plan = ActionPlan(
            entries=entries,
            exits=exits,
            sltp_updates=sltp_updates,
            debug=sig.debug,
        )
        return plan
