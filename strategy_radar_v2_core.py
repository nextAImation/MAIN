"""
RADAR v2.0 Core Strategy Logic (Python)

این ماژول:
- منطق استراتژی Pine (ورود/خروج/TP1/BE/Trail/WeakExit/Cooldown/Structure) را
  به صورت قابل استفاده در backtester پایتونی پیاده می‌کند.
- خودش هیچ Order واقعی نمی‌فرستد؛ فقط سیگنال و سطح می‌دهد.
- با مدل حرفه‌ای bar_state کار می‌کند (BacktestRunner → bar_state).

انتظار از caller:
- روی هر بار، یک instance از RadarV2Core را صدا بزند:
    signal = core.step(ctx)
- بعد، در لایه Adapter (Stage 2.3) از روی signal تصمیم به ارسال order بگیرد.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ========================================================================
# CONFIG
# ========================================================================

@dataclass
class RadarV2Config:
    # --- Trend / MA
    ma_type: str = "EMA"
    ma_fast_len: int = 20
    ma_slow_len: int = 50
    ma_reg_len: int = 200

    # --- Quality Filters (ADX / RSI / Vol)
    adx_len: int = 14
    adx_th_fixed: float = 22.0
    use_dyn_adx: bool = False
    adx_dyn_k: float = 0.90
    require_adx_up: bool = True

    use_rsi: bool = True
    rsi_len: int = 14
    rsi_th_long: float = 48.0

    use_vol: bool = True
    vol_len: int = 20
    vol_k_long: float = 0.7

    # --- Risk & ATR
    atr_len: int = 14
    atr_mult_sl: float = 2.0
    atr_mult_tp: float = 3.0
    risk_pct: float = 0.03

    # --- Context Filters
    use_trend_filter: bool = True
    daily_soft_k: float = 0.90
    use_regime_filter: bool = False

    use_cooldown: bool = True
    cooldown_bars: int = 6

    # --- Early Entry & Cross
    use_early_entry: bool = False
    early_rsi_long: float = 55.0
    early_rsi_short: float = 45.0
    early_adx: float = 18.0
    early_qty_frac: float = 0.25

    use_cross_entry: bool = True
    cross_need_adx: bool = True
    cross_need_daily: bool = True
    cross_need_regime: bool = False

    # --- Structure & Exits
    use_struct: bool = True
    swing_len: int = 5
    bos_buf_atr: float = 0.25
    use_choch_soft: bool = False
    strict_short: bool = True
    adx_soft_strict: bool = True

    tp2_trail_atr_l: float = 2.0
    tp2_trail_atr_s: float = 1.8
    be_buf: float = 0.2
    weak_confirm_bars: int = 9

    tp1_ratio: float = 1.8
    use_break_even: bool = True
    tp1_pct: int = 29  # درصد

    use_daily_in_bypass: bool = True
    allow_daily_neutral: bool = True
    use_range_bypass: bool = True


# ========================================================================
# CONTEXT & OUTPUT TYPES
# ========================================================================

@dataclass
class CoreContext:
    """
    کانتکست هر بار که از BacktestRunner به استراتژی داده می‌شود.
    این ساختار را می‌توانی در Adapter خودت مپ کنی به هر چیزی که الان در backtester داری.
    """
    index: int
    timestamp: Any

    open: float
    high: float
    low: float
    close: float
    volume: float

    equity: float

    position_size: float
    position_avg_price: Optional[float] = None

    # اندیکاتورها (از IndicatorEngine یا هر جای دیگر)
    # انتظار کلیدها:
    #   "fast_ma", "slow_ma", "reg_ma",
    #   "adx_s", "adx_base", "rsi",
    #   "vol_sma", "vol_ratio",
    #   "atr", "diplus", "diminus"
    indicators: Dict[str, float] = field(default_factory=dict)

    # HTF/Daily اندیکاتورها، اگر داری:
    #   "ema20_d", "ema50_d", "atr14_d", "hh_d", "ll_d"
    htf: Dict[str, float] = field(default_factory=dict)

    # خروجی StructureEngine (Stage 1)، اگر وصلش کنی:
    #   "swing_high", "swing_low", "trend", "bos", "choch"
    structure: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreSignal:
    """
    خروجی خام استراتژی برای هر بار.
    هیچ اوردر واقعی اینجا ساخته نمی‌شود؛ فقط منطق.
    """
    # ورود
    should_long: bool = False
    should_short: bool = False

    # خروج‌های Weak
    exit_weak_long: bool = False
    exit_weak_short: bool = False

    # Core RR sizing
    qty_core: float = 0.0

    # TP و استاپ‌های داینامیک پیشنهادی
    tp1_long: Optional[float] = None
    trail_stop_long: Optional[float] = None

    tp1_short: Optional[float] = None
    trail_stop_short: Optional[float] = None

    # دیباگ / توضیحات
    debug: Dict[str, Any] = field(default_factory=dict)


# ========================================================================
# HELPERS
# ========================================================================

def safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    return default if denom == 0 or denom is None else numer / denom


def trail_up(prev: Optional[float], candidate: float) -> float:
    return candidate if prev is None else max(prev, candidate)


def trail_down(prev: Optional[float], candidate: float) -> float:
    return candidate if prev is None else min(prev, candidate)


def get_any(d: Dict[str, float], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


# ========================================================================
# CORE STRATEGY
# ========================================================================

class RadarV2Core:
    """
    هسته‌ی استراتژی RADAR v2.0
    - stateful per-instance (برای هر سمبل/تایم‌فریم یک instance جدا)
    - Stage 2.2: فقط منطق؛ هیچ وابستگی به engine داخلی نداره.
    """

    def __init__(self, config: Optional[RadarV2Config] = None):
        self.cfg = config or RadarV2Config()

        # --- Cooldown & state
        self.last_exit_bar: Optional[int] = None
        self.prev_position_size: float = 0.0

        # --- Structure state (معادل structState، lastHH/LL/HL/LH در Pine)
        self.struct_state: int = 0
        self.last_hh: Optional[float] = None
        self.last_ll: Optional[float] = None
        self.last_hl: Optional[float] = None
        self.last_lh: Optional[float] = None
        self.choch_warning: bool = False

        # برای preHH / preLL (rolling)
        self.recent_highs: list[float] = []
        self.recent_lows: list[float] = []

        # --- TP / BE / Trail state
        self.entry_price_l: Optional[float] = None
        self.entry_price_s: Optional[float] = None
        self.tp1_hit_l: bool = False
        self.tp1_hit_s: bool = False
        self.dyn_stop_l: Optional[float] = None
        self.dyn_stop_s: Optional[float] = None
        self.atr_at_entry_l: Optional[float] = None
        self.atr_at_entry_s: Optional[float] = None
        self.inited_l: bool = False
        self.inited_s: bool = False

        # برای تایید Weak Exit (مشابه barssince)
        self.exit_weak_streak_l: int = 0
        self.exit_weak_streak_s: int = 0

    # ------------------------------------------------------------------
    # ADX POWER
    # ------------------------------------------------------------------
    def _power_ok(self, adx_s: float, adx_base: float, require_up: bool, adx_prev: Optional[float]) -> bool:
        ok = adx_s > adx_base
        if require_up and adx_prev is not None:
            return ok and (adx_s > adx_prev)
        return ok

    # ------------------------------------------------------------------
    # STRUCTURE MODULE (simplified port of Pine logic using StructureEngine swings)
    # ------------------------------------------------------------------
    def _update_structure(self, ctx: CoreContext, atr: float, adx_s: float) -> Dict[str, Any]:
        """
        ساختار بازار را با استفاده از swing_high / swing_low (از StructureEngine)
        و منطق BOS/CHOCH تطبیق‌داده‌شده از Pine آپدیت می‌کند.
        """
        cfg = self.cfg
        high = ctx.high
        low = ctx.low
        close = ctx.close

        # --- Rolling highs/lows برای preBOS (رنج ADX پایین)
        self.recent_highs.append(high)
        self.recent_lows.append(low)
        if len(self.recent_highs) > cfg.swing_len:
            self.recent_highs.pop(0)
        if len(self.recent_lows) > cfg.swing_len:
            self.recent_lows.pop(0)

        # --- Pivotها را از StructureEngine می‌گیریم:
        # get_values() → "swing_high", "swing_low"
        ph = ctx.structure.get("swing_high")
        pl = ctx.structure.get("swing_low")

        # تشخیص pivot جدید (چون swing_high همیشه آخرین مقدار را می‌دهد، نه فقط در لحظه تشکیل)
        new_ph = None
        if ph is not None and (self.last_hh is None or abs(ph - self.last_hh) > 1e-9):
            new_ph = ph
        new_pl = None
        if pl is not None and (self.last_ll is None or abs(pl - self.last_ll) > 1e-9):
            new_pl = pl

        # آپدیت سویینگ‌ها
        if new_ph is not None:
            self.last_hh = new_ph
            if self.struct_state == -1:
                self.last_lh = new_ph

        if new_pl is not None:
            self.last_ll = new_pl
            if self.struct_state == 1:
                self.last_hl = new_pl

        # --- Dynamic ADX-based gating
        adx_threshold = 25.0
        adx_low = adx_s < adx_threshold

        # Adaptive BOS buffer based on ADX
        buf_multiplier = 1.4 if adx_low else 0.8
        bos_buf = cfg.bos_buf_atr * atr * buf_multiplier

        # BOS logic
        bos_up = self.last_hh is not None and close > self.last_hh + bos_buf
        bos_dn = self.last_ll is not None and close < self.last_ll - bos_buf

        # State transitions
        if bos_up:
            self.struct_state = 1
            self.last_hl = None
        if bos_dn:
            self.struct_state = -1
            self.last_lh = None

        # Soft CHOCH
        choch_up = self.last_hl is not None and close < self.last_hl
        choch_dn = self.last_lh is not None and close > self.last_lh
        self.choch_warning = cfg.use_choch_soft and (choch_up or choch_dn)

        # Strictness
        strict_long_condition = cfg.adx_soft_strict and adx_low
        strict_short_condition = cfg.strict_short or (cfg.adx_soft_strict and adx_low)

        # Base structure gates
        struct_ok_long = (not cfg.use_struct) or (self.struct_state == 1 if strict_long_condition else self.struct_state >= 0)
        struct_ok_short = (not cfg.use_struct) or (self.struct_state == -1 if strict_short_condition else self.struct_state <= 0)

        # Rolling HH/LL برای preBOS (range bypass)
        pre_hh = max(self.recent_highs) if self.recent_highs else None
        pre_ll = min(self.recent_lows) if self.recent_lows else None

        pre_bos_up = pre_hh is not None and close > pre_hh + (atr * cfg.bos_buf_atr * 0.50)
        pre_bos_dn = pre_ll is not None and close < pre_ll - (atr * cfg.bos_buf_atr * 0.50)

        return {
            "adx_low": adx_low,
            "bos_buf": bos_buf,
            "struct_ok_long": struct_ok_long,
            "struct_ok_short": struct_ok_short,
            "pre_bos_up": pre_bos_up,
            "pre_bos_dn": pre_bos_dn,
        }

    # ------------------------------------------------------------------
    # MAIN STEP
    # ------------------------------------------------------------------
    def step(self, ctx: CoreContext) -> CoreSignal:
        cfg = self.cfg
        debug: Dict[str, Any] = {}

        # ========== اندیکاتورهای پایه از dict ==========
        ind = ctx.indicators
        fast_ma = get_any(ind, "fast_ma", "ema_fast", default=ctx.close)
        slow_ma = get_any(ind, "slow_ma", "ema_slow", default=ctx.close)
        reg_ma = get_any(ind, "reg_ma", "ema_regime", default=ctx.close)

        adx_s = get_any(ind, "adx_s", "adx", default=0.0)
        adx_base_ind = ind.get("adx_base")
        if cfg.use_dyn_adx and adx_base_ind is not None:
            adx_base = adx_base_ind
        else:
            adx_base = cfg.adx_th_fixed

        rsi = get_any(ind, "rsi", default=50.0)
        vol_sma = get_any(ind, "vol_sma", default=ctx.volume)
        vol_ratio = ind.get("vol_ratio")
        if vol_ratio is None:
            vol_ratio = safe_div(ctx.volume, vol_sma, default=1.0)

        atr = get_any(ind, "atr", default=0.0)
        diplus = get_any(ind, "diplus", "+di", default=0.0)
        diminus = get_any(ind, "diminus", "-di", default=0.0)

        # ADX previous for powerOk()
        adx_prev = ind.get("adx_s_prev")

        # ========== DAILY FILTER (اگر دیتا هست، استفاده؛ وگرنه بای‌پس) ==========
        d20 = ctx.htf.get("ema20_d")
        d50 = ctx.htf.get("ema50_d")

        if d20 is not None and d50 is not None and cfg.use_trend_filter:
            daily_up = d20 > (d50 * cfg.daily_soft_k)
            daily_down = d20 < (d50 / cfg.daily_soft_k)
        else:
            # اگر داده HTF نیست، فیلتر روزانه را خنثی می‌کنیم
            daily_up = True
            daily_down = True

        trend_filter_long_ok = (not cfg.use_trend_filter) or daily_up
        trend_filter_short_ok = (not cfg.use_trend_filter) or daily_down

        # ========== COOLDOWN ==========
        # تقریب از justClosed = change(strategy.closedtrades) > 0
        just_closed = (self.prev_position_size != 0 and ctx.position_size == 0)
        if just_closed:
            self.last_exit_bar = ctx.index

        in_cooldown = (
            cfg.use_cooldown and
            self.last_exit_bar is not None and
            (ctx.index - self.last_exit_bar) < cfg.cooldown_bars
        )
        can_trade = not in_cooldown

        # ========== POWEROK ==========
        power_ok = self._power_ok(adx_s=adx_s, adx_base=adx_base,
                                  require_up=cfg.require_adx_up, adx_prev=adx_prev)

        # ========== STRUCTURE MODULE ==========
        struct_info = self._update_structure(ctx, atr=atr, adx_s=adx_s)
        adx_low = struct_info["adx_low"]
        bos_buf = struct_info["bos_buf"]
        struct_ok_long = struct_info["struct_ok_long"]
        struct_ok_short = struct_info["struct_ok_short"]
        pre_bos_up = struct_info["pre_bos_up"]
        pre_bos_dn = struct_info["pre_bos_dn"]

        # Daily bias برای bypass
        atr_d = ctx.htf.get("atr14_d")
        roll_hh_d = ctx.htf.get("hh_d")
        roll_ll_d = ctx.htf.get("ll_d")

        if d20 is not None and d50 is not None:
            d_bias = 1 if d20 > d50 else -1 if d20 < d50 else 0
        else:
            d_bias = 0

        buf_d = (atr_d or 0.0) * cfg.bos_buf_atr
        d_bos_up = (roll_hh_d is not None) and (ctx.close > roll_hh_d + buf_d)
        d_bos_dn = (roll_ll_d is not None) and (ctx.close < roll_ll_d - buf_d)
        d_state = 1 if d_bos_up else -1 if d_bos_dn else d_bias

        daily_ok_long = (d_state > 0) or (cfg.allow_daily_neutral and d_state == 0)
        daily_ok_short = (d_state < 0) or (cfg.allow_daily_neutral and d_state == 0)

        # Relaxed structure gating (range bypass)
        if cfg.use_range_bypass and adx_low and pre_bos_up and (not cfg.use_daily_in_bypass or daily_ok_long):
            struct_ok_long_relaxed = True
        else:
            struct_ok_long_relaxed = struct_ok_long

        if cfg.use_range_bypass and adx_low and pre_bos_dn and (not cfg.use_daily_in_bypass or daily_ok_short):
            struct_ok_short_relaxed = True
        else:
            struct_ok_short_relaxed = struct_ok_short

        # ========== STATES برای long/short ==========
        trend_long = fast_ma > slow_ma
        regime_long = trend_long and (not cfg.use_regime_filter or ctx.close > reg_ma)
        dir_long = diplus > diminus
        rsi_long_ok = (not cfg.use_rsi) or (rsi > cfg.rsi_th_long)
        vol_long_ok = (not cfg.use_vol) or (vol_ratio > cfg.vol_k_long)

        trend_short = fast_ma < slow_ma
        regime_short = trend_short and (not cfg.use_regime_filter or ctx.close < reg_ma)
        dir_short = diminus > diplus
        rsi_short_ok = (not cfg.use_rsi) or (rsi < (100.0 - cfg.rsi_th_long))
        vol_short_ok = (not cfg.use_vol) or (vol_ratio > cfg.vol_k_long)

        rr_ok_long = cfg.tp1_ratio >= 1.6
        rr_ok_short = cfg.tp1_ratio >= 1.6

        # Breakout / Pullback / Early
        breakout_long = (
            regime_long and power_ok and dir_long and rsi_long_ok and vol_long_ok and
            trend_filter_long_ok and (ctx.close > fast_ma) and (ctx.close > slow_ma) and
            can_trade and rr_ok_long
        )
        pullback_long = (
            regime_long and power_ok and dir_long and rsi_long_ok and vol_long_ok and
            trend_filter_long_ok and (ctx.low <= fast_ma < ctx.close) and
            can_trade and rr_ok_long
        )
        early_long = (
            cfg.use_early_entry and can_trade and
            (fast_ma > slow_ma) and
            (rsi > cfg.early_rsi_long) and
            (adx_s > cfg.early_adx) and
            trend_filter_long_ok
        )

        # Shorts + سخت‌گیری ADX
        guard_short = adx_s > (adx_base + 1.5)
        breakout_short = (
            regime_short and power_ok and dir_short and rsi_short_ok and vol_short_ok and
            trend_filter_short_ok and (ctx.close < fast_ma) and (ctx.close < slow_ma) and
            can_trade and rr_ok_short and guard_short
        )
        pullback_short = (
            regime_short and power_ok and dir_short and rsi_short_ok and vol_short_ok and
            trend_filter_short_ok and (ctx.high >= fast_ma > ctx.close) and
            can_trade and rr_ok_short and guard_short
        )
        early_short = (
            cfg.use_early_entry and can_trade and
            (fast_ma < slow_ma) and
            (rsi < cfg.early_rsi_short) and
            (adx_s > cfg.early_adx) and
            trend_filter_short_ok and guard_short
        )

        # Cross entries
        cross_up = fast_ma > slow_ma and self.prev_position_size <= 0  # تقریب ساده crossover
        cross_down = fast_ma < slow_ma and self.prev_position_size >= 0  # تقریب crossunder

        cross_long_ok = (
            cfg.use_cross_entry and cross_up and
            (not cfg.cross_need_adx or power_ok) and
            (not cfg.cross_need_daily or trend_filter_long_ok) and
            (not cfg.cross_need_regime or (ctx.close > reg_ma)) and
            can_trade and rr_ok_long
        )
        cross_short_ok = (
            cfg.use_cross_entry and cross_down and
            (not cfg.cross_need_adx or power_ok) and
            (not cfg.cross_need_daily or trend_filter_short_ok) and
            (not cfg.cross_need_regime or (ctx.close < reg_ma)) and
            can_trade and rr_ok_short and guard_short
        )

        # CHOCH فقط Early را kill می‌کند
        if cfg.use_choch_soft and self.choch_warning:
            early_long_filtered = False
            early_short_filtered = False
        else:
            early_long_filtered = early_long
            early_short_filtered = early_short

        # سیگنال نهایی ورود
        should_long_now = (breakout_long or pullback_long or cross_long_ok or early_long_filtered) and struct_ok_long_relaxed
        should_short_now = (breakout_short or pullback_short or cross_short_ok or early_short_filtered) and struct_ok_short_relaxed

        # =====================================================================
        # SIZING
        # =====================================================================
        risk_cap = ctx.equity * cfg.risk_pct
        risk_den = max(cfg.atr_mult_sl * atr, 1e-10)
        qty_core = max(risk_cap / risk_den, 0.0)

        # =====================================================================
        # TP / TRAIL / WEAK EXIT STATE
        # =====================================================================

        # تشخیص باز شدن پوزیشن جدید (برای ست کردن entryPrice و atrAtEntry)
        # اگر از flat به long/short رفتیم:
        if self.prev_position_size == 0 and ctx.position_size > 0:
            self.entry_price_l = ctx.position_avg_price or ctx.close
            self.atr_at_entry_l = atr
            self.tp1_hit_l = False
            self.inited_l = True
        if self.prev_position_size == 0 and ctx.position_size < 0:
            self.entry_price_s = ctx.position_avg_price or ctx.close
            self.atr_at_entry_s = atr
            self.tp1_hit_s = False
            self.inited_s = True

        # سطح TP1
        tp1_l = None
        tp1_s = None
        if self.entry_price_l is not None and self.atr_at_entry_l is not None:
            tp1_l = self.entry_price_l + (cfg.atr_mult_sl * self.atr_at_entry_l * cfg.tp1_ratio)
        if self.entry_price_s is not None and self.atr_at_entry_s is not None:
            tp1_s = self.entry_price_s - (cfg.atr_mult_sl * self.atr_at_entry_s * cfg.tp1_ratio)

        # رسیدن به TP1 + فعال‌سازی BE
        if ctx.position_size > 0 and not self.tp1_hit_l and tp1_l is not None and ctx.high >= tp1_l:
            self.tp1_hit_l = True
            if cfg.use_break_even and self.entry_price_l is not None:
                be_level = self.entry_price_l - cfg.be_buf * atr
                self.dyn_stop_l = max(self.dyn_stop_l or be_level, be_level)

        if ctx.position_size < 0 and not self.tp1_hit_s and tp1_s is not None and ctx.low <= tp1_s:
            self.tp1_hit_s = True
            if cfg.use_break_even and self.entry_price_s is not None:
                be_level_s = self.entry_price_s + cfg.be_buf * atr
                self.dyn_stop_s = min(self.dyn_stop_s or be_level_s, be_level_s)

        # آپدیت Trail پس از TP1
        if ctx.position_size > 0 and self.tp1_hit_l:
            trail_candidate_l = ctx.close - cfg.tp2_trail_atr_l * atr
            self.dyn_stop_l = trail_up(self.dyn_stop_l, trail_candidate_l)
        elif ctx.position_size <= 0:
            self.dyn_stop_l = None

        if ctx.position_size < 0 and self.tp1_hit_s:
            trail_candidate_s = ctx.close + cfg.tp2_trail_atr_s * atr
            self.dyn_stop_s = trail_down(self.dyn_stop_s, trail_candidate_s)
        elif ctx.position_size >= 0:
            self.dyn_stop_s = None

        # STRUCTURAL EXIT TIGHTENING
        if ctx.position_size > 0 and self.struct_state == -1 and atr > 0:
            self.dyn_stop_l = max(self.dyn_stop_l or (ctx.close - 1.6 * atr), ctx.close - 1.6 * atr)
        if ctx.position_size < 0 and self.struct_state == 1 and atr > 0:
            self.dyn_stop_s = min(self.dyn_stop_s or (ctx.close + 1.6 * atr), ctx.close + 1.6 * atr)

        # WEAK EXIT CONDITIONS (Long)
        exit_weak_raw_l = (ctx.close < slow_ma) or (adx_s < 18) or (diplus < diminus)
        if exit_weak_raw_l:
            self.exit_weak_streak_l += 1
        else:
            self.exit_weak_streak_l = 0
        exit_weak_l = self.exit_weak_streak_l >= cfg.weak_confirm_bars

        # WEAK EXIT CONDITIONS (Short)
        exit_weak_raw_s = (ctx.close > slow_ma) or (adx_s < 20) or (diplus > diminus)
        if exit_weak_raw_s:
            self.exit_weak_streak_s += 1
        else:
            self.exit_weak_streak_s = 0
        exit_weak_s = self.exit_weak_streak_s >= cfg.weak_confirm_bars

        # WEAK TIGHTENING
        if self.inited_l and ctx.position_size > 0 and exit_weak_l:
            tight_l = max(slow_ma - 0.3 * atr, self.entry_price_l or slow_ma)
            self.dyn_stop_l = max(self.dyn_stop_l or tight_l, tight_l)

        if self.inited_s and ctx.position_size < 0 and exit_weak_s:
            tight_s = max(slow_ma, ctx.close + 0.2 * atr)
            self.dyn_stop_s = min(self.dyn_stop_s or tight_s, tight_s)

        # RESET وقتی flat شدیم
        if ctx.position_size == 0:
            self.tp1_hit_l = False
            self.tp1_hit_s = False
            self.entry_price_l = None
            self.entry_price_s = None
            self.atr_at_entry_l = None
            self.atr_at_entry_s = None
            self.inited_l = False
            self.inited_s = False
            self.dyn_stop_l = None
            self.dyn_stop_s = None

        # =====================================================================
        # CORE OUTPUT
        # =====================================================================
        signal = CoreSignal(
            should_long=should_long_now,
            should_short=should_short_now,
            exit_weak_long=exit_weak_l,
            exit_weak_short=exit_weak_s,
            qty_core=qty_core,
            tp1_long=tp1_l,
            trail_stop_long=self.dyn_stop_l,
            tp1_short=tp1_s,
            trail_stop_short=self.dyn_stop_s,
            debug={
                "in_cooldown": in_cooldown,
                "struct_state": self.struct_state,
                "struct_ok_long": struct_ok_long,
                "struct_ok_short": struct_ok_short,
                "struct_ok_long_relaxed": struct_ok_long_relaxed,
                "struct_ok_short_relaxed": struct_ok_short_relaxed,
                "adx_s": adx_s,
                "adx_base": adx_base,
                "adx_low": adx_low,
                "trend_long": trend_long,
                "trend_short": trend_short,
                "breakout_long": breakout_long,
                "pullback_long": pullback_long,
                "breakout_short": breakout_short,
                "pullback_short": pullback_short,
                "early_long": early_long,
                "early_short": early_short,
                "cross_long_ok": cross_long_ok,
                "cross_short_ok": cross_short_ok,
                "daily_state": d_state,
                "daily_ok_long": daily_ok_long,
                "daily_ok_short": daily_ok_short,
                "tp1_hit_l": self.tp1_hit_l,
                "tp1_hit_s": self.tp1_hit_s,
            }
        )

        # به‌روزرسانی prev_position_size در انتهای بار
        self.prev_position_size = ctx.position_size

        return signal
