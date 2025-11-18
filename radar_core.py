# radar_core.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import math

Number = Union[int, float]


# ============================================================
# CONFIG – معادل input های اصلی Pine
# ============================================================

@dataclass
class RadarCoreConfig:
    # MA
    ma_type: str = "EMA"         # "EMA", "TEMA", "HMA", "SMA"
    ma_fast_len: int = 20
    ma_slow_len: int = 50
    ma_reg_len: int = 200

    # ADX / DMI
    adx_len: int = 14
    adx_th_fixed: float = 22.0
    use_dyn_adx: bool = False
    adx_dyn_k: float = 0.90
    require_adx_up: bool = True

    # RSI
    use_rsi: bool = True
    rsi_len: int = 14
    rsi_th_long: float = 48.0

    # Volume
    use_vol: bool = True
    vol_len: int = 20
    vol_k_long: float = 0.7

    # ATR / Risk
    atr_len: int = 14
    atr_mult_sl: float = 2.0
    atr_mult_tp: float = 3.0
    risk_pct: float = 0.026

    # Trend filter (daily EMA20/50)
    use_trend_filter: bool = True
    daily_soft_k: float = 1.0

    # Regime filter (EMA200)
    use_regime_filter: bool = True

    # Cooldown
    use_cooldown: bool = True
    cooldown_bars: int = 6

    # Early Entry
    use_early_entry: bool = False
    early_rsi_long: float = 55.0
    early_rsi_short: float = 45.0
    early_adx: float = 18.0
    early_qty_frac: float = 0.25

    # Cross Entry
    use_cross_entry: bool = False
    cross_need_adx: bool = True
    cross_need_daily: bool = True
    cross_need_regime: bool = False

    # Market Structure
    use_struct: bool = True
    swing_len: int = 5
    bos_buf_atr: float = 0.30
    use_choch_soft: bool = False
    strict_short: bool = True
    adx_soft_strict: bool = True

    # Optimization inputs
    tp2_trail_atr_l: float = 2.4
    tp2_trail_atr_s: float = 1.8
    be_buf: float = 0.30
    weak_confirm_bars: int = 2

    # TP/BE config
    tp1_ratio: float = 1.6
    use_break_even: bool = True
    tp1_pct: int = 32   # درصد

    # Daily bypass
    use_daily_in_bypass: bool = True
    allow_daily_neutral: bool = True

    # Data integrity (from Pine inputs)
    use_data_sentinel: bool = True
    enable_fallback: bool = True
    max_data_gap: int = 5
    alert_on_data_issue: bool = True

    # Range bypass
    use_range_bypass: bool = True


# ============================================================
# STATE – تمام stateهای ضروری
# ============================================================

@dataclass
class RadarState:
    # bar index
    bar_index: int = -1

    # Intraday price series
    timestamps: List[int] = field(default_factory=list)
    opens: List[float] = field(default_factory=list)
    highs: List[float] = field(default_factory=list)
    lows: List[float] = field(default_factory=list)
    closes: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)

    # MAs
    fast_ma: List[float] = field(default_factory=list)
    slow_ma: List[float] = field(default_factory=list)
    reg_ma: List[float] = field(default_factory=list)
    tema1_cache: Dict[int, float] = field(default_factory=dict)
    tema2_cache: Dict[int, float] = field(default_factory=dict)
    tema3_cache: Dict[int, float] = field(default_factory=dict)

    # ATR intraday
    atr_series: List[float] = field(default_factory=list)

    # DMI / ADX
    di_plus: List[float] = field(default_factory=list)
    di_minus: List[float] = field(default_factory=list)
    adx_raw: List[float] = field(default_factory=list)
    adx_s: List[float] = field(default_factory=list)
    adx_base_series: List[float] = field(default_factory=list)

    # RSI
    rsi_series: List[float] = field(default_factory=list)
    rsi_avg_gain_series: List[float] = field(default_factory=list)
    rsi_avg_loss_series: List[float] = field(default_factory=list)

    # Volume
    vol_ma_series: List[float] = field(default_factory=list)
    vol_ratio_series: List[float] = field(default_factory=list)

    # Market structure
    struct_state: int = 0
    last_hh: float = math.nan
    last_ll: float = math.nan
    last_hl: float = math.nan
    last_lh: float = math.nan
    choch_warning: bool = False

    ph_series: List[float] = field(default_factory=list)
    pl_series: List[float] = field(default_factory=list)
    adx_low_series: List[bool] = field(default_factory=list)
    bos_buf_series: List[float] = field(default_factory=list)
    bos_up_series: List[bool] = field(default_factory=list)
    bos_dn_series: List[bool] = field(default_factory=list)

    struct_ok_long_series: List[bool] = field(default_factory=list)
    struct_ok_short_series: List[bool] = field(default_factory=list)
    struct_ok_long_relaxed_series: List[bool] = field(default_factory=list)
    struct_ok_short_relaxed_series: List[bool] = field(default_factory=list)

    # Daily aggregated bars
    current_day: Optional[datetime.date] = None
    day_high: float = math.nan
    day_low: float = math.nan
    day_close: float = math.nan

    daily_highs: List[float] = field(default_factory=list)
    daily_lows: List[float] = field(default_factory=list)
    daily_closes: List[float] = field(default_factory=list)

    # Daily ATR / EMA20/50 / rolling HH/LL / state
    atr_d_series: List[float] = field(default_factory=list)
    d20_series: List[float] = field(default_factory=list)
    d50_series: List[float] = field(default_factory=list)
    roll_hh_d_series: List[float] = field(default_factory=list)
    roll_ll_d_series: List[float] = field(default_factory=list)
    d_bias_series: List[int] = field(default_factory=list)
    d_state_series: List[int] = field(default_factory=list)

    # Daily OK flags
    daily_up_ok: bool = True
    daily_down_ok: bool = True
    daily_state: int = 0
    daily_trend_up_ok: bool = True
    daily_trend_down_ok: bool = True

    # Position / trade state
    equity: float = math.nan
    position_size: float = 0.0
    prev_position_size: float = 0.0
    position_avg_price: float = math.nan

    entry_price_l: float = math.nan
    entry_price_s: float = math.nan
    atr_at_entry_l: float = math.nan
    atr_at_entry_s: float = math.nan
    tp1_hit_l: bool = False
    tp1_hit_s: bool = False
    dyn_stop_l: float = math.nan
    dyn_stop_s: float = math.nan
    inited_l: bool = False
    inited_s: bool = False

    # Weak exit barssince state
    last_not_exitweak_l_idx: int = 0
    last_not_exitweak_s_idx: int = 0

    # Cooldown
    last_exit_bar: Optional[int] = None
    in_cooldown: bool = False

    # Entry signals for debug / exits
    should_long_now: bool = False
    should_short_now: bool = False
    breakout_long: bool = False
    breakout_short: bool = False
    pullback_long: bool = False
    pullback_short: bool = False
    early_long: bool = False
    early_short: bool = False
    cross_long: bool = False
    cross_short: bool = False

    # Cached last ATR for sizing
    atr_last: float = math.nan


# ============================================================
# RADAR CORE
# ============================================================

class RadarCore:
    """
    RADAR v2 Python Core – Self-contained.

    شامل:
      - اندیکاتورها
      - daily bias + BOS
      - market structure + BOS/CHOCH
      - entry logic (breakout/pullback/early/cross)
      - exit logic (TP1/Trail/BE/Weak Exit/Struct tightening)
      - تولید اکشن‌ها:
          entries / exits / modifications
    """

    def __init__(self, cfg: Optional[RadarCoreConfig] = None):
        self.cfg = cfg or RadarCoreConfig()
        self.state = RadarState()

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def on_bar(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        """
        bar dict expected keys:
          - timestamp
          - open, high, low, close
          - volume
          - equity
          - position_size
          - position_avg_price
        """
        s = self.state
        cfg = self.cfg

        s.bar_index += 1
        idx = s.bar_index

        # --- timestamp
        ts = self._to_timestamp(bar["timestamp"])
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        v = float(bar.get("volume", 0.0))

        # position/equity
        equity = float(bar.get("equity", math.nan))
        pos_size = float(bar.get("position_size", 0.0))
        pos_avg = float(bar.get("position_avg_price", math.nan))

        s.prev_position_size = s.position_size
        s.position_size = pos_size
        s.position_avg_price = pos_avg
        s.equity = equity

        # --- store OHLCV
        s.timestamps.append(ts)
        s.opens.append(o)
        s.highs.append(h)
        s.lows.append(l)
        s.closes.append(c)
        s.volumes.append(v)

        # ========================================================
        # Phase 2 – Indicators
        # ========================================================
        fast_ma = self._update_ma(s.fast_ma, s.closes, cfg.ma_fast_len, cfg.ma_type)
        slow_ma = self._update_ma(s.slow_ma, s.closes, cfg.ma_slow_len, cfg.ma_type)
        reg_ma = self._update_ma(s.reg_ma, s.closes, cfg.ma_reg_len, cfg.ma_type)

        atr = self._update_atr()
        s.atr_last = atr

        di_plus, di_minus, adx_raw, adx_s, adx_base = self._update_dmi_adx()
        rsi = self._update_rsi()
        vol_ma, vol_ratio = self._update_volume_indicators()

        # ========================================================
        # Phase 4 – Daily Filters
        # ========================================================
        daily_info = self._update_daily_filters(
            ts=ts,
            h=h,
            l=l,
            c=c,
        )
        d_state = daily_info["d_state"]
        daily_ok_long = daily_info["daily_ok_long"]
        daily_ok_short = daily_info["daily_ok_short"]
        trend_ok_long = daily_info["trend_ok_long"]
        trend_ok_short = daily_info["trend_ok_short"]
        s.daily_state = d_state
        s.daily_up_ok = trend_ok_long
        s.daily_down_ok = trend_ok_short
        s.daily_trend_up_ok = trend_ok_long
        s.daily_trend_down_ok = trend_ok_short

        # ========================================================
        # Phase 3 – Market Structure (needs ATR, ADX, daily)
        # ========================================================
        self._update_market_structure(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            atr=atr,
            adx_s=adx_s,
            adx_base=adx_base,
            c=c,
            h=h,
            l=l,
            d_state=d_state,
            daily_ok_long=daily_ok_long,
            daily_ok_short=daily_ok_short,
        )

        # Cooldown
        if cfg.use_cooldown and s.last_exit_bar is not None:
            s.in_cooldown = (idx - s.last_exit_bar) < cfg.cooldown_bars
        else:
            s.in_cooldown = False

        # ========================================================
        # Phase 5 – Entry Signals
        # ========================================================
        entry_signals = self._compute_entry_signals(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            reg_ma=reg_ma,
            diplus=di_plus,
            diminus=di_minus,
            adx_s=adx_s,
            adx_base=adx_base,
            rsi=rsi,
            vol_ratio=vol_ratio,
            atr=atr,
            c=c,
            h=h,
            l=l,
        )

        s.should_long_now = entry_signals["should_long"]
        s.should_short_now = entry_signals["should_short"]
        s.breakout_long = entry_signals["breakout_long"]
        s.breakout_short = entry_signals["breakout_short"]
        s.pullback_long = entry_signals["pullback_long"]
        s.pullback_short = entry_signals["pullback_short"]
        s.early_long = entry_signals["early_long"]
        s.early_short = entry_signals["early_short"]
        s.cross_long = entry_signals["cross_long"]
        s.cross_short = entry_signals["cross_short"]

        # ========================================================
        # Phase 6 – Exit Logic
        # ========================================================
        exit_info = self._update_exit_logic(
            slow_ma=slow_ma,
            di_plus=di_plus,
            di_minus=di_minus,
            adx_s=adx_s,
            atr=atr,
            high=h,
            low=l,
            close=c,
        )

        # ========================================================
        # Phase 7 – Build Actions
        # ========================================================
        actions = self._build_actions(bar=bar, exit_info=exit_info)

        # update prev position for next bar
        s.prev_position_size = s.position_size

        return {
            "entries": actions["entries"],
            "exits": actions["exits"],
            "modifications": actions["modifications"],
            "debug": {
                "bar_index": s.bar_index,
                "price": {"o": o, "h": h, "l": l, "c": c},
                "volume": v,
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "reg_ma": reg_ma,
                "atr": atr,
                "di_plus": di_plus,
                "di_minus": di_minus,
                "adx_raw": adx_raw,
                "adx_s": adx_s,
                "adx_base": adx_base,
                "rsi": rsi,
                "vol_ma": vol_ma,
                "vol_ratio": vol_ratio,
                "struct_state": s.struct_state,
                "last_hh": s.last_hh,
                "last_ll": s.last_ll,
                "last_hl": s.last_hl,
                "last_lh": s.last_lh,
                "choch_warning": s.choch_warning,
                "daily_state": s.daily_state,
                "daily_ok_long": s.daily_up_ok,
                "daily_ok_short": s.daily_down_ok,
                "struct_ok_long": s.struct_ok_long_series[-1] if s.struct_ok_long_series else True,
                "struct_ok_short": s.struct_ok_short_series[-1] if s.struct_ok_short_series else True,
                "struct_ok_long_relaxed": s.struct_ok_long_relaxed_series[-1] if s.struct_ok_long_relaxed_series else True,
                "struct_ok_short_relaxed": s.struct_ok_short_relaxed_series[-1] if s.struct_ok_short_relaxed_series else True,
                "should_long": s.should_long_now,
                "should_short": s.should_short_now,
                "breakout_long": s.breakout_long,
                "breakout_short": s.breakout_short,
                "pullback_long": s.pullback_long,
                "pullback_short": s.pullback_short,
                "early_long": s.early_long,
                "early_short": s.early_short,
                "cross_long": s.cross_long,
                "cross_short": s.cross_short,
                "tp1L": exit_info["tp1L"],
                "tp1S": exit_info["tp1S"],
                "dynStopL": exit_info["dynStopL"],
                "dynStopS": exit_info["dynStopS"],
                "exitWeakL": exit_info["exitWeakL"],
                "exitWeakS": exit_info["exitWeakS"],
                "tp1HitL": exit_info["tp1HitL"],
                "tp1HitS": exit_info["tp1HitS"],
            },
        }

    # ------------------------------------------------------------
    # Helpers – time
    # ------------------------------------------------------------
    @staticmethod
    def _to_timestamp(ts: Union[int, float, datetime]) -> int:
        if isinstance(ts, datetime):
            return int(ts.timestamp())
        if isinstance(ts, (int, float)):
            return int(ts / 1000) if ts > 10**11 else int(ts)
        raise TypeError(f"Unsupported timestamp type: {type(ts)}")

    # ------------------------------------------------------------
    # Indicator Helpers
    # ------------------------------------------------------------
    @staticmethod
    def _ema(prev: Optional[float], value: float, length: int) -> float:
        if length <= 1:
            return value
        alpha = 2.0 / (length + 1.0)
        if prev is None or math.isnan(prev):
            return value
        return alpha * value + (1.0 - alpha) * prev

    @staticmethod
    def _sma(series: List[float], length: int) -> float:
        if length <= 0 or len(series) < length:
            return math.nan
        window = series[-length:]
        return sum(window) / float(length)

    @staticmethod
    def _wma(series: List[float], length: int) -> float:
        if length <= 0 or len(series) < length:
            return math.nan
        window = series[-length:]
        weights = list(range(1, length + 1))
        s_val = sum(v * w for v, w in zip(window, weights))
        return s_val / float(sum(weights))

    def _hma(self, series: List[float], length: int) -> float:
        if length <= 0 or len(series) < length:
            return math.nan

        half_len = max(1, length // 2)
        sqrt_len = max(1, int(math.sqrt(length)))

        temps: List[float] = []
        for i in range(sqrt_len):
            sub_end = len(series) - i
            if sub_end < length:
                break
            sub_series = series[:sub_end]
            wma_half = self._wma(sub_series, half_len)
            wma_full = self._wma(sub_series, length)
            if math.isnan(wma_half) or math.isnan(wma_full):
                temp = math.nan
            else:
                temp = 2.0 * wma_half - wma_full
            temps.append(temp)

        if len(temps) < sqrt_len:
            return math.nan

        temps = list(reversed(temps))
        return self._wma(temps, sqrt_len)

    def _update_ma(self, series: List[float], src: List[float], length: int, ma_type: str) -> float:
        if len(src) == 0:
            series.append(math.nan)
            return math.nan

        price = src[-1]
        ma_type_upper = (ma_type or "EMA").upper()

        if ma_type_upper == "EMA":
            prev = series[-1] if series else None
            val = self._ema(prev, price, length)
        elif ma_type_upper == "TEMA":
            prev1 = self.state.tema1_cache.get(length, math.nan)
            prev2 = self.state.tema2_cache.get(length, math.nan)
            prev3 = self.state.tema3_cache.get(length, math.nan)

            ema1 = self._ema(prev1 if not math.isnan(prev1) else None, price, length)
            ema2 = self._ema(prev2 if not math.isnan(prev2) else None, ema1, length)
            ema3 = self._ema(prev3 if not math.isnan(prev3) else None, ema2, length)

            self.state.tema1_cache[length] = ema1
            self.state.tema2_cache[length] = ema2
            self.state.tema3_cache[length] = ema3

            val = 3.0 * (ema1 - ema2) + ema3
        elif ma_type_upper == "HMA":
            val = self._hma(src, length)
        else:
            val = self._sma(src, length)

        series.append(val)
        return val

    def _update_atr(self) -> float:
        s = self.state
        length = self.cfg.atr_len
        n = len(s.closes)

        if n == 1:
            s.atr_series.append(math.nan)
            return math.nan

        high = s.highs[-1]
        low = s.lows[-1]
        prev_close = s.closes[-2]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

        if len(s.atr_series) == 0 or math.isnan(s.atr_series[-1]):
            trs: List[float] = []
            for i in range(n):
                hi = s.highs[i]
                lo = s.lows[i]
                pc = s.closes[i - 1] if i > 0 else s.closes[i]
                trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))

            window = min(length, len(trs))
            atr = sum(trs[-window:]) / float(window) if window > 0 else math.nan
        else:
            prev_atr = s.atr_series[-1]
            atr = (prev_atr * (length - 1) + tr) / float(length)

        s.atr_series.append(atr)
        return atr

    def _update_dmi_adx(self):
        s = self.state
        cfg = self.cfg
        length = cfg.adx_len
        n = len(s.closes)

        if n < 2:
            s.di_plus.append(math.nan)
            s.di_minus.append(math.nan)
            s.adx_raw.append(math.nan)
            s.adx_s.append(math.nan)
            s.adx_base_series.append(cfg.adx_th_fixed)
            return math.nan, math.nan, math.nan, math.nan, cfg.adx_th_fixed

        high = s.highs[-1]
        low = s.lows[-1]
        prev_high = s.highs[-2]
        prev_low = s.lows[-2]

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = max(up_move, 0.0) if up_move > down_move else 0.0
        minus_dm = max(down_move, 0.0) if down_move > up_move else 0.0

        atr = s.atr_series[-1] if s.atr_series else math.nan
        if atr is None or atr <= 0 or math.isnan(atr):
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = 100.0 * plus_dm / atr
            minus_di = 100.0 * minus_dm / atr

        s.di_plus.append(plus_di)
        s.di_minus.append(minus_di)

        denom = plus_di + minus_di
        if denom == 0:
            dx = 0.0
        else:
            dx = 100.0 * abs(plus_di - minus_di) / denom

        if len(s.adx_raw) < length:
            s.adx_raw.append(dx)
            adx_val = self._sma(s.adx_raw, min(length, len(s.adx_raw)))
        else:
            prev_adx = s.adx_raw[-1]
            adx_val = (prev_adx * (length - 1) + dx) / float(length)
            s.adx_raw.append(adx_val)

        adx_s = self._sma(s.adx_raw, 3)
        s.adx_s.append(adx_s)

        if cfg.use_dyn_adx:
            base_raw = self._sma(s.adx_raw, 30)
            adx_base = base_raw * cfg.adx_dyn_k if not math.isnan(base_raw) else cfg.adx_th_fixed
        else:
            adx_base = cfg.adx_th_fixed

        s.adx_base_series.append(adx_base)
        return plus_di, minus_di, adx_val, adx_s, adx_base

    def _update_rsi(self) -> float:
        s = self.state
        length = self.cfg.rsi_len
        closes = s.closes

        if len(closes) < 2:
            s.rsi_series.append(math.nan)
            s.rsi_avg_gain_series.append(math.nan)
            s.rsi_avg_loss_series.append(math.nan)
            return math.nan

        change = closes[-1] - closes[-2]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if not s.rsi_avg_gain_series or math.isnan(s.rsi_avg_gain_series[-1]):
            if len(closes) - 1 < length:
                s.rsi_series.append(math.nan)
                s.rsi_avg_gain_series.append(math.nan)
                s.rsi_avg_loss_series.append(math.nan)
                return math.nan

            gains = 0.0
            losses = 0.0
            for i in range(len(closes) - length, len(closes)):
                ch = closes[i] - closes[i - 1]
                if ch > 0:
                    gains += ch
                else:
                    losses -= ch
            avg_gain = gains / length
            avg_loss = losses / length
        else:
            prev_avg_gain = s.rsi_avg_gain_series[-1]
            prev_avg_loss = s.rsi_avg_loss_series[-1]
            avg_gain = (prev_avg_gain * (length - 1) + gain) / length
            avg_loss = (prev_avg_loss * (length - 1) + loss) / length

        s.rsi_avg_gain_series.append(avg_gain)
        s.rsi_avg_loss_series.append(avg_loss)

        if avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 0.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)

        s.rsi_series.append(rsi)
        return rsi

    def _update_volume_indicators(self):
        s = self.state
        length = self.cfg.vol_len
        vol_ma = self._sma(s.volumes, length)
        vol = s.volumes[-1]

        if vol_ma == 0 or math.isnan(vol_ma):
            vol_ratio = 0.0
        else:
            vol_ratio = vol / vol_ma

        s.vol_ma_series.append(vol_ma)
        s.vol_ratio_series.append(vol_ratio)
        return vol_ma, vol_ratio

    # ------------------------------------------------------------
    # Daily Filters
    # ------------------------------------------------------------
    def _update_daily_filters(self, ts: int, h: float, l: float, c: float) -> Dict[str, Any]:
        cfg = self.cfg
        s = self.state

        current_date = datetime.utcfromtimestamp(ts).date()

        if s.current_day is None:
            s.current_day = current_date
            s.day_high = h
            s.day_low = l
            s.day_close = c
        else:
            if current_date == s.current_day:
                s.day_high = max(s.day_high, h)
                s.day_low = min(s.day_low, l)
                s.day_close = c
            else:
                # finalize previous
                self._finalize_previous_day()
                s.current_day = current_date
                s.day_high = h
                s.day_low = l
                s.day_close = c

        self._update_daily_derived_series()

        d20 = s.d20_series[-1] if s.d20_series else math.nan
        d50 = s.d50_series[-1] if s.d50_series else math.nan

        if math.isnan(d20) or math.isnan(d50):
            daily_up = False
            daily_down = False
        else:
            daily_up = d20 > (d50 * cfg.daily_soft_k)
            daily_down = d20 < (d50 / cfg.daily_soft_k)

        trend_ok_long = (not cfg.use_trend_filter) or daily_up
        trend_ok_short = (not cfg.use_trend_filter) or daily_down

        if math.isnan(d20) or math.isnan(d50):
            d_bias = 0
        else:
            if d20 > d50:
                d_bias = 1
            elif d20 < d50:
                d_bias = -1
            else:
                d_bias = 0

        s.d_bias_series.append(d_bias)

        atr_d = s.atr_d_series[-1] if s.atr_d_series else math.nan
        roll_hh_d = s.roll_hh_d_series[-1] if s.roll_hh_d_series else math.nan
        roll_ll_d = s.roll_ll_d_series[-1] if s.roll_ll_d_series else math.nan

        buf_d = (atr_d if not math.isnan(atr_d) else 0.0) * cfg.bos_buf_atr

        if not math.isnan(roll_hh_d):
            d_bos_up = c > roll_hh_d + buf_d
        else:
            d_bos_up = False

        if not math.isnan(roll_ll_d):
            d_bos_dn = c < roll_ll_d - buf_d
        else:
            d_bos_dn = False

        if d_bos_up:
            d_state = 1
        elif d_bos_dn:
            d_state = -1
        else:
            d_state = d_bias

        s.d_state_series.append(d_state)

        daily_ok_long = (d_state > 0) or (cfg.allow_daily_neutral and d_state == 0)
        daily_ok_short = (d_state < 0) or (cfg.allow_daily_neutral and d_state == 0)

        return {
            "d_state": d_state,
            "daily_ok_long": daily_ok_long,
            "daily_ok_short": daily_ok_short,
            "trend_ok_long": trend_ok_long,
            "trend_ok_short": trend_ok_short,
        }

    def _finalize_previous_day(self):
        s = self.state
        if s.current_day is None:
            return

        s.daily_highs.append(s.day_high)
        s.daily_lows.append(s.day_low)
        s.daily_closes.append(s.day_close)

        self._update_daily_atr()

        d_close = s.daily_closes[-1]
        prev_d20 = s.d20_series[-1] if s.d20_series else None
        prev_d50 = s.d50_series[-1] if s.d50_series else None

        d20 = self._ema(prev_d20, d_close, 20)
        d50 = self._ema(prev_d50, d_close, 50)

        s.d20_series.append(d20)
        s.d50_series.append(d50)

        window = max(1, int(round(self.cfg.swing_len * 1.5)))
        if len(s.daily_highs) >= window:
            roll_hh = max(s.daily_highs[-window:])
            roll_ll = min(s.daily_lows[-window:])
        else:
            roll_hh = s.daily_highs[-1]
            roll_ll = s.daily_lows[-1]

        s.roll_hh_d_series.append(roll_hh)
        s.roll_ll_d_series.append(roll_ll)

    def _update_daily_atr(self):
        s = self.state
        length = 14
        n = len(s.daily_closes)
        if n == 0:
            return

        high = s.daily_highs[-1]
        low = s.daily_lows[-1]
        prev_close = s.daily_closes[-2] if n >= 2 else s.daily_closes[-1]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

        if len(s.atr_d_series) == 0 or math.isnan(s.atr_d_series[-1]):
            total_bars = n
            trs = []
            for i in range(total_bars):
                hi = s.daily_highs[i]
                lo = s.daily_lows[i]
                pc = s.daily_closes[i - 1] if i > 0 else s.daily_closes[i]
                trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))

            window = min(length, len(trs))
            atr_seed = sum(trs[-window:]) / float(window)
            atr_d = atr_seed
        else:
            prev_atr_d = s.atr_d_series[-1]
            atr_d = (prev_atr_d * (length - 1) + tr) / float(length)

        s.atr_d_series.append(atr_d)

    def _update_daily_derived_series(self):
        # فعلاً تمام محاسبات داخل _finalize_previous_day انجام شده
        return

    # ------------------------------------------------------------
    # Market Structure
    # ------------------------------------------------------------
    def _pivot_high(self, idx: int, length: int) -> float:
        s = self.state
        if idx - length < 0 or idx + length >= len(s.highs):
            return math.nan
        center = s.highs[idx]
        left = s.highs[idx - length: idx]
        right = s.highs[idx + 1: idx + length + 1]
        if center == max(left + [center] + right):
            return center
        return math.nan

    def _pivot_low(self, idx: int, length: int) -> float:
        s = self.state
        if idx - length < 0 or idx + length >= len(s.lows):
            return math.nan
        center = s.lows[idx]
        left = s.lows[idx - length: idx]
        right = s.lows[idx + 1: idx + length + 1]
        if center == min(left + [center] + right):
            return center
        return math.nan

    def _update_market_structure(
        self,
        fast_ma: float,
        slow_ma: float,
        atr: float,
        adx_s: float,
        adx_base: float,
        c: float,
        h: float,
        l: float,
        d_state: int,
        daily_ok_long: bool,
        daily_ok_short: bool,
    ):
        cfg = self.cfg
        s = self.state
        idx = s.bar_index
        swing = cfg.swing_len

        ph = self._pivot_high(idx, swing)
        pl = self._pivot_low(idx, swing)

        s.ph_series.append(ph)
        s.pl_series.append(pl)

        if not math.isnan(ph):
            s.last_hh = ph
            if s.struct_state == -1:
                s.last_lh = ph

        if not math.isnan(pl):
            s.last_ll = pl
            if s.struct_state == 1:
                s.last_hl = pl

        adx_threshold = 25
        adx_low = (adx_s < adx_threshold) if not math.isnan(adx_s) else False
        s.adx_low_series.append(adx_low)

        buf_mult = 1.4 if adx_low else 0.8
        bos_buf = cfg.bos_buf_atr * (atr if not math.isnan(atr) else 0.0) * buf_mult
        s.bos_buf_series.append(bos_buf)

        if not math.isnan(s.last_hh):
            bos_up = c > s.last_hh + bos_buf
        else:
            bos_up = False

        if not math.isnan(s.last_ll):
            bos_dn = c < s.last_ll - bos_buf
        else:
            bos_dn = False

        s.bos_up_series.append(bos_up)
        s.bos_dn_series.append(bos_dn)

        if bos_up:
            s.struct_state = 1
            s.last_hl = math.nan
        if bos_dn:
            s.struct_state = -1
            s.last_lh = math.nan

        choch_up = (not math.isnan(s.last_hl)) and (c < s.last_hl)
        choch_dn = (not math.isnan(s.last_lh)) and (c > s.last_lh)
        s.choch_warning = cfg.use_choch_soft and (choch_up or choch_dn)

        strict_long = cfg.adx_soft_strict and adx_low
        strict_short = cfg.strict_short or (cfg.adx_soft_strict and adx_low)

        if not cfg.use_struct:
            struct_ok_long = True
            struct_ok_short = True
        else:
            struct_ok_long = s.struct_state == 1 if strict_long else s.struct_state >= 0
            struct_ok_short = s.struct_state == -1 if strict_short else s.struct_state <= 0

        s.struct_ok_long_series.append(struct_ok_long)
        s.struct_ok_short_series.append(struct_ok_short)

        if idx >= swing:
            pre_hh = max(s.highs[idx - swing: idx + 1])
            pre_ll = min(s.lows[idx - swing: idx + 1])
        else:
            pre_hh = h
            pre_ll = l

        pre_bos_up = c > pre_hh + (atr * cfg.bos_buf_atr * 0.50 if not math.isnan(atr) else 0.0)
        pre_bos_dn = c < pre_ll - (atr * cfg.bos_buf_atr * 0.50 if not math.isnan(atr) else 0.0)

        if cfg.use_range_bypass and adx_low and pre_bos_up:
            if not cfg.use_daily_in_bypass or daily_ok_long:
                relax_long = True
            else:
                relax_long = False
        else:
            relax_long = False

        if cfg.use_range_bypass and adx_low and pre_bos_dn:
            if not cfg.use_daily_in_bypass or daily_ok_short:
                relax_short = True
            else:
                relax_short = False
        else:
            relax_short = False

        struct_ok_long_relaxed = struct_ok_long or relax_long
        struct_ok_short_relaxed = struct_ok_short or relax_short

        s.struct_ok_long_relaxed_series.append(struct_ok_long_relaxed)
        s.struct_ok_short_relaxed_series.append(struct_ok_short_relaxed)

    # ------------------------------------------------------------
    # Entry Signals
    # ------------------------------------------------------------
    def _prev(self, val: float) -> float:
        # کمکی برای require_adx_up (در اینجا فقط تقریب)
        # اگر نیاز شد می‌توان آریه adx_s را استفاده کرد
        return val  # در این سطح کاربرد ندارد، فقط برای سازگاری

    def _cross(self, a: float, b: float) -> bool:
        s = self.state
        if len(s.fast_ma) < 2 or len(s.slow_ma) < 2:
            return False
        prev_a = s.fast_ma[-2]
        prev_b = s.slow_ma[-2]
        return prev_a <= prev_b and a > b

    def _crossunder(self, a: float, b: float) -> bool:
        s = self.state
        if len(s.fast_ma) < 2 or len(s.slow_ma) < 2:
            return False
        prev_a = s.fast_ma[-2]
        prev_b = s.slow_ma[-2]
        return prev_a >= prev_b and a < b

    def _compute_entry_signals(
        self,
        fast_ma,
        slow_ma,
        reg_ma,
        diplus,
        diminus,
        adx_s,
        adx_base,
        rsi,
        vol_ratio,
        atr,
        c,
        h,
        l,
    ):
        cfg = self.cfg
        s = self.state

        trend_long = fast_ma > slow_ma
        trend_short = fast_ma < slow_ma

        regime_long = trend_long and (not cfg.use_regime_filter or c > reg_ma)
        regime_short = trend_short and (not cfg.use_regime_filter or c < reg_ma)

        dir_long = diplus > diminus
        dir_short = diminus > diplus

        rsi_long_ok = (not cfg.use_rsi) or (rsi > cfg.rsi_th_long)
        rsi_short_ok = (not cfg.use_rsi) or (rsi < (100 - cfg.rsi_th_long))

        vol_ok_long = (not cfg.use_vol) or (vol_ratio > cfg.vol_k_long)
        vol_ok_short = (not cfg.use_vol) or (vol_ratio > cfg.vol_k_long)

        prev_adx = s.adx_s[-2] if len(s.adx_s) >= 2 else math.nan
        adx_rising_ok = (not cfg.require_adx_up) or (
            (not math.isnan(prev_adx)) and (adx_s > prev_adx)
        )
        power_ok = (adx_s > adx_base) and adx_rising_ok

        rr_ok_long = cfg.tp1_ratio >= 1.6
        rr_ok_short = cfg.tp1_ratio >= 1.6

        # LONG
        breakout_long = (
            regime_long
            and power_ok
            and dir_long
            and rsi_long_ok
            and vol_ok_long
            and s.daily_up_ok
            and (c > fast_ma)
            and (c > slow_ma)
            and not s.in_cooldown
            and rr_ok_long
        )

        pullback_long = (
            regime_long
            and power_ok
            and dir_long
            and rsi_long_ok
            and vol_ok_long
            and s.daily_up_ok
            and (l <= fast_ma and c > fast_ma)
            and not s.in_cooldown
            and rr_ok_long
        )

        early_long = (
            cfg.use_early_entry
            and not s.in_cooldown
            and trend_long
            and (rsi > cfg.early_rsi_long)
            and (adx_s > cfg.early_adx)
            and s.daily_up_ok
            and not (cfg.use_choch_soft and s.choch_warning)
        )

        # SHORT
        guard_short = adx_s > (adx_base + 1.5)

        breakout_short = (
            regime_short
            and power_ok
            and dir_short
            and rsi_short_ok
            and vol_ok_short
            and s.daily_down_ok
            and (c < fast_ma)
            and (c < slow_ma)
            and not s.in_cooldown
            and rr_ok_short
            and guard_short
        )

        pullback_short = (
            regime_short
            and power_ok
            and dir_short
            and rsi_short_ok
            and vol_ok_short
            and s.daily_down_ok
            and (h >= fast_ma and c < fast_ma)
            and not s.in_cooldown
            and rr_ok_short
            and guard_short
        )

        early_short = (
            cfg.use_early_entry
            and not s.in_cooldown
            and trend_short
            and (rsi < cfg.early_rsi_short)
            and (adx_s > cfg.early_adx)
            and s.daily_down_ok
            and guard_short
            and not (cfg.use_choch_soft and s.choch_warning)
        )

        cross_up = self._cross(fast_ma, slow_ma)
        cross_down = self._crossunder(fast_ma, slow_ma)

        cross_long = (
            cfg.use_cross_entry
            and cross_up
            and ((not cfg.cross_need_adx) or power_ok)
            and ((not cfg.cross_need_daily) or s.daily_up_ok)
            and ((not cfg.cross_need_regime) or (c > reg_ma))
            and not s.in_cooldown
            and rr_ok_long
        )

        cross_short = (
            cfg.use_cross_entry
            and cross_down
            and ((not cfg.cross_need_adx) or power_ok)
            and ((not cfg.cross_need_daily) or s.daily_down_ok)
            and ((not cfg.cross_need_regime) or (c < reg_ma))
            and not s.in_cooldown
            and rr_ok_short
            and guard_short
        )

        struct_ok_long_relaxed = (
            s.struct_ok_long_relaxed_series[-1]
            if s.struct_ok_long_relaxed_series
            else True
        )
        struct_ok_short_relaxed = (
            s.struct_ok_short_relaxed_series[-1]
            if s.struct_ok_short_relaxed_series
            else True
        )

        final_long = (
            (breakout_long or pullback_long or cross_long or early_long)
            and struct_ok_long_relaxed
        )
        final_short = (
            (breakout_short or pullback_short or cross_short or early_short)
            and struct_ok_short_relaxed
        )

        return {
            "should_long": final_long,
            "should_short": final_short,
            "breakout_long": breakout_long,
            "breakout_short": breakout_short,
            "pullback_long": pullback_long,
            "pullback_short": pullback_short,
            "early_long": early_long,
            "early_short": early_short,
            "cross_long": cross_long,
            "cross_short": cross_short,
        }

    # ------------------------------------------------------------
    # Exit Logic
    # ------------------------------------------------------------
    def _update_exit_logic(
        self,
        slow_ma: float,
        di_plus: float,
        di_minus: float,
        adx_s: float,
        atr: float,
        high: float,
        low: float,
        close: float,
    ):
        cfg = self.cfg
        s = self.state
        idx = s.bar_index

        pos_size = s.position_size
        prev_pos_size = s.prev_position_size
        pos_avg = s.position_avg_price

        # ورود جدید Long
        if pos_size > 0 and prev_pos_size <= 0:
            s.entry_price_l = pos_avg if not math.isnan(pos_avg) else close
            s.atr_at_entry_l = atr
            s.tp1_hit_l = False
            s.dyn_stop_l = math.nan
            s.inited_l = True

        # ورود جدید Short
        if pos_size < 0 and prev_pos_size >= 0:
            s.entry_price_s = pos_avg if not math.isnan(pos_avg) else close
            s.atr_at_entry_s = atr
            s.tp1_hit_s = False
            s.dyn_stop_s = math.nan
            s.inited_s = True

        # بسته شدن کامل پوزیشن
        if pos_size == 0 and prev_pos_size != 0:
            s.last_exit_bar = idx
            s.tp1_hit_l = False
            s.tp1_hit_s = False
            s.entry_price_l = math.nan
            s.entry_price_s = math.nan
            s.atr_at_entry_l = math.nan
            s.atr_at_entry_s = math.nan
            s.inited_l = False
            s.inited_s = False
            s.dyn_stop_l = math.nan
            s.dyn_stop_s = math.nan

        tp1L = math.nan
        tp2TrailL = math.nan
        exitWeakRawL = False
        exitWeakL = False

        if pos_size > 0 and s.inited_l and not math.isnan(s.entry_price_l) and not math.isnan(s.atr_at_entry_l):
            tp1L = s.entry_price_l + (cfg.atr_mult_sl * s.atr_at_entry_l * cfg.tp1_ratio)
            tp2TrailL = close - (cfg.tp2_trail_atr_l * atr) if not math.isnan(atr) else math.nan

            if (not s.tp1_hit_l) and (high >= tp1L):
                s.tp1_hit_l = True
                if cfg.use_break_even:
                    be_level = s.entry_price_l - cfg.be_buf * atr
                    if math.isnan(s.dyn_stop_l):
                        s.dyn_stop_l = be_level
                    else:
                        s.dyn_stop_l = max(s.dyn_stop_l, be_level)

            if s.tp1_hit_l and not math.isnan(tp2TrailL):
                if math.isnan(s.dyn_stop_l):
                    s.dyn_stop_l = tp2TrailL
                else:
                    s.dyn_stop_l = max(s.dyn_stop_l, tp2TrailL)

            if s.struct_state == -1 and not math.isnan(atr):
                tighten_level = close - 1.6 * atr
                if math.isnan(s.dyn_stop_l):
                    s.dyn_stop_l = tighten_level
                else:
                    s.dyn_stop_l = max(s.dyn_stop_l, tighten_level)

            exitWeakRawL = (close < slow_ma) or (adx_s < 18) or (di_plus < di_minus)

            last_not_idx = s.last_not_exitweak_l_idx
            if not exitWeakRawL:
                s.last_not_exitweak_l_idx = idx
                last_not_idx = idx
            bars_since_not = idx - last_not_idx
            exitWeakL = bars_since_not >= cfg.weak_confirm_bars

            if s.inited_l and exitWeakL:
                candidates = []
                if not math.isnan(s.dyn_stop_l):
                    candidates.append(s.dyn_stop_l)
                if not math.isnan(slow_ma) and not math.isnan(atr):
                    candidates.append(slow_ma - 0.3 * atr)
                if not math.isnan(s.entry_price_l):
                    candidates.append(s.entry_price_l)
                if candidates:
                    s.dyn_stop_l = max(candidates)

        tp1S = math.nan
        tp2TrailS = math.nan
        exitWeakRawS = False
        exitWeakS = False

        if pos_size < 0 and s.inited_s and not math.isnan(s.entry_price_s) and not math.isnan(s.atr_at_entry_s):
            tp1S = s.entry_price_s - (cfg.atr_mult_sl * s.atr_at_entry_s * cfg.tp1_ratio)
            tp2TrailS = close + (cfg.tp2_trail_atr_s * atr) if not math.isnan(atr) else math.nan

            if (not s.tp1_hit_s) and (low <= tp1S):
                s.tp1_hit_s = True
                if cfg.use_break_even:
                    be_level_s = s.entry_price_s + cfg.be_buf * atr
                    if math.isnan(s.dyn_stop_s):
                        s.dyn_stop_s = be_level_s
                    else:
                        s.dyn_stop_s = min(s.dyn_stop_s, be_level_s)

            if s.tp1_hit_s and not math.isnan(tp2TrailS):
                if math.isnan(s.dyn_stop_s):
                    s.dyn_stop_s = tp2TrailS
                else:
                    s.dyn_stop_s = min(s.dyn_stop_s, tp2TrailS)

            if s.struct_state == 1 and not math.isnan(atr):
                tighten_s = close + 1.6 * atr
                if math.isnan(s.dyn_stop_s):
                    s.dyn_stop_s = tighten_s
                else:
                    s.dyn_stop_s = min(s.dyn_stop_s, tighten_s)

            exitWeakRawS = (close > slow_ma) or (adx_s < 20) or (di_plus > di_minus)

            last_not_idx_s = s.last_not_exitweak_s_idx
            if not exitWeakRawS:
                s.last_not_exitweak_s_idx = idx
                last_not_idx_s = idx
            bars_since_not_s = idx - last_not_idx_s
            exitWeakS = bars_since_not_s >= cfg.weak_confirm_bars

            if s.inited_s and exitWeakS:
                candidates_s = []
                if not math.isnan(s.dyn_stop_s):
                    candidates_s.append(s.dyn_stop_s)
                if not math.isnan(slow_ma) and not math.isnan(atr):
                    tight_s = max(slow_ma, close + 0.2 * atr)
                    candidates_s.append(tight_s)
                if candidates_s:
                    s.dyn_stop_s = min(candidates_s)

        if pos_size == 0:
            dynStopL = math.nan
            dynStopS = math.nan
        else:
            dynStopL = s.dyn_stop_l
            dynStopS = s.dyn_stop_s

        return {
            "tp1L": tp1L,
            "tp1S": tp1S,
            "dynStopL": dynStopL,
            "dynStopS": dynStopS,
            "exitWeakL": exitWeakL,
            "exitWeakS": exitWeakS,
            "tp1HitL": bool(s.tp1_hit_l),
            "tp1HitS": bool(s.tp1_hit_s),
        }

    # ------------------------------------------------------------
    # Build Actions
    # ------------------------------------------------------------
    def _build_actions(self, bar: dict, exit_info: dict) -> dict:
        cfg = self.cfg
        s = self.state

        entries = []
        exits = []
        modifications = []

        equity = s.equity
        pos_size = s.position_size
        atr = s.atr_last

        if math.isnan(atr) or atr <= 0:
            return {"entries": [], "exits": [], "modifications": []}

        risk_cap = equity * cfg.risk_pct if not math.isnan(equity) else 0.0
        risk_den = max(cfg.atr_mult_sl * atr, 1e-10)
        qty_core = max(risk_cap / risk_den, 0.0)

        # LONG ENTRIES
        should_long = s.should_long_now
        early_long = s.early_long
        cross_long = s.cross_long
        breakout_long = s.breakout_long
        pullback_long = s.pullback_long
        struct_ok_long = s.struct_ok_long_series[-1] if s.struct_ok_long_series else True
        struct_ok_long_relaxed = (
            s.struct_ok_long_relaxed_series[-1]
            if s.struct_ok_long_relaxed_series
            else True
        )

        cur_pos_l = pos_size if pos_size > 0 else 0.0

        if (
            early_long
            and struct_ok_long
            and not (cfg.use_choch_soft and s.choch_warning)
        ):
            target_early = qty_core * cfg.early_qty_frac
            add_early = max(target_early - cur_pos_l, 0.0)
            if add_early > 0:
                entries.append(
                    {
                        "id": "RADAR_Early_L",
                        "side": "buy",
                        "type": "market",
                        "qty": add_early,
                        "tag": "early",
                    }
                )
                cur_pos_l += add_early

        long_trigger = (breakout_long or pullback_long or cross_long) and should_long
        if long_trigger and struct_ok_long_relaxed:
            target_qty_l = qty_core
            add_qty_l = max(target_qty_l - cur_pos_l, 0.0)
            if add_qty_l > 0:
                entries.append(
                    {
                        "id": "RADAR_Long",
                        "side": "buy",
                        "type": "market",
                        "qty": add_qty_l,
                        "tag": "core",
                    }
                )

        # SHORT ENTRIES
        should_short = s.should_short_now
        early_short = s.early_short
        cross_short = s.cross_short
        breakout_short = s.breakout_short
        pullback_short = s.pullback_short
        struct_ok_short = s.struct_ok_short_series[-1] if s.struct_ok_short_series else True
        struct_ok_short_relaxed = (
            s.struct_ok_short_relaxed_series[-1]
            if s.struct_ok_short_relaxed_series
            else True
        )

        cur_pos_s = -pos_size if pos_size < 0 else 0.0

        if (
            early_short
            and struct_ok_short
            and not (cfg.use_choch_soft and s.choch_warning)
        ):
            target_early_s = qty_core * cfg.early_qty_frac
            add_early_s = max(target_early_s - cur_pos_s, 0.0)
            if add_early_s > 0:
                entries.append(
                    {
                        "id": "RADAR_Early_S",
                        "side": "sell",
                        "type": "market",
                        "qty": add_early_s,
                        "tag": "early",
                    }
                )
                cur_pos_s += add_early_s

        short_trigger = (breakout_short or pullback_short or cross_short) and should_short
        if short_trigger and struct_ok_short_relaxed:
            target_qty_s = qty_core
            add_qty_s = max(target_qty_s - cur_pos_s, 0.0)
            if add_qty_s > 0:
                entries.append(
                    {
                        "id": "RADAR_Short",
                        "side": "sell",
                        "type": "market",
                        "qty": add_qty_s,
                        "tag": "core",
                    }
                )

        # EXITS
        tp1L = exit_info.get("tp1L", math.nan)
        tp1S = exit_info.get("tp1S", math.nan)
        dynStopL = exit_info.get("dynStopL", math.nan)
        dynStopS = exit_info.get("dynStopS", math.nan)
        exitWeakL = bool(exit_info.get("exitWeakL", False))
        exitWeakS = bool(exit_info.get("exitWeakS", False))

        tp1_pct = cfg.tp1_pct
        trail_pct = max(0, 100 - tp1_pct)

        # LONG EXITS
        if pos_size > 0:
            if exitWeakL:
                exits.append(
                    {
                        "id": "Weak_Exit_L",
                        "mode": "close_full",
                        "side": "sell",
                        "type": "market",
                        "price": None,
                        "qty_pct": 100,
                        "reason": "Weak_Exit_L",
                        "labels": ["RADAR_Long", "RADAR_Early_L", "RADAR_Cross_L"],
                    }
                )
            else:
                if not math.isnan(tp1L):
                    exits.append(
                        {
                            "id": "L-TP1",
                            "mode": "reduce",
                            "side": "sell",
                            "type": "limit",
                            "price": tp1L,
                            "qty_pct": tp1_pct,
                            "reason": "TP1_L",
                        }
                    )
                if not math.isnan(dynStopL):
                    exits.append(
                        {
                            "id": "L-TRAIL",
                            "mode": "reduce_or_close",
                            "side": "sell",
                            "type": "stop",
                            "price": dynStopL,
                            "qty_pct": trail_pct,
                            "reason": "TRAIL_L",
                        }
                    )

        # SHORT EXITS
        if pos_size < 0:
            if exitWeakS:
                exits.append(
                    {
                        "id": "Weak_Exit_S",
                        "mode": "close_full",
                        "side": "buy",
                        "type": "market",
                        "price": None,
                        "qty_pct": 100,
                        "reason": "Weak_Exit_S",
                        "labels": ["RADAR_Short", "RADAR_Early_S", "RADAR_Cross_S"],
                    }
                )
            else:
                if not math.isnan(tp1S):
                    exits.append(
                        {
                            "id": "S-TP1",
                            "mode": "reduce",
                            "side": "buy",
                            "type": "limit",
                            "price": tp1S,
                            "qty_pct": tp1_pct,
                            "reason": "TP1_S",
                        }
                    )
                if not math.isnan(dynStopS):
                    exits.append(
                        {
                            "id": "S-TRAIL",
                            "mode": "reduce_or_close",
                            "side": "buy",
                            "type": "stop",
                            "price": dynStopS,
                            "qty_pct": trail_pct,
                            "reason": "TRAIL_S",
                        }
                    )

        return {
            "entries": entries,
            "exits": exits,
            "modifications": modifications,
        }
