@dataclass
class RadarV2Config:
    ma_type: str = "EMA"
    ma_fast_len: int = 20
    ma_slow_len: int = 50
    ma_reg_len: int = 200

    adx_len: int = 14
    adx_th_fixed: float = 22
    use_dyn_adx: bool = False
    adx_dyn_k: float = 0.9
    require_adx_up: bool = True

    use_rsi: bool = True
    rsi_len: int = 14
    rsi_th_long: float = 48

    use_vol: bool = True
    vol_len: int = 20
    vol_k_long: float = 0.7

    atr_len: int = 14
    atr_mult_sl: float = 2.0
    atr_mult_tp: float = 3.0
    risk_pct: float = 0.03

    use_trend_filter: bool = True
    daily_soft_k: float = 0.9
    use_regime_filter: bool = False

    use_cooldown: bool = True
    cooldown_bars: int = 6

    use_early_entry: bool = False
    early_rsi_long: float = 55
    early_rsi_short: float = 45
    early_adx: float = 18
    early_qty_frac: float = 0.25

    use_cross_entry: bool = True
    cross_need_adx: bool = True
    cross_need_daily: bool = True
    cross_need_regime: bool = False

    use_struct: bool = True
    swing_len: int = 5
    bos_buf_atr: float = 0.25
    use_choch_soft: bool = False
    strict_short: bool = True
    adx_soft_strict: bool = True

    tp2_trail_atr_long: float = 2.0
    tp2_trail_atr_short: float = 1.8
    be_buf: float = 0.2
    weak_confirm_bars: int = 9

    tp1_ratio: float = 1.8
    use_break_even: bool = True
    tp1_pct: int = 29

    use_daily_in_bypass: bool = True
    allow_daily_neutral: bool = True
    use_range_bypass: bool = True
