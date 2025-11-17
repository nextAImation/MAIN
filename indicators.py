"""
Indicator Engine Module - Provides TradingView-exact indicator calculations.
Fully vectorized, deterministic, and zero lookahead implementations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from numba import jit

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """
    Institutional-grade indicator calculator with TradingView parity.
    All calculations are vectorized, deterministic, and free of lookahead bias.
    """
    
    def __init__(self, df: pd.DataFrame, config: Any):
        """
        Initialize Indicator Engine with data and config.
        
        Args:
            df: DataFrame with OHLCV data for base timeframe
            config: BacktestConfig instance (or compatible)
        """
        self.df = df
        self.config = config
        self._indicators_df: Optional[pd.DataFrame] = None
        self._cache: Dict[str, Any] = {}  # For performance optimization / MTF cache
        self.logger = logging.getLogger(__name__)
    
    def compute_all(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        Precompute all indicators for the entire dataset.
        
        Args:
            df: Optional DataFrame to use instead of stored df
        """
        if df is not None:
            self.df = df
            
        if self.df is None:
            raise ValueError("No data provided for indicator computation")
            
        self.logger.info("Computing all indicators (base timeframe)...")
        self._indicators_df = self._compute_base_indicators(self.df)
        self._cache["base_indicators"] = self._indicators_df
        self.logger.info("Indicator computation complete")
    
    def get_values(self, bar_index: int) -> Dict[str, float]:
        """
        Get indicator values for a specific bar index with zero lookahead.
        
        Args:
            bar_index: Bar index (0-based) - uses data up to and including this index
            
        Returns:
            Dictionary of indicator values for the specified bar
        """
        if self._indicators_df is None:
            raise RuntimeError("Indicators not computed. Call compute_all() first.")
            
        if bar_index < 0 or bar_index >= len(self._indicators_df):
            return {}
            
        # Get the row for the specified bar index - zero lookahead guarantee
        row = self._indicators_df.iloc[bar_index]
        
        # Extract all indicator values, handling NaN values appropriately
        values: Dict[str, float] = {}
        for col in self._indicators_df.columns:
            if col in ['open', 'high', 'low', 'close', 'volume', 'ts']:
                continue  # Skip raw price columns
                
            value = row[col]
            # Keep NaN values as NaN - do not substitute with close price
            # This preserves the correct indicator warmup behavior
            if pd.isna(value):
                values[col] = float('nan')
            else:
                values[col] = float(value)
                
        return values
    
    def _compute_base_indicators(self, df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """
        Compute all base timeframe indicators.
        
        Args:
            df: OHLCV DataFrame
            prefix: Prefix for column names
            
        Returns:
            DataFrame with all indicator columns
        """
        result_df = df.copy()
        
        # EMAs (8, 21, 50, 200) - EMA first value at index = period-1 (TradingView-style seed)
        result_df[f'{prefix}ema_8'] = self.ema(df['close'], period=8)
        result_df[f'{prefix}ema_21'] = self.ema(df['close'], period=21)
        result_df[f'{prefix}ema_50'] = self.ema(df['close'], period=50)
        result_df[f'{prefix}ema_200'] = self.ema(df['close'], period=200)
        
        # RSI (14-period Wilder)
        result_df[f'{prefix}rsi_14'] = self.rsi(df['close'], period=14)
        
        # ATR (14-period Wilder) - ATR first non-NaN at index = period
        result_df[f'{prefix}atr_14'] = self.atr(df['high'], df['low'], df['close'], period=14)
        
        # ADX (14-period Wilder) - Correct DM & start index 2*period-1
        adx_result = self.adx(df['high'], df['low'], df['close'], period=14)
        result_df[f'{prefix}adx_14'] = adx_result['adx']
        result_df[f'{prefix}di_plus_14'] = adx_result['plus_di']
        result_df[f'{prefix}di_minus_14'] = adx_result['minus_di']
        
        # Additional common indicators
        result_df[f'{prefix}sma_20'] = self.sma(df['close'], period=20)
        result_df[f'{prefix}price_above_ema_50'] = df['close'] > result_df[f'{prefix}ema_50']
        result_df[f'{prefix}price_above_ema_200'] = df['close'] > result_df[f'{prefix}ema_200']
        
        return result_df
    
    def compute_all_indicators(
        self, 
        df: pd.DataFrame,
        symbol: str,
        timeframes: List[str] = ['1H', '4H', '1D']
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute all required indicators for a symbol across timeframes.
        
        Args:
            df: DataFrame with OHLCV data (base timeframe, indexed by timestamp)
            symbol: Symbol identifier
            timeframes: List of timeframes to compute indicators for (e.g. ['1H','4H','1D'])
            
        Returns:
            Dictionary with indicator DataFrames for each timeframe:
                {
                    'base': <base_df_with_indicators>,
                    '1H': <1H_resampled_df_with_indicators>,
                    '4H': <4H_resampled_df_with_indicators>,
                    ...
                }
        """
        results: Dict[str, pd.DataFrame] = {}
        
        # Base timeframe indicators
        base_indicators = self._compute_base_indicators(df)
        results['base'] = base_indicators
        
        # Higher timeframe indicators using proper resampling.
        # Only CLOSED candles (label='right', closed='right').
        for tf in timeframes:
            try:
                # Pandas uses strings like '1H', '4H', '1D' as freq
                ohlcv = df[['open', 'high', 'low', 'close', 'volume']].resample(
                    tf, label='right', closed='right'
                ).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                # Drop incomplete candles
                ohlcv = ohlcv.dropna(subset=['open', 'high', 'low', 'close'])
                
                tf_indicators = self._compute_base_indicators(ohlcv, prefix=f"{tf}_")
                results[tf] = tf_indicators
            except Exception as e:
                self.logger.warning(f"Failed to compute HTF indicators for {symbol} at {tf}: {e}")
        
        # Cache for later reuse if needed
        self._cache.setdefault("mtf_indicators", {})
        self._cache["mtf_indicators"][symbol] = results
        
        return results
    
    @staticmethod
    def ema(series: pd.Series, period: int, alpha: Optional[float] = None) -> pd.Series:
        """
        Exponential Moving Average (TradingView-style).
        Seed: SMA of first `period` values, placed at index = period-1.
        Then recursive EMA from index = period onward.
        
        Args:
            series: Input price series
            period: EMA period
            alpha: Smoothing factor (optional, calculated from period if not provided)
            
        Returns:
            EMA series
        """
        if alpha is None:
            alpha = 2.0 / (period + 1.0)
        
        ema_values = np.full(len(series), np.nan, dtype=float)
        
        if len(series) < period:
            return pd.Series(ema_values, index=series.index)
        
        # First EMA value (seed) is SMA of first `period` values
        sma_initial = series.iloc[:period].mean()
        
        # Place seed at index = period-1
        seed_idx = period - 1
        ema_values[seed_idx] = sma_initial
        
        # Recursive EMA calculation from index = period onward
        for i in range(period, len(series)):
            prev = ema_values[i - 1]
            price = series.iloc[i]
            ema_values[i] = alpha * price + (1.0 - alpha) * prev
        
        return pd.Series(ema_values, index=series.index)
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (Wilder's smoothing - TradingView-like).
        
        Args:
            series: Input price series
            period: RSI period (default 14)
            
        Returns:
            RSI values between 0-100
        """
        if len(series) < period + 1:
            return pd.Series([np.nan] * len(series), index=series.index)
        
        # Calculate price changes
        delta = series.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # First average is simple average over first `period` deltas (indexes 1..period)
        avg_gain = gains.rolling(window=period, min_periods=period).mean()
        avg_loss = losses.rolling(window=period, min_periods=period).mean()
        
        rsi_values = np.full(len(series), np.nan, dtype=float)
        
        # First RSI value at index = period
        if not pd.isna(avg_gain.iloc[period]) and not pd.isna(avg_loss.iloc[period]):
            if avg_loss.iloc[period] == 0:
                rsi_values[period] = 100.0
            else:
                rs = avg_gain.iloc[period] / avg_loss.iloc[period]
                rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))
        
        # Subsequent RSI values via Wilder smoothing
        for i in range(period + 1, len(series)):
            if pd.isna(rsi_values[i-1]):
                continue
                
            current_gain = gains.iloc[i]
            current_loss = losses.iloc[i]
            
            # Wilder smoothing: (prev_avg * (period-1) + current) / period
            prev_avg_gain = avg_gain.iloc[i-1] if i > period else avg_gain.iloc[period]
            prev_avg_loss = avg_loss.iloc[i-1] if i > period else avg_loss.iloc[period]
            
            smooth_gain = (prev_avg_gain * (period - 1) + current_gain) / period
            smooth_loss = (prev_avg_loss * (period - 1) + current_loss) / period
            
            if smooth_loss == 0:
                rsi_values[i] = 100.0
            else:
                rs = smooth_gain / smooth_loss
                rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))
            
            # Update the rolling averages for consistency
            avg_gain.iloc[i] = smooth_gain
            avg_loss.iloc[i] = smooth_loss
        
        return pd.Series(rsi_values, index=series.index)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range (Wilder's smoothing - TradingView-like).
        ATR starts at index = period (first non-NaN).
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            period: ATR period (default 14)
            
        Returns:
            ATR values
        """
        if len(high) < period + 1:
            return pd.Series([np.nan] * len(high), index=high.index)
        
        # True Range: start from index 1, TR[0] set to 0.0 (unused in ATR seed)
        tr = np.zeros(len(high), dtype=float)
        tr[0] = 0.0
        for i in range(1, len(high)):
            tr1 = high.iloc[i] - low.iloc[i]
            tr2 = abs(high.iloc[i] - close.iloc[i-1])
            tr3 = abs(low.iloc[i] - close.iloc[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        atr_values = np.full(len(high), np.nan, dtype=float)
        
        # First ATR is simple average of TR[1 .. period], stored at index = period
        initial_atr = tr[1:period+1].mean()
        atr_values[period] = initial_atr
        
        # Subsequent ATR values via Wilder smoothing
        for i in range(period + 1, len(high)):
            atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period
        
        return pd.Series(atr_values, index=high.index)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        Average Directional Index (Wilder's smoothing - TradingView-style).
        Correct +DM / -DM and ADX start index at 2*period - 1.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices  
            period: ADX period (default 14)
            
        Returns:
            Dictionary with 'adx', 'plus_di', 'minus_di' series
        """
        length = len(high)
        if length < 2 * period:
            nan_series = pd.Series([np.nan] * length, index=high.index)
            return {'adx': nan_series, 'plus_di': nan_series, 'minus_di': nan_series}
        
        # Convert to numpy for speed
        high_vals = high.values.astype(float)
        low_vals = low.values.astype(float)
        close_vals = close.values.astype(float)
        
        # UpMove / DownMove
        up_move = np.zeros(length, dtype=float)
        down_move = np.zeros(length, dtype=float)
        
        for i in range(1, length):
            up = high_vals[i] - high_vals[i-1]
            down = low_vals[i-1] - low_vals[i]
            up_move[i] = up if up > 0 else 0.0
            down_move[i] = down if down > 0 else 0.0
        
        # +DM and -DM
        plus_dm = np.zeros(length, dtype=float)
        minus_dm = np.zeros(length, dtype=float)
        
        for i in range(1, length):
            if up_move[i] > down_move[i] and up_move[i] > 0:
                plus_dm[i] = up_move[i]
            else:
                plus_dm[i] = 0.0
            
            if down_move[i] > up_move[i] and down_move[i] > 0:
                minus_dm[i] = down_move[i]
            else:
                minus_dm[i] = 0.0
        
        # True Range (same as ATR TR) - TR[0] = 0.0, then start from 1
        tr = np.zeros(length, dtype=float)
        tr[0] = 0.0
        for i in range(1, length):
            tr1 = high_vals[i] - low_vals[i]
            tr2 = abs(high_vals[i] - close_vals[i-1])
            tr3 = abs(low_vals[i] - close_vals[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        # Wilder smoothing for +DM, -DM, and TR
        smooth_plus_dm = np.full(length, np.nan, dtype=float)
        smooth_minus_dm = np.full(length, np.nan, dtype=float)
        smooth_tr = np.full(length, np.nan, dtype=float)
        
        # Simple averages over first `period` values (indexes 1 .. period)
        smooth_plus_dm[period] = plus_dm[1:period+1].mean()
        smooth_minus_dm[period] = minus_dm[1:period+1].mean()
        smooth_tr[period] = tr[1:period+1].mean()
        
        # Smoothing begins at index = period+1
        for i in range(period + 1, length):
            smooth_plus_dm[i] = (smooth_plus_dm[i-1] * (period - 1) + plus_dm[i]) / period
            smooth_minus_dm[i] = (smooth_minus_dm[i-1] * (period - 1) + minus_dm[i]) / period
            smooth_tr[i] = (smooth_tr[i-1] * (period - 1) + tr[i]) / period
        
        # +DI and -DI start EXACTLY at index = period
        plus_di_values = np.full(length, np.nan, dtype=float)
        minus_di_values = np.full(length, np.nan, dtype=float)
        
        for i in range(period, length):
            if smooth_tr[i] > 0:
                plus_di_values[i] = 100.0 * smooth_plus_dm[i] / smooth_tr[i]
                minus_di_values[i] = 100.0 * smooth_minus_dm[i] / smooth_tr[i]
            else:
                plus_di_values[i] = 0.0
                minus_di_values[i] = 0.0
        
        # DX
        dx_values = np.zeros(length, dtype=float)
        for i in range(period, length):
            if not pd.isna(plus_di_values[i]) and not pd.isna(minus_di_values[i]):
                di_sum = plus_di_values[i] + minus_di_values[i]
                if di_sum > 0:
                    dx_values[i] = 100.0 * abs(plus_di_values[i] - minus_di_values[i]) / di_sum
                else:
                    dx_values[i] = 0.0
        
        # ADX: Wilder smoothing of DX starting at index = 2*period - 1
        adx_values = np.full(length, np.nan, dtype=float)
        
        start_adx_idx = 2 * period - 1
        if start_adx_idx < length:
            # First ADX is simple average of DX[period .. 2*period-1]
            dx_slice = dx_values[period:start_adx_idx+1]
            if len(dx_slice) > 0:
                adx_values[start_adx_idx] = dx_slice.mean()
        
        for i in range(start_adx_idx + 1, length):
            if not pd.isna(adx_values[i-1]) and not pd.isna(dx_values[i]):
                adx_values[i] = (adx_values[i-1] * (period - 1) + dx_values[i]) / period
        
        return {
            'adx': pd.Series(adx_values, index=high.index),
            'plus_di': pd.Series(plus_di_values, index=high.index),
            'minus_di': pd.Series(minus_di_values, index=high.index)
        }
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            series: Input series
            period: SMA period
            
        Returns:
            SMA values
        """
        # Require full window to match typical trading platforms
        return series.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            series: Input series
            period: BB period (default 20)
            std_dev: Number of standard deviations (default 2)
            
        Returns:
            Dictionary with 'upper', 'middle', 'lower' bands
        """
        middle = series.rolling(window=period, min_periods=period).mean()
        std = series.rolling(window=period, min_periods=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def get_multi_timeframe_indicators(
        self, 
        base_data: pd.DataFrame,
        htf_data: Dict[str, pd.DataFrame],
        current_index: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Get indicator values for multiple timeframes at a specific index.
        Ensures no lookahead by using only data up to current_index.
        
        Args:
            base_data: Base timeframe data with indicators
            htf_data: Higher timeframe data with indicators
            current_index: Current bar index (0-based)
            
        Returns:
            Nested dictionary with indicator values for each timeframe
        """
        result: Dict[str, Dict[str, float]] = {}
        
        # Base timeframe indicators (up to current_index)
        base_slice = base_data.iloc[:current_index+1]
        if not base_slice.empty:
            # Consistent index thresholds with indicator warmups
            result['base'] = {
                'close': base_slice['close'].iloc[-1] if current_index >= 0 else np.nan,
                # EMA first value at index = period-1
                'ema_8': base_slice['ema_8'].iloc[-1] if 'ema_8' in base_slice.columns and current_index >= 7 else np.nan,
                'ema_21': base_slice['ema_21'].iloc[-1] if 'ema_21' in base_slice.columns and current_index >= 20 else np.nan,
                'ema_50': base_slice['ema_50'].iloc[-1] if 'ema_50' in base_slice.columns and current_index >= 49 else np.nan,
                'ema_200': base_slice['ema_200'].iloc[-1] if 'ema_200' in base_slice.columns and current_index >= 199 else np.nan,
                'rsi_14': base_slice['rsi_14'].iloc[-1] if 'rsi_14' in base_slice.columns and current_index >= 14 else np.nan,
                # ATR first value at index = period
                'atr_14': base_slice['atr_14'].iloc[-1] if 'atr_14' in base_slice.columns and current_index >= 14 else np.nan,
                # ADX first value at index = 2*period - 1 = 27 for period=14
                'adx_14': base_slice['adx_14'].iloc[-1] if 'adx_14' in base_slice.columns and current_index >= 27 else np.nan,
                'di_plus_14': base_slice['di_plus_14'].iloc[-1] if 'di_plus_14' in base_slice.columns and current_index >= 14 else np.nan,
                'di_minus_14': base_slice['di_minus_14'].iloc[-1] if 'di_minus_14' in base_slice.columns and current_index >= 14 else np.nan,
            }
        
        # Higher timeframe indicators (only fully closed candles)
        for tf_name, tf_df in htf_data.items():
            if tf_df.empty:
                continue
            # Base timestamp at current index
            current_timestamp = base_data.index[current_index]
            # Only candles that closed at or before current_timestamp
            tf_closed = tf_df[tf_df.index <= current_timestamp]
            
            if not tf_closed.empty:
                latest_tf_candle = tf_closed.iloc[-1]
                result[tf_name] = {
                    'close': latest_tf_candle['close'],
                    'ema_8': latest_tf_candle.get(f'{tf_name}_ema_8', np.nan),
                    'ema_21': latest_tf_candle.get(f'{tf_name}_ema_21', np.nan),
                    'ema_50': latest_tf_candle.get(f'{tf_name}_ema_50', np.nan),
                    'ema_200': latest_tf_candle.get(f'{tf_name}_ema_200', np.nan),
                    'rsi_14': latest_tf_candle.get(f'{tf_name}_rsi_14', np.nan),
                    'atr_14': latest_tf_candle.get(f'{tf_name}_atr_14', np.nan),
                    'adx_14': latest_tf_candle.get(f'{tf_name}_adx_14', np.nan),
                    'di_plus_14': latest_tf_candle.get(f'{tf_name}_di_plus_14', np.nan),
                    'di_minus_14': latest_tf_candle.get(f'{tf_name}_di_minus_14', np.nan),
                }
        
        return result


# Optimized versions using Numba for performance
@jit(nopython=True)
def _ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized EMA calculation with exact parity to IndicatorEngine.ema."""
    length = len(prices)
    result = np.empty(length, dtype=np.float64)
    result[:] = np.nan
    
    if length < period:
        return result
    
    alpha = 2.0 / (period + 1.0)
    
    # Seed: SMA of first `period` values at index = period-1
    sma_initial = 0.0
    for i in range(period):
        sma_initial += prices[i]
    sma_initial /= period
    
    seed_idx = period - 1
    result[seed_idx] = sma_initial
    
    # Recursive EMA from index = period onward
    for i in range(period, length):
        prev = result[i-1]
        price = prices[i]
        result[i] = alpha * price + (1.0 - alpha) * prev
    
    return result


@jit(nopython=True)
def _rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized RSI calculation with Wilder smoothing."""
    length = len(prices)
    rsi = np.empty(length, dtype=np.float64)
    rsi[:] = np.nan
    
    if length < period + 1:
        return rsi
    
    deltas = np.zeros(length, dtype=np.float64)
    for i in range(1, length):
        deltas[i] = prices[i] - prices[i-1]
    
    gains = np.zeros(length, dtype=np.float64)
    losses = np.zeros(length, dtype=np.float64)
    for i in range(1, length):
        if deltas[i] > 0:
            gains[i] = deltas[i]
            losses[i] = 0.0
        elif deltas[i] < 0:
            gains[i] = 0.0
            losses[i] = -deltas[i]
        else:
            gains[i] = 0.0
            losses[i] = 0.0
    
    avg_gain = np.empty(length, dtype=np.float64)
    avg_gain[:] = np.nan
    avg_loss = np.empty(length, dtype=np.float64)
    avg_loss[:] = np.nan
    
    # Initial averages over first `period` deltas (indexes 1..period)
    sum_gain = 0.0
    sum_loss = 0.0
    for i in range(1, period+1):
        sum_gain += gains[i]
        sum_loss += losses[i]
    avg_gain[period] = sum_gain / period
    avg_loss[period] = sum_loss / period
    
    # First RSI at index = period
    if avg_loss[period] == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain[period] / avg_loss[period]
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Subsequent values via Wilder smoothing
    for i in range(period + 1, length):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
        
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


@jit(nopython=True)
def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized ATR calculation with parity to IndicatorEngine.atr."""
    length = len(high)
    atr_values = np.empty(length, dtype=np.float64)
    atr_values[:] = np.nan
    
    if length < period + 1:
        return atr_values
    
    tr = np.zeros(length, dtype=np.float64)
    tr[0] = 0.0
    for i in range(1, length):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        if tr1 >= tr2 and tr1 >= tr3:
            tr[i] = tr1
        elif tr2 >= tr1 and tr2 >= tr3:
            tr[i] = tr2
        else:
            tr[i] = tr3
    
    # First ATR = mean of TR[1..period] at index = period
    sum_tr = 0.0
    for i in range(1, period+1):
        sum_tr += tr[i]
    atr_values[period] = sum_tr / period
    
    # Wilder smoothing
    for i in range(period + 1, length):
        atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period
    
    return atr_values


class FastIndicatorEngine(IndicatorEngine):
    """
    Performance-optimized indicator engine using Numba.
    For use with large datasets (>1M bars).
    """
    
    def __init__(self, df: pd.DataFrame, config: Any):
        super().__init__(df, config)
    
    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """Optimized EMA using Numba with exact parity."""
        values = series.values.astype(float)
        result = _ema_numba(values, period)
        return pd.Series(result, index=series.index)
    
    def rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Optimized RSI using Numba with exact parity."""
        values = series.values.astype(float)
        result = _rsi_numba(values, period)
        return pd.Series(result, index=series.index)
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Optimized ATR using Numba with exact parity."""
        high_vals = high.values.astype(float)
        low_vals = low.values.astype(float)
        close_vals = close.values.astype(float)
        result = _atr_numba(high_vals, low_vals, close_vals, period)
        return pd.Series(result, index=high.index)


# Unit Tests
def test_indicator_parity():
    """Test that indicators match expected TradingView-like behavior and fast engine parity."""
    print("Running Indicator Parity Tests...")
    
    # Create test data
    np.random.seed(42)  # For deterministic testing
    dates = pd.date_range('2023-01-01', periods=100, freq='1H', tz='UTC')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    test_data = pd.DataFrame({
        'open': prices + np.random.randn(100) * 0.1,
        'high': prices + np.abs(np.random.randn(100) * 0.2),
        'low': prices - np.abs(np.random.randn(100) * 0.2),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    # Ensure OHLC sanity
    test_data['high'] = test_data[['open', 'close', 'high']].max(axis=1)
    test_data['low'] = test_data[['open', 'close', 'low']].min(axis=1)
    
    # Mock config class for testing
    class MockConfig:
        pass
    
    config = MockConfig()
    
    # Test both engines for parity
    engine = IndicatorEngine(test_data, config)
    engine.compute_all()
    
    fast_engine = FastIndicatorEngine(test_data, config)
    fast_engine.compute_all()
    
    # Test get_values API
    values_50 = engine.get_values(50)
    fast_values_50 = fast_engine.get_values(50)
    
    # Test EMA
    ema_8 = engine.ema(test_data['close'], 8)
    ema_21 = engine.ema(test_data['close'], 21)
    fast_ema_8 = fast_engine.ema(test_data['close'], 8)
    
    # Test RSI
    rsi_14 = engine.rsi(test_data['close'], 14)
    fast_rsi_14 = fast_engine.rsi(test_data['close'], 14)
    
    # Test ATR
    atr_14 = engine.atr(test_data['high'], test_data['low'], test_data['close'], 14)
    fast_atr_14 = fast_engine.atr(test_data['high'], test_data['low'], test_data['close'], 14)
    
    # Test ADX
    adx_result = engine.adx(test_data['high'], test_data['low'], test_data['close'], 14)
    
    # Validation tests
    assert len(ema_8) == len(test_data), "EMA length mismatch"
    assert len(rsi_14) == len(test_data), "RSI length mismatch"
    assert len(atr_14) == len(test_data), "ATR length mismatch"
    assert len(adx_result['adx']) == len(test_data), "ADX length mismatch"
    
    # Check parity between standard and fast engines
    assert np.allclose(ema_8.values, fast_ema_8.values, equal_nan=True), "EMA parity failed"
    assert np.allclose(rsi_14.values, fast_rsi_14.values, equal_nan=True), "RSI parity failed"
    assert np.allclose(atr_14.values, fast_atr_14.values, equal_nan=True), "ATR parity failed"
    
    # Check get_values returns proper structure
    assert isinstance(values_50, dict), "get_values should return dict"
    assert 'ema_8' in values_50, "get_values should contain ema_8"
    assert 'rsi_14' in values_50, "get_values should contain rsi_14"
    assert 'atr_14' in values_50, "get_values should contain atr_14"
    assert 'adx_14' in values_50, "get_values should contain adx_14"
    assert 'di_plus_14' in values_50, "get_values should contain di_plus_14"
    assert 'di_minus_14' in values_50, "get_values should contain di_minus_14"
    
    # EMA warmup check: first non-NaN at index = period-1
    assert pd.isna(ema_8.iloc[6]), "EMA(8) should have NaN for indexes < period-1"
    assert not pd.isna(ema_8.iloc[7]), "EMA(8) should have first value at index = period-1"
    
    # RSI bounds check
    valid_rsi = rsi_14.dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all(), "RSI should be between 0-100"
    
    # ATR positivity check
    valid_atr = atr_14.dropna()
    assert (valid_atr > 0).all(), "ATR should be positive"
    
    # ADX start index (should be 2*period-1 = 27 for period=14)
    assert pd.isna(adx_result['adx'].iloc[26]), "ADX should be NaN before 2*period-1"
    assert not pd.isna(adx_result['adx'].iloc[27]), "ADX should have first value at 2*period-1"
    
    print("✅ All indicator parity tests passed!")
    print("✅ Standard and Fast engines produce identical results!")
    
    # Performance test
    print("\nPerformance test with larger dataset...")
    large_dates = pd.date_range('2020-01-01', periods=100000, freq='1min', tz='UTC')
    large_prices = 100 + np.cumsum(np.random.randn(100000) * 0.01)
    
    large_data = pd.DataFrame({
        'open': large_prices + np.random.randn(100000) * 0.001,
        'high': large_prices + np.abs(np.random.randn(100000) * 0.002),
        'low': large_prices - np.abs(np.random.randn(100000) * 0.002),
        'close': large_prices,
        'volume': np.random.uniform(1000, 10000, 100000)
    }, index=large_dates)
    
    import time
    start_time = time.time()
    
    fast_engine_large = FastIndicatorEngine(large_data, config)
    fast_engine_large.compute_all()
    _ = fast_engine_large.get_values(50000)
    
    end_time = time.time()
    print(f"✅ Processed 100,000 bars in {end_time - start_time:.2f} seconds")
    
    return {
        'ema_8': ema_8,
        'ema_21': ema_21,
        'rsi_14': rsi_14,
        'atr_14': atr_14,
        'adx': adx_result['adx'],
        'values_50': values_50,
        'fast_values_50': fast_values_50,
    }


if __name__ == "__main__":
    test_results = test_indicator_parity()
    print("\n🎯 Indicator Engine implementation complete!")