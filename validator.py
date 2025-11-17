"""
Data validation module - Ensures data quality and integrity.
Detects and handles missing data, duplicates, and sanity violations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from enum import Enum

from config import DataSchema, ValidationMode, BacktestConfig, SUPPORTED_TIMEFRAMES


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class DataValidator:
    """
    Comprehensive data validator for financial time series.
    Implements strict and soft validation modes.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.schema = self.config.data_schema
        self.mode = self.config.validation_mode
    
    def validate_bundle(self, bundle) -> bool:
        """Validate entire DataBundle"""
        logger.info("Validating DataBundle...")
        
        try:
            bundle.validate()
            
            for symbol in bundle.metadata["symbols"]:
                self.validate_symbol_data(bundle, symbol)
            
            logger.info("DataBundle validation passed")
            return True
            
        except ValidationError as e:
            if self.mode == ValidationMode.STRICT:
                raise
            else:
                logger.warning(f"Soft mode: Allowing validation issue: {e}")
                return False
    
    def validate_symbol_data(self, bundle, symbol: str):
        """Validate data for a specific symbol"""
        base_df = bundle.base_tf_candles[symbol]
        htf_data = bundle.htf_candles[symbol]
        
        # Base timeframe validation
        self._validate_base_data(base_df, symbol)
        
        # Higher timeframe validation
        for tf_name, htf_df in htf_data.items():
            self._validate_htf_data(htf_df, symbol, tf_name)
        
        # Cross-timeframe validation
        self._validate_cross_timeframe(base_df, htf_data, symbol)
    
    def _validate_base_data(self, df: pd.DataFrame, symbol: str):
        """Validate base timeframe data"""
        logger.info(f"Validating base data for {symbol}")
        
        # 1. Check for missing candles
        self._check_missing_candles(df, symbol)
        
        # 2. Check for duplicates
        self._check_duplicate_timestamps(df, symbol)
        
        # 3. Check timestamp order
        self._check_timestamp_order(df, symbol)
        
        # 4. OHLC sanity checks
        self._check_ohlc_sanity(df, symbol)
        
        # 5. Bid/ask sanity checks (if present)
        self._check_bid_ask_sanity(df, symbol)
        
        # 6. Check for NaN values
        self._check_nan_values(df, symbol)
    
    def _validate_htf_data(self, df: pd.DataFrame, symbol: str, tf_name: str):
        """Validate higher timeframe data"""
        logger.info(f"Validating {tf_name} data for {symbol}")
        
        self._check_ohlc_sanity(df, f"{symbol}_{tf_name}")
        self._check_nan_values(df, f"{symbol}_{tf_name}")
    
    def _validate_cross_timeframe(self, base_df: pd.DataFrame, htf_data: Dict[str, pd.DataFrame], symbol: str):
        """Validate consistency between timeframes"""
        base_start, base_end = base_df.index[0], base_df.index[-1]
        
        for tf_name, htf_df in htf_data.items():
            # Check timestamp order
            if not htf_df.index.is_monotonic_increasing:
                self._handle_validation_issue(
                    f"HTF {tf_name} timestamps not in increasing order for {symbol}"
                )
            
            # Check duplicates
            duplicates = htf_df.index.duplicated()
            if duplicates.any():
                duplicate_times = htf_df.index[duplicates].unique()[:5]
                self._handle_validation_issue(
                    f"Duplicate HTF {tf_name} timestamps in {symbol}: {list(duplicate_times)}"
                )
            
            # Check missing candles in HTF sequence
            self._check_missing_candles(htf_df, f"{symbol}_{tf_name}")
            
            # Check HTF timestamps are within base data range with tolerance
            htf_start, htf_end = htf_df.index[0], htf_df.index[-1]
            
            # Allow small tolerance for timeframe alignment
            tolerance = pd.Timedelta(minutes=5)
            
            if htf_start < (base_start - tolerance) or htf_end > (base_end + tolerance):
                self._handle_validation_issue(
                    f"HTF {tf_name} range [{htf_start}, {htf_end}] outside base range [{base_start}, {base_end}]"
                )
            
            # Check HTF timestamp alignment
            tf_minutes = SUPPORTED_TIMEFRAMES[tf_name].minutes
            for ts in htf_df.index[:100]:  # Check first 100 timestamps
                if not self._is_aligned(ts, tf_minutes):
                    self._handle_validation_issue(f"HTF {tf_name} misaligned timestamp: {ts}")
                    break
    
    def _is_aligned(self, ts, minutes):
        """Check if timestamp is aligned to timeframe"""
        return (ts.minute % minutes == 0) and ts.second == 0 and ts.microsecond == 0
    
    def _check_missing_candles(self, df: pd.DataFrame, symbol: str):
        """Detect missing candles in time series"""
        if len(df) < 2:
            return
        
        time_diffs = df.index.to_series().diff().dropna()
        expected_delta = pd.Timedelta(seconds=self.config.expected_time_delta)
        
        # Allow small tolerance for floating point issues
        tolerance = pd.Timedelta('1ms')
        gaps = time_diffs[time_diffs > (expected_delta + tolerance)]
        
        if not gaps.empty:
            gap_info = []
            for i in range(len(gaps)):
                gap_start = gaps.index[i]
                gap_end = gap_start + gaps.iloc[i]
                gap_info.append(f"({gap_start} to {gap_end})")
                if len(gap_info) >= 5:  # Show first 5 gaps
                    break
            
            self._handle_validation_issue(
                f"Missing candles in {symbol} at: {gap_info}"
            )
    
    def _check_duplicate_timestamps(self, df: pd.DataFrame, symbol: str):
        """Detect duplicate timestamps"""
        duplicates = df.index.duplicated()
        if duplicates.any():
            duplicate_times = df.index[duplicates].unique()[:5]
            self._handle_validation_issue(
                f"Duplicate timestamps in {symbol}: {list(duplicate_times)}"
            )
    
    def _check_timestamp_order(self, df: pd.DataFrame, symbol: str):
        """Check if timestamps are in correct order"""
        if not df.index.is_monotonic_increasing:
            self._handle_validation_issue(
                f"Timestamps not in increasing order for {symbol}"
            )
    
    def _check_ohlc_sanity(self, df: pd.DataFrame, identifier: str):
        """Perform OHLC sanity checks"""
        # High >= max(Open, Close)
        high_violations = df[self.schema.HIGH] < df[[self.schema.OPEN, self.schema.CLOSE]].max(axis=1)
        if high_violations.any():
            self._handle_validation_issue(
                f"High < max(Open, Close) violations in {identifier}: {high_violations.sum()}"
            )
        
        # Low <= min(Open, Close)
        low_violations = df[self.schema.LOW] > df[[self.schema.OPEN, self.schema.CLOSE]].min(axis=1)
        if low_violations.any():
            self._handle_validation_issue(
                f"Low > min(Open, Close) violations in {identifier}: {low_violations.sum()}"
            )
        
        # High >= Low
        hl_violations = df[self.schema.HIGH] < df[self.schema.LOW]
        if hl_violations.any():
            self._handle_validation_issue(
                f"High < Low violations in {identifier}: {hl_violations.sum()}"
            )
        
        # Volume >= 0
        if self.schema.VOLUME in df.columns:
            volume_violations = df[self.schema.VOLUME] < 0
            if volume_violations.any():
                self._handle_validation_issue(
                    f"Negative volume in {identifier}: {volume_violations.sum()}"
                )
    
    def _check_bid_ask_sanity(self, df: pd.DataFrame, symbol: str):
        """Check bid/ask sanity if bid/ask columns are present"""
        bid_ask_pairs = [
            (self.schema.BID_OPEN, self.schema.ASK_OPEN),
            (self.schema.BID_HIGH, self.schema.ASK_HIGH),
            (self.schema.BID_LOW, self.schema.ASK_LOW),
            (self.schema.BID_CLOSE, self.schema.ASK_CLOSE),
        ]
        
        for bid_col, ask_col in bid_ask_pairs:
            if bid_col in df.columns and ask_col in df.columns:
                # Bid should be <= Ask
                violations = df[bid_col] > df[ask_col]
                if violations.any():
                    self._handle_validation_issue(
                        f"Bid > Ask violations in {symbol} ({bid_col}/{ask_col}): {violations.sum()}"
                    )
    
    def _check_nan_values(self, df: pd.DataFrame, identifier: str):
        """Check for NaN values in critical columns"""
        critical_cols = [self.schema.OPEN, self.schema.HIGH, self.schema.LOW, self.schema.CLOSE]
        
        for col in critical_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    self._handle_validation_issue(
                        f"NaN values in {identifier}.{col}: {nan_count}"
                    )
    
    def _handle_validation_issue(self, message: str):
        """Handle validation issues based on mode"""
        if self.mode == ValidationMode.STRICT:
            raise ValidationError(message)
        else:
            logger.warning(f"Validation issue (soft mode): {message}")
    
    def create_synthetic_candle(self, prev_candle: pd.Series, next_candle: pd.Series) -> pd.Series:
        """Create synthetic candle for missing data (soft mode only)"""
        synthetic = prev_candle.copy()
        
        # Simple interpolation for OHLC
        for col in [self.schema.OPEN, self.schema.HIGH, self.schema.LOW, self.schema.CLOSE]:
            if col in synthetic.index:
                synthetic[col] = (prev_candle[col] + next_candle[col]) / 2
        
        # Volume as average
        if self.schema.VOLUME in synthetic.index:
            synthetic[self.schema.VOLUME] = (prev_candle[self.schema.VOLUME] + next_candle[self.schema.VOLUME]) / 2
        
        return synthetic


# Unit Tests
def test_validator():
    """Unit tests for DataValidator"""
    import tempfile
    import os
    
    print("Running DataValidator tests...")
    
    # Test data
    test_data = pd.DataFrame({
        'ts': pd.date_range('2023-01-01', periods=100, freq='1min', tz='UTC'),
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    })
    
    # Create intentional violations
    test_data.loc[10, 'high'] = 50  # High < Low violation
    test_data.loc[20, 'low'] = 300  # Low > High violation
    
    config = BacktestConfig(validation_mode=ValidationMode.STRICT)
    validator = DataValidator(config)
    
    try:
        validator._check_ohlc_sanity(test_data, "test_data")
        print("❌ Should have detected OHLC violations")
    except ValidationError:
        print("✅ Correctly detected OHLC violations")
    
    # Test soft mode
    config_soft = BacktestConfig(validation_mode=ValidationMode.SOFT)
    validator_soft = DataValidator(config_soft)
    
    try:
        validator_soft._check_ohlc_sanity(test_data, "test_data")
        print("✅ Soft mode handled violations without exception")
    except ValidationError:
        print("❌ Soft mode should not raise exceptions")
    
    print("All validator tests completed!")


if __name__ == "__main__":
    test_validator()