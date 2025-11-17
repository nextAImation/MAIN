"""
DataLoader module - Responsible for loading, normalizing, and resampling candle data.
Produces DataBundle structure for consumption by other modules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import pyarrow.parquet as pq

from config import DataSchema, BacktestConfig, Timeframe, SUPPORTED_TIMEFRAMES, ValidationMode
from validator import DataValidator


logger = logging.getLogger(__name__)


class DataBundle:
    """
    Container for all data required by the backtesting engine.
    Ensures structured access to base and higher timeframe data.
    """
    
    def __init__(self):
        self.base_tf_candles: Dict[str, pd.DataFrame] = {}
        self.htf_candles: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.metadata: Dict[str, Any] = {
            "tick_size": {},
            "contract_type": {},
            "fee_rates": {},
            "timezone": "UTC",
            "symbols": [],
        }
    
    def add_symbol_data(
        self,
        symbol: str,
        base_candles: pd.DataFrame,
        htf_candles: Dict[str, pd.DataFrame],
        tick_size: float,
        contract_type: str,
        fee_rate: float
    ):
        """Add complete data for a symbol"""
        self.base_tf_candles[symbol] = base_candles
        self.htf_candles[symbol] = htf_candles
        self.metadata["tick_size"][symbol] = tick_size
        self.metadata["contract_type"][symbol] = contract_type
        self.metadata["fee_rates"][symbol] = fee_rate
        self.metadata["symbols"].append(symbol)
    
    def validate(self) -> bool:
        """Basic validation of data bundle integrity"""
        if not self.base_tf_candles:
            raise ValueError("No base timeframe data loaded")
        
        for symbol in self.metadata["symbols"]:
            if symbol not in self.base_tf_candles:
                raise ValueError(f"Missing base data for symbol: {symbol}")
            if symbol not in self.htf_candles:
                raise ValueError(f"Missing HTF data for symbol: {symbol}")
        
        return True


class DataLoader:
    """
    Main data loading and preprocessing engine.
    Handles CSV/Parquet loading, schema normalization, and resampling.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.schema = self.config.data_schema
        self.validator = DataValidator(config)
        
    def load_from_csv(
        self,
        file_path: str,
        symbol: str,
        tick_size: Optional[float] = None,
        contract_type: Optional[str] = None,
        fee_rate: Optional[float] = None
    ) -> DataBundle:
        """Load data from CSV file"""
        logger.info(f"Loading data from CSV: {file_path}")
        
        df = pd.read_csv(file_path)
        return self._process_dataframe(
            df, symbol, tick_size, contract_type, fee_rate
        )
    
    def load_from_parquet(
        self,
        file_path: str,
        symbol: str,
        tick_size: Optional[float] = None,
        contract_type: Optional[str] = None,
        fee_rate: Optional[float] = None
    ) -> DataBundle:
        """Load data from Parquet file"""
        logger.info(f"Loading data from Parquet: {file_path}")
        
        df = pd.read_parquet(file_path)
        return self._process_dataframe(
            df, symbol, tick_size, contract_type, fee_rate
        )
    
    def _process_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        tick_size: Optional[float],
        contract_type: Optional[str],
        fee_rate: Optional[float]
    ) -> DataBundle:
        """Process DataFrame into DataBundle"""
        # Normalize schema
        df = self._normalize_schema(df)
        
        # Ensure UTC timestamps and set as index
        df = self._normalize_timestamps(df)
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Set expected time delta for validation
        self.config.expected_time_delta = self.config.base_timeframe.to_seconds() * 1000
        self.validator.config.expected_time_delta = self.config.expected_time_delta
        
        # Resample to higher timeframes
        htf_candles = self._resample_htf_candles(df)
        
        # Create DataBundle
        bundle = DataBundle()
        bundle.add_symbol_data(
            symbol=symbol,
            base_candles=df,
            htf_candles=htf_candles,
            tick_size=tick_size or self.config.default_tick_size,
            contract_type=contract_type or self.config.default_contract_type.value,
            fee_rate=fee_rate or self.config.default_fee_rate
        )
        
        # Validate data if not in NONE mode
        if self.config.validation_mode != ValidationMode.NONE:
            logger.info("Validating loaded data...")
            self.validator.validate_bundle(bundle)
        
        return bundle
    
    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard schema"""
        column_mapping = {}
        
        # Map common column names to standard schema
        common_names = {
            'timestamp': self.schema.TIMESTAMP,
            'time': self.schema.TIMESTAMP,
            'date': self.schema.TIMESTAMP,
            'open': self.schema.OPEN,
            'high': self.schema.HIGH, 
            'low': self.schema.LOW,
            'close': self.schema.CLOSE,
            'volume': self.schema.VOLUME,
            'bid_open': self.schema.BID_OPEN,
            'bid_high': self.schema.BID_HIGH,
            'bid_low': self.schema.BID_LOW,
            'bid_close': self.schema.BID_CLOSE,
            'ask_open': self.schema.ASK_OPEN,
            'ask_high': self.schema.ASK_HIGH,
            'ask_low': self.schema.ASK_LOW,
            'ask_close': self.schema.ASK_CLOSE,
        }
        
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in common_names:
                column_mapping[col] = common_names[col_lower]
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = [self.schema.TIMESTAMP, self.schema.OPEN, self.schema.HIGH, 
                        self.schema.LOW, self.schema.CLOSE, self.schema.VOLUME]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamp column to UTC timezone-aware index"""
        ts_col = self.schema.TIMESTAMP
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col])
        
        # Ensure UTC timezone
        if df[ts_col].dt.tz is None:
            df[ts_col] = df[ts_col].dt.tz_localize('UTC')
        else:
            df[ts_col] = df[ts_col].dt.tz_convert('UTC')
        
        # Set as index
        df = df.set_index(ts_col)
        
        # Add numeric ts column in milliseconds
        df['ts'] = df.index.view('int64') // 1_000_000
        
        return df
    
    def _resample_htf_candles(self, base_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Resample base timeframe to higher timeframes"""
        htf_candles = {}
        
        for timeframe in self.config.htf_timeframes:
            # Skip base timeframe
            if timeframe.minutes <= self.config.base_timeframe.minutes:
                continue
            
            # Resample rule
            rule = f"{timeframe.minutes}T"  # minute-based resampling
            
            # OHLC resampling
            resampled = base_df.resample(rule).agg({
                self.schema.OPEN: 'first',
                self.schema.HIGH: 'max',
                self.schema.LOW: 'min',
                self.schema.CLOSE: 'last',
                self.schema.VOLUME: 'sum'
            }).dropna()
            
            # Handle bid/ask columns if present
            bid_ask_cols = [
                self.schema.BID_OPEN, self.schema.BID_HIGH, self.schema.BID_LOW, self.schema.BID_CLOSE,
                self.schema.ASK_OPEN, self.schema.ASK_HIGH, self.schema.ASK_LOW, self.schema.ASK_CLOSE
            ]
            
            for col in bid_ask_cols:
                if col in base_df.columns:
                    if 'bid' in col:
                        resampled[col] = base_df[col].resample(rule).agg('first')
                    elif 'ask' in col:
                        resampled[col] = base_df[col].resample(rule).agg('last')
            
            # Add ts column in milliseconds
            resampled['ts'] = resampled.index.view('int64') // 1_000_000
            
            htf_candles[timeframe.name] = resampled
        
        return htf_candles
    
    def merge_bundles(self, bundles: List[DataBundle]) -> DataBundle:
        """Merge multiple DataBundles into one"""
        if not bundles:
            raise ValueError("No bundles to merge")
        
        merged = DataBundle()
        
        for bundle in bundles:
            for symbol in bundle.metadata["symbols"]:
                merged.add_symbol_data(
                    symbol=symbol,
                    base_candles=bundle.base_tf_candles[symbol],
                    htf_candles=bundle.htf_candles[symbol],
                    tick_size=bundle.metadata["tick_size"][symbol],
                    contract_type=bundle.metadata["contract_type"][symbol],
                    fee_rate=bundle.metadata["fee_rates"][symbol]
                )
        
        return merged