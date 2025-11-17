"""
Institutional-grade report builder with comprehensive analytics.
Produces CSV, JSON, and formatted reports for backtest results.
"""

import json
import csv
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from pathlib import Path


class ReportBuilder:
    """
    Professional report generation for backtest results.
    Supports multiple formats and comprehensive analytics.
    """
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_all_reports(self, output_dir: str = "reports") -> Dict[str, str]:
        """
        Generate all report formats.
        Returns dictionary of file paths.
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        report_files = {}
        
        # Generate CSV reports
        report_files['trades'] = self.generate_trades_csv(output_dir)
        report_files['equity'] = self.generate_equity_csv(output_dir)
        report_files['orders'] = self.generate_orders_csv(output_dir)
        
        # Generate JSON summary
        report_files['summary'] = self.generate_summary_json(output_dir)
        
        # Generate formatted text report
        report_files['text'] = self.generate_text_report(output_dir)
        
        return report_files
    
    def generate_trades_csv(self, output_dir: str) -> str:
        """Generate comprehensive trades CSV report"""
        filename = f"{output_dir}/trades_{self.timestamp}.csv"
        
        trades = self.results.get('trades', [])
        if not trades:
            # Create empty file with headers
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'trade_id', 'symbol', 'entry_timestamp', 'exit_timestamp',
                    'entry_bar', 'exit_bar', 'entry_price', 'exit_price',
                    'quantity', 'side', 'pnl', 'pnl_pct', 'commission',
                    'net_pnl', 'duration_bars', 'duration_seconds',
                    'entry_reason', 'exit_reason', 'tags', 'mae', 'mfe',
                    'avg_entry_price', 'avg_exit_price', 'max_drawdown_trade', 'runup'
                ])
            return filename
        
        # Write trades data
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
        
        return filename
    
    def generate_equity_csv(self, output_dir: str) -> str:
        """Generate equity curve CSV report"""
        filename = f"{output_dir}/equity_curve_{self.timestamp}.csv"
        
        equity_curve = self.results.get('equity_curve', [])
        if not equity_curve:
            # Create empty file with headers
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'bar_index', 'equity', 'unrealized_pnl',
                    'realized_pnl', 'drawdown', 'drawdown_pct'
                ])
            return filename
        
        # Write equity curve data
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=equity_curve[0].keys())
            writer.writeheader()
            writer.writerows(equity_curve)
        
        return filename
    
    def generate_orders_csv(self, output_dir: str) -> str:
        """Generate orders history CSV report"""
        filename = f"{output_dir}/orders_{self.timestamp}.csv"
        
        orders = self.results.get('orders', [])
        if not orders:
            # Create empty file with headers
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'order_id', 'symbol', 'order_type', 'side', 'quantity',
                    'filled_quantity', 'limit_price', 'stop_price', 'reduce_only',
                    'post_only', 'created_timestamp', 'activation_time',
                    'last_updated_timestamp', 'average_fill_price', 'status'
                ])
            return filename
        
        # Write orders data
        with open(filename, 'w', newline='') as f:
            # Flatten orders data for CSV
            flat_orders = []
            for order in orders:
                flat_order = {
                    'order_id': order.get('order_id', ''),
                    'symbol': order.get('symbol', ''),
                    'order_type': order.get('order_type', ''),
                    'side': order.get('side', ''),
                    'quantity': order.get('quantity', 0),
                    'filled_quantity': order.get('filled_quantity', 0),
                    'limit_price': order.get('limit_price'),
                    'stop_price': order.get('stop_price'),
                    'reduce_only': order.get('reduce_only', False),
                    'post_only': order.get('post_only', False),
                    'created_timestamp': order.get('created_timestamp', 0),
                    'activation_time': order.get('activation_time', 0),
                    'last_updated_timestamp': order.get('last_updated_timestamp', 0),
                    'average_fill_price': order.get('average_fill_price', 0),
                    'status': order.get('status', '')
                }
                flat_orders.append(flat_order)
            
            if flat_orders:
                writer = csv.DictWriter(f, fieldnames=flat_orders[0].keys())
                writer.writeheader()
                writer.writerows(flat_orders)
        
        return filename
    
    def generate_summary_json(self, output_dir: str) -> str:
        """Generate comprehensive JSON summary report"""
        filename = f"{output_dir}/summary_{self.timestamp}.json"
        
        summary_data = {
            'backtest_info': {
                'timestamp': self.results.get('completed_timestamp'),
                'symbol': self.results['config'].get('symbol'),
                'timeframe': self.results['config'].get('base_timeframe'),
                'initial_equity': self.results['config'].get('initial_equity'),
                'final_equity': self.results.get('final_equity'),
                'total_bars': self.results.get('total_bars')
            },
            'performance_metrics': self.results.get('metrics', {}),
            'risk_metrics': self.results.get('risk', {}),
            'position_summary': self.results.get('position', {}),
            'trade_analytics': self._calculate_trade_analytics()
        }
        
        with open(filename, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        return filename
    
    def generate_text_report(self, output_dir: str) -> str:
        """Generate formatted text report"""
        filename = f"{output_dir}/report_{self.timestamp}.txt"
        
        metrics = self.results.get('metrics', {})
        risk = self.results.get('risk', {})
        config = self.results.get('config', {})
        
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("BACKTEST PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic Information
            f.write("BASIC INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Symbol: {config.get('symbol', 'N/A')}\n")
            f.write(f"Timeframe: {config.get('base_timeframe', 'N/A')}\n")
            f.write(f"Initial Equity: ${config.get('initial_equity', 0):.2f}\n")
            f.write(f"Final Equity: ${self.results.get('final_equity', 0):.2f}\n")
            f.write(f"Total Bars: {self.results.get('total_bars', 0)}\n")
            f.write(f"Completed: {self.results.get('completed_timestamp', 'N/A')}\n\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%\n")
            f.write(f"Total PnL: ${metrics.get('total_net_pnl', 0):.2f}\n")
            f.write(f"Total Trades: {metrics.get('total_trades', 0)}\n")
            f.write(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%\n")
            f.write(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n")
            f.write(f"Expectancy: ${metrics.get('expectancy', 0):.2f}\n\n")
            
            # Risk Metrics
            f.write("RISK METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0)*100:.2f}%\n")
            f.write(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}\n")
            f.write(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n")
            f.write(f"Volatility: {metrics.get('volatility', 0)*100:.2f}%\n\n")
            
            # Trade Analytics
            f.write("TRADE ANALYTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Avg Trade PnL: ${metrics.get('avg_trade_pnl', 0):.2f}\n")
            f.write(f"Avg Winning Trade: ${metrics.get('avg_winning_trade', 0):.2f}\n")
            f.write(f"Avg Losing Trade: ${metrics.get('avg_losing_trade', 0):.2f}\n")
            f.write(f"Largest Win: ${metrics.get('largest_win', 0):.2f}\n")
            f.write(f"Largest Loss: ${metrics.get('largest_loss', 0):.2f}\n")
            f.write(f"Avg MAE: {metrics.get('avg_mae', 0):.2f}%\n")
            f.write(f"Avg MFE: {metrics.get('avg_mfe', 0):.2f}%\n")
            f.write(f"MFE/MAE Ratio: {metrics.get('mfe_mae_ratio', 0):.2f}\n\n")
            
            # Commission & Costs
            f.write("COSTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Commission: ${metrics.get('total_commission', 0):.2f}\n")
            f.write(f"Commission % of PnL: {metrics.get('total_commission', 0)/max(metrics.get('total_net_pnl', 1), 1)*100:.2f}%\n")
        
        return filename
    
    def _calculate_trade_analytics(self) -> Dict[str, Any]:
        """Calculate additional trade analytics"""
        trades = self.results.get('trades', [])
        if not trades:
            return {}
        
        df_trades = pd.DataFrame(trades)
        
        # Monthly performance
        monthly_perf = {}
        if 'exit_timestamp' in df_trades.columns:
            df_trades['exit_date'] = pd.to_datetime(df_trades['exit_timestamp'], unit='ms')
            monthly_data = df_trades.groupby(df_trades['exit_date'].dt.to_period('M'))['net_pnl'].agg(['sum', 'count']).reset_index()
            monthly_perf = monthly_data.to_dict('records')
        
        # Winning/losing streaks
        streaks = self._calculate_streaks(df_trades)
        
        # Holding period analysis
        holding_periods = {
            'avg_bars': df_trades['duration_bars'].mean() if 'duration_bars' in df_trades.columns else 0,
            'avg_seconds': df_trades['duration_seconds'].mean() if 'duration_seconds' in df_trades.columns else 0,
            'min_bars': df_trades['duration_bars'].min() if 'duration_bars' in df_trades.columns else 0,
            'max_bars': df_trades['duration_bars'].max() if 'duration_bars' in df_trades.columns else 0
        }
        
        return {
            'monthly_performance': monthly_perf,
            'streaks': streaks,
            'holding_periods': holding_periods,
            'trade_count': len(trades)
        }
    
    def _calculate_streaks(self, df_trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate winning and losing streaks"""
        if 'net_pnl' not in df_trades.columns or len(df_trades) == 0:
            return {'max_win_streak': 0, 'max_loss_streak': 0}
        
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for pnl in df_trades['net_pnl']:
            if pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
        
        return {
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak
        }
    
    def get_report_summary(self) -> str:
        """Get brief report summary for quick review"""
        metrics = self.results.get('metrics', {})
        
        summary = f"""
Backtest Summary:
----------------
Symbol: {self.results['config'].get('symbol', 'N/A')}
Period: {self.results.get('total_bars', 0)} bars
Initial Equity: ${self.results['config'].get('initial_equity', 0):.2f}
Final Equity: ${self.results.get('final_equity', 0):.2f}
Return: {metrics.get('total_return_pct', 0):.2f}%
Win Rate: {metrics.get('win_rate', 0)*100:.1f}%
Max DD: {metrics.get('max_drawdown_pct', 0)*100:.2f}%
Sharpe: {metrics.get('sharpe_ratio', 0):.2f}
        """.strip()
        
        return summary


# Unit Tests
def test_report_builder():
    """Unit tests for ReportBuilder"""
    print("Testing ReportBuilder...")
    
    # Test data
    test_results = {
        'config': {
            'symbol': 'TEST',
            'base_timeframe': '1m',
            'initial_equity': 10000.0
        },
        'metrics': {
            'total_return_pct': 15.5,
            'total_net_pnl': 1550.0,
            'win_rate': 0.65,
            'max_drawdown_pct': 0.08
        },
        'trades': [],
        'equity_curve': [],
        'orders': [],
        'final_equity': 11550.0,
        'total_bars': 1000,
        'completed_timestamp': '2024-01-01T00:00:00'
    }
    
    # Test 1: Initialization
    builder = ReportBuilder(test_results)
    assert builder.results == test_results
    print("✓ Test 1 passed: Initialization")
    
    # Test 2: Report Generation
    reports = builder.generate_all_reports("test_reports")
    
    assert 'trades' in reports
    assert 'equity' in reports
    assert 'orders' in reports
    assert 'summary' in reports
    assert 'text' in reports
    print("✓ Test 2 passed: Report generation")
    
    # Test 3: Summary Text
    summary = builder.get_report_summary()
    assert "Backtest Summary" in summary
    assert "TEST" in summary
    print("✓ Test 3 passed: Summary text")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_reports", ignore_errors=True)
    
    print("All ReportBuilder tests passed! ✅")

if __name__ == "__main__":
    test_report_builder()