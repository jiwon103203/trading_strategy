"""
기본 전략 클래스
모든 전략의 추상 기본 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 유틸리티 모듈 import
from data_cache import data_cache
from parallel_downloader import SmartDataDownloader
from data_utils import data_validator, data_converter
from optimized_calculations import OptimizedCalculations
from config_manager import config_manager
from logger import get_logger
from memory_optimization import MemoryOptimizer, memory_monitor

class BaseStrategy(ABC):
    """모든 전략의 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Parameters:
        - config: 전략 설정 딕셔너리
            - benchmark: 벤치마크 티커
            - components: 구성요소 딕셔너리 {ticker: name}
            - name: 전략 이름
            - 기타 전략별 설정
        """
        self.config = config
        self.benchmark = config.get('benchmark')
        self.components = config.get('components', {})
        self.strategy_name = config.get('name', 'Strategy')
        
        # 설정 관리자
        self.global_config = config_manager.config
        
        # 로거 초기화
        self.logger = get_logger(self.strategy_name)
        
        # 데이터 다운로더 초기화
        self.data_downloader = SmartDataDownloader(
            max_workers=self.global_config.max_parallel_downloads
        )
        
        # 계산 모듈 초기화
        self.calculator = OptimizedCalculations()
        
        # 메모리 최적화
        self.memory_optimizer = MemoryOptimizer()
        
        # 검증
        self._validate_config()
        
        self.logger.info(f"{self.strategy_name} 초기화 완료")
        self.logger.info(f"벤치마크: {self.benchmark}")
        self.logger.info(f"구성요소: {len(self.components)}개")
    
    def _validate_config(self):
        """설정 검증"""
        if not self.benchmark:
            raise ValueError("Benchmark ticker is required")
        
        if not self.components:
            raise ValueError("At least one component is required")
        
        # 추가 검증은 하위 클래스에서 구현
        self._validate_strategy_config()
    
    @abstractmethod
    def _validate_strategy_config(self):
        """전략별 설정 검증 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def select_components(self, price_data: Dict[str, pd.Series], 
                         benchmark_data: pd.Series, 
                         date: datetime) -> List[Dict[str, Any]]:
        """
        구성요소 선택 로직 (하위 클래스에서 구현)
        
        Returns:
            선택된 구성요소 리스트 [{ticker, name, score, ...}]
        """
        pass
    
    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """신호 계산 로직 (하위 클래스에서 구현)"""
        pass
    
    def download_data(self, start_date: datetime, end_date: datetime, 
                     show_progress: bool = True) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """데이터 다운로드 (캐시 및 병렬 처리)"""
        with self.logger.timer("Data Download"):
            # 모든 티커 리스트
            all_tickers = [self.benchmark] + list(self.components.keys())
            
            # 병렬 다운로드
            data_dict, failed = self.data_downloader.download_multiple_tickers(
                all_tickers, start_date, end_date, show_progress
            )
            
            if failed:
                self.logger.warning(f"Failed to download: {[t for t, _ in failed]}")
            
            # 벤치마크 추출
            benchmark_data = data_dict.pop(self.benchmark, None)
            if benchmark_data is None:
                raise ValueError(f"Failed to download benchmark {self.benchmark}")
            
            # Series 추출
            benchmark_series = data_validator.safe_extract_series(benchmark_data, 'Close')
            
            component_series = {}
            for ticker, data in data_dict.items():
                series = data_validator.safe_extract_series(data, 'Close')
                if series is not None and len(series) >= self.global_config.rs_length:
                    component_series[ticker] = series
            
            self.logger.info(f"Downloaded: {len(component_series)}/{len(self.components)} components")
            
            return benchmark_series, component_series
    
    def backtest(self, start_date: datetime, end_date: datetime, 
                 initial_capital: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """백테스트 실행 (공통 로직)"""
        if initial_capital is None:
            initial_capital = self.global_config.initial_capital
        
        self.logger.info(f"백테스트 시작: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        with memory_monitor.memory_tracker("Backtest"):
            # 데이터 준비
            extra_days = 200 if self.global_config.rs_timeframe == 'weekly' else 100
            extended_start = start_date - timedelta(days=extra_days)
            
            benchmark_data, price_data = self.download_data(extended_start, end_date)
            
            # 리밸런싱 날짜
            rebalance_dates = pd.date_range(
                start=start_date, 
                end=end_date, 
                freq=self.global_config.rebalance_frequency
            )
            
            # 백테스트 실행
            portfolio_df, trades_df = self._run_backtest(
                price_data, benchmark_data, rebalance_dates, initial_capital
            )
            
            # 성과 지표 계산
            if not portfolio_df.empty:
                metrics = self.calculate_performance_metrics(portfolio_df)
                self.logger.log_backtest_summary(start_date, end_date, metrics)
            else:
                metrics = {}
            
            return portfolio_df, trades_df, metrics
    
    def _run_backtest(self, price_data: Dict[str, pd.Series], benchmark_data: pd.Series,
                      rebalance_dates: pd.DatetimeIndex, initial_capital: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """백테스트 실행 로직"""
        portfolio_value = initial_capital
        cash = initial_capital
        holdings = {}
        portfolio_history = []
        trade_history = []
        
        # 거래 비용
        commission_rate = self.global_config.transaction_cost
        slippage_rate = self.global_config.slippage
        
        for i, rebal_date in enumerate(rebalance_dates):
            self.logger.debug(f"리밸런싱: {rebal_date.strftime('%Y-%m-%d')}")
            
            # 이전 보유 종목 가치 평가
            if holdings:
                portfolio_value = cash
                for ticker, position in holdings.items():
                    if ticker in price_data and rebal_date in price_data[ticker].index:
                        current_price = price_data[ticker].loc[rebal_date]
                        portfolio_value += position['shares'] * current_price
            
            # 구성요소 선택
            selected = self.select_components(price_data, benchmark_data, rebal_date)
            
            if not selected:
                # 모든 포지션 청산
                if holdings:
                    for ticker, position in holdings.items():
                        if ticker in price_data and rebal_date in price_data[ticker].index:
                            exit_price = price_data[ticker].loc[rebal_date]
                            exit_value = position['shares'] * exit_price * (1 - slippage_rate)
                            cash += exit_value * (1 - commission_rate)
                            
                            trade_history.append({
                                'date': rebal_date,
                                'ticker': ticker,
                                'name': position['name'],
                                'action': 'SELL',
                                'shares': position['shares'],
                                'price': exit_price,
                                'value': exit_value
                            })
                    holdings = {}
                
                portfolio_value = cash
            else:
                # 새로운 포트폴리오 구성
                new_holdings = self._rebalance_portfolio(
                    selected, holdings, price_data, rebal_date, 
                    portfolio_value, cash, commission_rate, slippage_rate
                )
                
                # 거래 기록
                for trade in new_holdings['trades']:
                    trade_history.append(trade)
                    self.logger.log_trade(trade)
                
                holdings = new_holdings['holdings']
                cash = new_holdings['cash']
            
            # 일별 포트폴리오 추적
            next_date = rebalance_dates[i + 1] if i < len(rebalance_dates) - 1 else end_date
            
            for date in pd.date_range(start=rebal_date, end=next_date, freq='D'):
                if date <= end_date:
                    daily_value = cash
                    
                    for ticker, position in holdings.items():
                        if ticker in price_data and date in price_data[ticker].index:
                            current_price = price_data[ticker].loc[date]
                            daily_value += position['shares'] * current_price
                        else:
                            # 이전 가격 사용
                            daily_value += position['shares'] * position['last_price']
                    
                    portfolio_history.append({
                        'date': date,
                        'value': daily_value,
                        'cash': cash,
                        'holdings': len(holdings),
                        'invested': daily_value - cash
                    })
        
        # 결과 정리
        portfolio_df = pd.DataFrame(portfolio_history).set_index('date')
        trades_df = pd.DataFrame(trade_history)
        
        # 메모리 최적화
        portfolio_df = self.memory_optimizer.optimize_dataframe(portfolio_df)
        if not trades_df.empty:
            trades_df = self.memory_optimizer.optimize_dataframe(trades_df)
        
        return portfolio_df, trades_df
    
    def _rebalance_portfolio(self, selected: List[Dict], current_holdings: Dict,
                           price_data: Dict[str, pd.Series], date: datetime,
                           portfolio_value: float, cash: float,
                           commission_rate: float, slippage_rate: float) -> Dict[str, Any]:
        """포트폴리오 리밸런싱"""
        trades = []
        new_holdings = {}
        
        # 현재 보유 종목 중 선택되지 않은 것 매도
        selected_tickers = {s['ticker'] for s in selected}
        
        for ticker, position in current_holdings.items():
            if ticker not in selected_tickers:
                if ticker in price_data and date in price_data[ticker].index:
                    exit_price = price_data[ticker].loc[date]
                    exit_value = position['shares'] * exit_price * (1 - slippage_rate)
                    cash += exit_value * (1 - commission_rate)
                    
                    trades.append({
                        'date': date,
                        'ticker': ticker,
                        'name': position['name'],
                        'action': 'SELL',
                        'shares': position['shares'],
                        'price': exit_price,
                        'value': exit_value
                    })
        
        # 새로운 종목 매수
        if selected:
            # 동일 가중 배분
            target_value_per_component = portfolio_value / len(selected)
            
            for component in selected:
                ticker = component['ticker']
                
                if ticker in price_data and date in price_data[ticker].index:
                    current_price = price_data[ticker].loc[date]
                    
                    # 현재 보유량 확인
                    current_shares = current_holdings.get(ticker, {}).get('shares', 0)
                    current_value = current_shares * current_price
                    
                    # 목표 주식 수 계산
                    target_shares = int(target_value_per_component / current_price)
                    shares_to_trade = target_shares - current_shares
                    
                    if shares_to_trade > 0:  # 매수
                        trade_value = shares_to_trade * current_price * (1 + slippage_rate)
                        if cash >= trade_value * (1 + commission_rate):
                            cash -= trade_value * (1 + commission_rate)
                            
                            trades.append({
                                'date': date,
                                'ticker': ticker,
                                'name': component['name'],
                                'action': 'BUY',
                                'shares': shares_to_trade,
                                'price': current_price,
                                'value': trade_value
                            })
                            
                            new_holdings[ticker] = {
                                'shares': target_shares,
                                'name': component['name'],
                                'last_price': current_price
                            }
                    
                    elif shares_to_trade < 0:  # 매도
                        exit_value = abs(shares_to_trade) * current_price * (1 - slippage_rate)
                        cash += exit_value * (1 - commission_rate)
                        
                        trades.append({
                            'date': date,
                            'ticker': ticker,
                            'name': component['name'],
                            'action': 'SELL_PARTIAL',
                            'shares': abs(shares_to_trade),
                            'price': current_price,
                            'value': exit_value
                        })
                        
                        if target_shares > 0:
                            new_holdings[ticker] = {
                                'shares': target_shares,
                                'name': component['name'],
                                'last_price': current_price
                            }
                    
                    else:  # 변동 없음
                        new_holdings[ticker] = current_holdings[ticker]
                        new_holdings[ticker]['last_price'] = current_price
        
        return {
            'holdings': new_holdings,
            'cash': cash,
            'trades': trades
        }
    
    def calculate_performance_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """성과 지표 계산"""
        if portfolio_df.empty:
            return {}
        
        metrics = {}
        
        # 기본 수익률 지표
        initial_value = portfolio_df['value'].iloc[0]
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        # 기간
        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        years = days / 365.25
        
        # 연율화 수익률
        annual_return = (np.power(1 + total_return/100, 1/years) - 1) * 100 if years > 0 else 0
        
        # 일일 수익률
        daily_returns = portfolio_df['value'].pct_change().dropna()
        
        # 변동성
        annual_volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # 샤프 비율
        rf_rate = self.global_config.default_rf_rate
        sharpe_ratio = self.calculator.calculate_sharpe_ratio_vectorized(
            daily_returns.values, rf_rate
        )
        
        # 소르티노 비율
        sortino_ratio = self.calculator.calculate_sortino_ratio_vectorized(
            daily_returns.values, rf_rate
        )
        
        # 최대 낙폭
        max_dd, peak_idx, trough_idx = self.calculator.calculate_max_drawdown_numba(
            portfolio_df['value'].values
        )
        
        # 승률
        positive_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_rate = positive_days / total_days * 100 if total_days > 0 else 0
        
        metrics = {
            '총 수익률': f"{total_return:.2f}%",
            '연율화 수익률': f"{annual_return:.2f}%",
            '연율화 변동성': f"{annual_volatility:.2f}%",
            '샤프 비율': f"{sharpe_ratio:.3f}",
            '소르티노 비율': f"{sortino_ratio:.3f}" if sortino_ratio != float('inf') else "∞",
            '최대 낙폭': f"{max_dd:.2f}%",
            '승률': f"{win_rate:.1f}%",
            '거래일수': days,
            '초기 자본': f"{initial_value:,.0f}",
            '최종 자본': f"{final_value:,.0f}"
        }
        
        return metrics
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """데이터 검증"""
        return data_validator.validate_data_length(data, self.global_config.rs_length * 2)
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 정리"""
        return data_validator.clean_price_data(data)
