"""
최적화된 RS 전략
벡터화 및 병렬 처리를 활용한 고성능 RS 전략
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from base_strategy import BaseStrategy
from data_utils import data_validator
from optimized_calculations import OptimizedCalculations
from logger import get_logger

class OptimizedRSStrategy(BaseStrategy):
    """최적화된 RS (Relative Strength) 전략"""
    
    def __init__(self, preset_config: Dict[str, Any], 
                 rs_length: int = 20, 
                 rs_timeframe: str = 'daily',
                 rs_recent_cross_days: Optional[int] = None,
                 use_parallel: bool = True):
        """
        Parameters:
        - preset_config: 프리셋 설정 (benchmark, components, name)
        - rs_length: RS 계산 기간
        - rs_timeframe: 시간 프레임 ('daily' or 'weekly')
        - rs_recent_cross_days: 최근 크로스 필터링 기간
        - use_parallel: 병렬 처리 사용 여부
        """
        # 설정 통합
        config = preset_config.copy()
        config.update({
            'rs_length': rs_length,
            'rs_timeframe': rs_timeframe,
            'rs_recent_cross_days': rs_recent_cross_days,
            'use_parallel': use_parallel
        })
        
        super().__init__(config)
        
        self.rs_length = rs_length
        self.rs_timeframe = rs_timeframe
        self.rs_recent_cross_days = rs_recent_cross_days
        self.use_parallel = use_parallel
        
        # 전용 계산기
        self.rs_calculator = OptimizedCalculations()
    
    def _validate_strategy_config(self):
        """전략 설정 검증"""
        if self.rs_length < 5 or self.rs_length > 100:
            raise ValueError(f"rs_length should be between 5 and 100, got {self.rs_length}")
        
        if self.rs_timeframe not in ['daily', 'weekly']:
            raise ValueError(f"rs_timeframe should be 'daily' or 'weekly', got {self.rs_timeframe}")
        
        if self.rs_recent_cross_days is not None and self.rs_recent_cross_days < 5:
            raise ValueError(f"rs_recent_cross_days should be at least 5, got {self.rs_recent_cross_days}")
    
    def select_components(self, price_data: Dict[str, pd.Series], 
                         benchmark_data: pd.Series, 
                         date: datetime) -> List[Dict[str, Any]]:
        """RS 기반 구성요소 선택 (병렬 처리)"""
        self.logger.debug(f"Selecting components for {date.strftime('%Y-%m-%d')}")
        
        # 분석 기간 설정
        if self.rs_timeframe == 'weekly':
            # 주봉 데이터는 금요일 기준
            days_ahead = 4 - date.weekday()
            if days_ahead < 0:
                days_ahead += 7
            analysis_date = date + timedelta(days=days_ahead)
        else:
            analysis_date = date
        
        # 데이터 준비 기간
        lookback_days = self.rs_length * 4
        start_date = analysis_date - timedelta(days=lookback_days)
        
        if self.use_parallel:
            return self._select_components_parallel(
                price_data, benchmark_data, start_date, analysis_date
            )
        else:
            return self._select_components_sequential(
                price_data, benchmark_data, start_date, analysis_date
            )
    
    def _select_components_parallel(self, price_data: Dict[str, pd.Series], 
                                  benchmark_data: pd.Series,
                                  start_date: datetime, 
                                  end_date: datetime) -> List[Dict[str, Any]]:
        """병렬 처리를 사용한 구성요소 선택"""
        selected_components = []
        
        # 벤치마크 데이터 준비
        benchmark_period = benchmark_data[start_date:end_date]
        
        if not data_validator.validate_data_length(benchmark_period, self.rs_length * 2):
            self.logger.warning("Insufficient benchmark data")
            return []
        
        # 병렬 RS 계산
        with ThreadPoolExecutor(max_workers=min(len(price_data), 10)) as executor:
            # 작업 제출
            future_to_ticker = {}
            
            for ticker, price_series in price_data.items():
                if ticker in self.components:
                    future = executor.submit(
                        self._calculate_component_rs,
                        ticker,
                        price_series,
                        benchmark_period,
                        start_date,
                        end_date
                    )
                    future_to_ticker[future] = ticker
            
            # 결과 수집
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result is not None:
                        selected_components.append(result)
                except Exception as e:
                    self.logger.error(f"Error calculating RS for {ticker}: {e}")
        
        # 점수 기준 정렬
        selected_components.sort(
            key=lambda x: x['rs_ratio'] + x['rs_momentum'], 
            reverse=True
        )
        
        return selected_components
    
    def _select_components_sequential(self, price_data: Dict[str, pd.Series], 
                                    benchmark_data: pd.Series,
                                    start_date: datetime, 
                                    end_date: datetime) -> List[Dict[str, Any]]:
        """순차 처리를 사용한 구성요소 선택"""
        selected_components = []
        
        # 벤치마크 데이터 준비
        benchmark_period = benchmark_data[start_date:end_date]
        
        if not data_validator.validate_data_length(benchmark_period, self.rs_length * 2):
            self.logger.warning("Insufficient benchmark data")
            return []
        
        # 각 구성요소 분석
        for ticker, price_series in price_data.items():
            if ticker in self.components:
                result = self._calculate_component_rs(
                    ticker, price_series, benchmark_period, start_date, end_date
                )
                if result is not None:
                    selected_components.append(result)
        
        # 점수 기준 정렬
        selected_components.sort(
            key=lambda x: x['rs_ratio'] + x['rs_momentum'], 
            reverse=True
        )
        
        return selected_components
    
    def _calculate_component_rs(self, ticker: str, price_series: pd.Series,
                              benchmark_period: pd.Series,
                              start_date: datetime, 
                              end_date: datetime) -> Optional[Dict[str, Any]]:
        """개별 구성요소의 RS 계산"""
        try:
            # 기간 데이터 추출
            ticker_period = price_series[start_date:end_date]
            
            # 데이터 검증
            if not data_validator.validate_data_length(ticker_period, self.rs_length * 2):
                return None
            
            # RS 계산 (벡터화)
            rs_components = self.rs_calculator.calculate_rs_components_vectorized(
                ticker_period, benchmark_period, self.rs_length
            )
            
            if rs_components.empty:
                return None
            
            # 최신 값 추출
            latest_rs_ratio = rs_components['rs_ratio'].iloc[-1]
            latest_rs_momentum = rs_components['rs_momentum'].iloc[-1]
            
            # 기본 조건: RS-Ratio >= 100 and RS-Momentum >= 100
            if latest_rs_ratio < 100 or latest_rs_momentum < 100:
                return None
            
            # 최근 크로스 필터링 (선택적)
            if self.rs_recent_cross_days:
                if not self._check_recent_cross(rs_components, self.rs_recent_cross_days):
                    return None
            
            return {
                'ticker': ticker,
                'name': self.components[ticker],
                'rs_ratio': float(latest_rs_ratio),
                'rs_momentum': float(latest_rs_momentum),
                'rs_score': float(latest_rs_ratio + latest_rs_momentum),
                'date': end_date
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating RS for {ticker}: {e}")
            return None
    
    def _check_recent_cross(self, rs_components: pd.DataFrame, 
                          cross_days: int) -> bool:
        """최근 크로스 확인"""
        if len(rs_components) < cross_days:
            return False
        
        recent_data = rs_components.tail(cross_days)
        
        # RS-Ratio가 100을 상향 돌파
        ratio_cross = (
            (recent_data['rs_ratio'].iloc[0] < 100) and 
            (recent_data['rs_ratio'].iloc[-1] >= 100)
        )
        
        # RS-Momentum이 100을 상향 돌파
        momentum_cross = (
            (recent_data['rs_momentum'].iloc[0] < 100) and 
            (recent_data['rs_momentum'].iloc[-1] >= 100)
        )
        
        return ratio_cross or momentum_cross
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """신호 계산"""
        if data.empty:
            return pd.DataFrame()
        
        # RS 계산
        if 'close' in data.columns and 'benchmark' in data.columns:
            rs_components = self.rs_calculator.calculate_rs_components_vectorized(
                data['close'], data['benchmark'], self.rs_length
            )
            
            # 신호 생성
            signals = pd.DataFrame(index=data.index)
            signals['rs_ratio'] = rs_components['rs_ratio']
            signals['rs_momentum'] = rs_components['rs_momentum']
            signals['signal'] = (
                (signals['rs_ratio'] >= 100) & 
                (signals['rs_momentum'] >= 100)
            ).astype(int)
            
            return signals
        
        return pd.DataFrame()
    
    def get_component_rankings(self, date: datetime = None) -> pd.DataFrame:
        """현재 구성요소 순위"""
        if date is None:
            date = datetime.now()
        
        # 데이터 다운로드
        start_date = date - timedelta(days=self.rs_length * 5)
        benchmark_data, price_data = self.download_data(start_date, date, show_progress=False)
        
        # RS 계산
        selected = self.select_components(price_data, benchmark_data, date)
        
        if selected:
            # DataFrame으로 변환
            rankings_df = pd.DataFrame(selected)
            rankings_df['rank'] = range(1, len(rankings_df) + 1)
            
            return rankings_df[['rank', 'ticker', 'name', 'rs_ratio', 'rs_momentum', 'rs_score']]
        
        return pd.DataFrame()
    
    def analyze_correlation(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """구성요소 간 상관관계 분석"""
        # 데이터 다운로드
        benchmark_data, price_data = self.download_data(start_date, end_date, show_progress=False)
        
        if len(price_data) < 2:
            return pd.DataFrame()
        
        # 수익률 계산
        returns_dict = {}
        for ticker, prices in price_data.items():
            returns = prices.pct_change().dropna()
            if len(returns) > 20:
                returns_dict[ticker] = returns
        
        # DataFrame 생성
        returns_df = pd.DataFrame(returns_dict)
        
        # 상관관계 계산
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix

class OptimizedRSWithMA(OptimizedRSStrategy):
    """이동평균 필터가 추가된 RS 전략"""
    
    def __init__(self, preset_config: Dict[str, Any], 
                 ma_period: int = 200, **kwargs):
        super().__init__(preset_config, **kwargs)
        self.ma_period = ma_period
    
    def _calculate_component_rs(self, ticker: str, price_series: pd.Series,
                              benchmark_period: pd.Series,
                              start_date: datetime, 
                              end_date: datetime) -> Optional[Dict[str, Any]]:
        """MA 필터가 추가된 RS 계산"""
        # 기본 RS 계산
        result = super()._calculate_component_rs(
            ticker, price_series, benchmark_period, start_date, end_date
        )
        
        if result is None:
            return None
        
        # MA 필터 확인
        if len(price_series) >= self.ma_period:
            current_price = price_series.iloc[-1]
            ma_value = price_series.rolling(window=self.ma_period).mean().iloc[-1]
            
            # 현재가가 MA 위에 있어야 함
            if current_price < ma_value:
                return None
            
            # MA 정보 추가
            result['ma_filter'] = True
            result['price_to_ma'] = (current_price / ma_value - 1) * 100
        
        return result

# 편의 함수
def create_optimized_rs_strategy(preset_config: Dict[str, Any], **kwargs) -> OptimizedRSStrategy:
    """최적화된 RS 전략 생성"""
    return OptimizedRSStrategy(preset_config, **kwargs)

def quick_rs_analysis(preset_config: Dict[str, Any], 
                     date: datetime = None) -> pd.DataFrame:
    """빠른 RS 분석"""
    strategy = OptimizedRSStrategy(preset_config)
    return strategy.get_component_rankings(date)
