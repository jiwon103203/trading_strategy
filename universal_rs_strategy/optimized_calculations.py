"""
최적화된 계산 모듈
Numpy 벡터화를 활용한 고속 계산
"""

import numpy as np
import pandas as pd
from numba import jit, njit
from typing import Union, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class OptimizedCalculations:
    """벡터화된 계산 메서드"""
    
    @staticmethod
    @njit
    def calculate_wma_numba(values: np.ndarray, period: int) -> np.ndarray:
        """Numba를 사용한 고속 가중이동평균"""
        n = len(values)
        result = np.empty(n)
        result[:period-1] = np.nan
        
        # 가중치 계산
        weights = np.arange(1, period + 1, dtype=np.float64)
        weight_sum = weights.sum()
        
        # 초기 윈도우 계산
        for i in range(period - 1, n):
            window_sum = 0.0
            for j in range(period):
                window_sum += values[i - period + 1 + j] * weights[j]
            result[i] = window_sum / weight_sum
        
        return result
    
    @staticmethod
    def calculate_wma_vectorized(data: Union[np.ndarray, pd.Series], period: int) -> Union[np.ndarray, pd.Series]:
        """벡터화된 가중이동평균"""
        if isinstance(data, pd.Series):
            values = data.values
            index = data.index
            result = OptimizedCalculations.calculate_wma_numba(values, period)
            return pd.Series(result, index=index)
        else:
            return OptimizedCalculations.calculate_wma_numba(data, period)
    
    @staticmethod
    def calculate_rs_components_vectorized(price_data: pd.Series, 
                                         benchmark_data: pd.Series, 
                                         length: int) -> pd.DataFrame:
        """벡터화된 RS 계산"""
        # 데이터 정렬
        aligned = pd.DataFrame({
            'price': price_data,
            'benchmark': benchmark_data
        }).dropna()
        
        if len(aligned) < length * 2:
            return pd.DataFrame()
        
        # Numpy 배열로 변환
        price_values = aligned['price'].values
        bench_values = aligned['benchmark'].values
        
        # RS 비율 계산 (벡터화)
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(bench_values != 0, price_values / bench_values, np.nan)
        
        # WMA 계산 (벡터화)
        wma_rs = OptimizedCalculations.calculate_wma_numba(rs, length)
        
        # RS Ratio 계산 (벡터화)
        with np.errstate(divide='ignore', invalid='ignore'):
            rs_ratio_raw = np.where(wma_rs != 0, rs / wma_rs * 100, 100)
        
        rs_ratio = OptimizedCalculations.calculate_wma_numba(rs_ratio_raw, length)
        
        # RS Momentum 계산 (벡터화)
        wma_rs_ratio = OptimizedCalculations.calculate_wma_numba(rs_ratio, length)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs_momentum = np.where(wma_rs_ratio != 0, rs_ratio / wma_rs_ratio * 100, 100)
        
        # NaN 처리
        rs_ratio = np.nan_to_num(rs_ratio, nan=100.0)
        rs_momentum = np.nan_to_num(rs_momentum, nan=100.0)
        
        return pd.DataFrame({
            'rs_ratio': rs_ratio,
            'rs_momentum': rs_momentum
        }, index=aligned.index)
    
    @staticmethod
    @njit
    def calculate_rolling_stats_numba(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba를 사용한 롤링 통계 (평균, 표준편차)"""
        n = len(values)
        means = np.empty(n)
        stds = np.empty(n)
        
        means[:window-1] = np.nan
        stds[:window-1] = np.nan
        
        for i in range(window - 1, n):
            window_data = values[i - window + 1:i + 1]
            means[i] = np.mean(window_data)
            stds[i] = np.std(window_data)
        
        return means, stds
    
    @staticmethod
    def calculate_sharpe_ratio_vectorized(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """벡터화된 샤프 비율 계산"""
        if len(returns) == 0:
            return 0.0
        
        # 일일 무위험 수익률
        daily_rf = risk_free_rate / 252
        
        # 초과 수익률
        excess_returns = returns - daily_rf
        
        # 연율화
        mean_excess = np.mean(excess_returns) * 252
        std_excess = np.std(excess_returns) * np.sqrt(252)
        
        if std_excess == 0:
            return 0.0
        
        return mean_excess / std_excess
    
    @staticmethod
    def calculate_sortino_ratio_vectorized(returns: np.ndarray, 
                                         risk_free_rate: float = 0.02,
                                         target_return: float = 0.0) -> float:
        """벡터화된 소르티노 비율 계산"""
        if len(returns) == 0:
            return 0.0
        
        # 일일 무위험 수익률
        daily_rf = risk_free_rate / 252
        daily_target = target_return / 252
        
        # 초과 수익률
        excess_returns = returns - daily_rf
        
        # 하방 편차 (target 이하의 수익률만)
        downside_returns = excess_returns[excess_returns < daily_target]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        # 연율화
        mean_excess = np.mean(excess_returns) * 252
        downside_std = np.std(downside_returns) * np.sqrt(252)
        
        if downside_std == 0:
            return float('inf')
        
        return mean_excess / downside_std
    
    @staticmethod
    @njit
    def calculate_max_drawdown_numba(values: np.ndarray) -> Tuple[float, int, int]:
        """Numba를 사용한 최대 낙폭 계산"""
        n = len(values)
        if n < 2:
            return 0.0, 0, 0
        
        max_dd = 0.0
        peak_idx = 0
        trough_idx = 0
        current_peak_idx = 0
        current_peak = values[0]
        
        for i in range(1, n):
            if values[i] > current_peak:
                current_peak = values[i]
                current_peak_idx = i
            
            dd = (current_peak - values[i]) / current_peak
            
            if dd > max_dd:
                max_dd = dd
                peak_idx = current_peak_idx
                trough_idx = i
        
        return max_dd * 100, peak_idx, trough_idx
    
    @staticmethod
    def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series, 
                                    window: int = 20) -> pd.Series:
        """롤링 상관계수 계산"""
        # 데이터 정렬
        aligned = pd.DataFrame({
            's1': series1,
            's2': series2
        }).dropna()
        
        if len(aligned) < window:
            return pd.Series()
        
        # 롤링 상관계수
        return aligned['s1'].rolling(window).corr(aligned['s2'])
    
    @staticmethod
    def calculate_beta_vectorized(asset_returns: np.ndarray, 
                                market_returns: np.ndarray) -> float:
        """벡터화된 베타 계산"""
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 1.0
        
        # 공분산과 시장 분산
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    @staticmethod
    def calculate_information_ratio(returns: np.ndarray, 
                                  benchmark_returns: np.ndarray) -> float:
        """정보 비율 계산"""
        if len(returns) != len(benchmark_returns) or len(returns) == 0:
            return 0.0
        
        # 초과 수익률
        excess_returns = returns - benchmark_returns
        
        # 추적 오차
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        # 연율화
        annual_excess_return = np.mean(excess_returns) * 252
        annual_tracking_error = tracking_error * np.sqrt(252)
        
        return annual_excess_return / annual_tracking_error

class TechnicalIndicators:
    """기술적 지표 계산 (벡터화)"""
    
    @staticmethod
    def calculate_rsi_vectorized(prices: pd.Series, period: int = 14) -> pd.Series:
        """벡터화된 RSI 계산"""
        if len(prices) < period:
            return pd.Series()
        
        # 가격 변화
        delta = prices.diff()
        
        # 상승/하락 분리
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # 초기 평균
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # RS 계산
        rs = avg_gains / avg_losses
        
        # RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                                num_std: float = 2.0) -> pd.DataFrame:
        """볼린저 밴드 계산"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': (upper - lower) / middle,
            'percent_b': (prices - lower) / (upper - lower)
        })
    
    @staticmethod
    def calculate_atr_vectorized(high: pd.Series, low: pd.Series, 
                               close: pd.Series, period: int = 14) -> pd.Series:
        """벡터화된 ATR (Average True Range) 계산"""
        # True Range 계산
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # ATR
        atr = true_range.rolling(window=period).mean()
        
        return atr

# 성능 벤치마크 함수
def benchmark_calculations():
    """계산 성능 벤치마크"""
    import time
    
    # 테스트 데이터 생성
    n = 10000
    prices = pd.Series(np.random.randn(n).cumsum() + 100, 
                      index=pd.date_range('2020-01-01', periods=n))
    benchmark = pd.Series(np.random.randn(n).cumsum() + 100, 
                         index=pd.date_range('2020-01-01', periods=n))
    
    # WMA 벤치마크
    start = time.time()
    wma_result = OptimizedCalculations.calculate_wma_vectorized(prices, 20)
    wma_time = time.time() - start
    
    # RS 계산 벤치마크
    start = time.time()
    rs_result = OptimizedCalculations.calculate_rs_components_vectorized(prices, benchmark, 20)
    rs_time = time.time() - start
    
    # 최대 낙폭 벤치마크
    start = time.time()
    dd_result = OptimizedCalculations.calculate_max_drawdown_numba(prices.values)
    dd_time = time.time() - start
    
    print(f"Performance Benchmark Results:")
    print(f"WMA Calculation: {wma_time:.4f}s for {n} points")
    print(f"RS Calculation: {rs_time:.4f}s for {n} points")
    print(f"Max Drawdown: {dd_time:.4f}s for {n} points")
    
    return {
        'wma_time': wma_time,
        'rs_time': rs_time,
        'dd_time': dd_time,
        'data_points': n
    }

# 전역 인스턴스
calculator = OptimizedCalculations()
indicators = TechnicalIndicators()
