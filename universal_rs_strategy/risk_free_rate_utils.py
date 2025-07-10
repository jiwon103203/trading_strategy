"""
Risk-Free Rate 유틸리티
미국 3개월물 금리 (^IRX)를 사용한 동적 risk-free rate 계산
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RiskFreeRateManager:
    """Risk-Free Rate 관리 클래스"""
    
    def __init__(self, rf_ticker='^IRX', default_rate=0.02):
        """
        Parameters:
        - rf_ticker: 3개월물 금리 티커 (기본: ^IRX)
        - default_rate: 기본 금리 (데이터 없을 때 사용, 기본: 2%)
        """
        self.rf_ticker = rf_ticker
        self.default_rate = default_rate
        self.rf_data = None
    
    def download_risk_free_rate(self, start_date, end_date):
        """
        Risk-free rate 데이터 다운로드
        
        Parameters:
        - start_date: 시작일
        - end_date: 종료일
        
        Returns:
        - pandas.Series: 일일 risk-free rate (소수점 형태, 예: 0.05 = 5%)
        """
        try:
            print(f"Risk-free rate 데이터 다운로드 중... ({self.rf_ticker})")
            
            # 약간 여유를 두고 데이터 다운로드
            extended_start = start_date - timedelta(days=30)
            
            rf_df = yf.download(
                self.rf_ticker,
                start=extended_start,
                end=end_date + timedelta(days=1),
                progress=False,
                auto_adjust=True
            )
            
            if rf_df.empty:
                print(f"Warning: {self.rf_ticker} 데이터가 없습니다. 기본 금리 {self.default_rate*100:.1f}% 사용")
                return self._create_default_rf_series(start_date, end_date)
            
            # Close 가격 추출 (연율 %)
            if 'Close' in rf_df.columns:
                rf_series = rf_df['Close']
            else:
                rf_series = rf_df.iloc[:, 0]  # 첫 번째 컬럼 사용
            
            # NaN 값 처리
            rf_series = rf_series.fillna(method='ffill').fillna(method='bfill')
            
            # 백분율을 소수점으로 변환 (예: 5.0 -> 0.05)
            rf_series = rf_series / 100.0
            
            # 요청 기간으로 제한
            rf_series = rf_series[start_date:end_date]
            
            if rf_series.empty:
                print(f"Warning: 요청 기간의 {self.rf_ticker} 데이터가 없습니다. 기본 금리 사용")
                return self._create_default_rf_series(start_date, end_date)
            
            print(f"Risk-free rate 데이터: {len(rf_series)}개 (평균: {rf_series.mean()*100:.2f}%)")
            
            self.rf_data = rf_series
            return rf_series
            
        except Exception as e:
            print(f"Risk-free rate 다운로드 실패: {e}")
            print(f"기본 금리 {self.default_rate*100:.1f}% 사용")
            return self._create_default_rf_series(start_date, end_date)
    
    def _create_default_rf_series(self, start_date, end_date):
        """기본 risk-free rate 시리즈 생성"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.Series(self.default_rate, index=dates)
    
    def get_daily_risk_free_rate(self, portfolio_dates):
        """
        포트폴리오 날짜에 맞는 일일 risk-free rate 반환
        
        Parameters:
        - portfolio_dates: 포트폴리오 날짜 인덱스
        
        Returns:
        - pandas.Series: 일일 risk-free rate (연율을 252로 나눈 값)
        """
        if self.rf_data is None:
            print("Warning: Risk-free rate 데이터가 없습니다. 기본값 사용")
            return pd.Series(self.default_rate / 252, index=portfolio_dates)
        
        # 포트폴리오 날짜에 맞춰 정렬 (forward fill)
        aligned_rf = self.rf_data.reindex(portfolio_dates, method='ffill')
        
        # NaN 값이 있으면 기본값으로 채움
        aligned_rf = aligned_rf.fillna(self.default_rate)
        
        # 일일 risk-free rate로 변환 (연율 / 252)
        daily_rf = aligned_rf / 252
        
        return daily_rf
    
    def calculate_sharpe_ratio(self, returns, portfolio_dates):
        """
        동적 risk-free rate를 사용한 Sharpe ratio 계산
        
        Parameters:
        - returns: 일일 수익률 시리즈
        - portfolio_dates: 포트폴리오 날짜
        
        Returns:
        - float: Sharpe ratio
        """
        try:
            daily_rf = self.get_daily_risk_free_rate(portfolio_dates)
            
            # 수익률과 risk-free rate 정렬
            aligned_returns = returns.reindex(portfolio_dates)
            
            # 초과 수익률 계산
            excess_returns = aligned_returns - daily_rf
            
            # Sharpe ratio 계산
            if excess_returns.std() > 0:
                sharpe = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
                return float(sharpe)
            else:
                return 0.0
                
        except Exception as e:
            print(f"Sharpe ratio 계산 오류: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns, portfolio_dates):
        """
        동적 risk-free rate를 사용한 Sortino ratio 계산
        
        Parameters:
        - returns: 일일 수익률 시리즈
        - portfolio_dates: 포트폴리오 날짜
        
        Returns:
        - float: Sortino ratio
        """
        try:
            daily_rf = self.get_daily_risk_free_rate(portfolio_dates)
            
            # 수익률과 risk-free rate 정렬
            aligned_returns = returns.reindex(portfolio_dates)
            
            # 초과 수익률 계산
            excess_returns = aligned_returns - daily_rf
            
            # 하방 변동성 계산 (음의 초과 수익률만)
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) > 0:
                downside_volatility = downside_returns.std() * np.sqrt(252)
                annual_excess_return = excess_returns.mean() * 252
                
                if downside_volatility > 0:
                    sortino = annual_excess_return / downside_volatility
                    return float(sortino)
                else:
                    return float('inf')  # 하방 변동성이 0인 경우
            else:
                return float('inf')  # 모든 수익률이 양수인 경우
                
        except Exception as e:
            print(f"Sortino ratio 계산 오류: {e}")
            return 0.0
    
    def get_risk_free_rate_stats(self, start_date, end_date):
        """Risk-free rate 통계 정보"""
        if self.rf_data is None:
            self.download_risk_free_rate(start_date, end_date)
        
        if self.rf_data is not None and not self.rf_data.empty:
            stats = {
                'mean_rate': self.rf_data.mean() * 100,
                'min_rate': self.rf_data.min() * 100,
                'max_rate': self.rf_data.max() * 100,
                'std_rate': self.rf_data.std() * 100,
                'start_rate': self.rf_data.iloc[0] * 100,
                'end_rate': self.rf_data.iloc[-1] * 100
            }
            return stats
        else:
            return {
                'mean_rate': self.default_rate * 100,
                'min_rate': self.default_rate * 100,
                'max_rate': self.default_rate * 100,
                'std_rate': 0.0,
                'start_rate': self.default_rate * 100,
                'end_rate': self.default_rate * 100
            }


# 전역 risk-free rate 관리자 (싱글톤 패턴)
_global_rf_manager = None

def get_risk_free_manager():
    """전역 RiskFreeRateManager 인스턴스 반환"""
    global _global_rf_manager
    if _global_rf_manager is None:
        _global_rf_manager = RiskFreeRateManager()
    return _global_rf_manager

def calculate_dynamic_sharpe_ratio(portfolio_df, rf_ticker='^IRX', default_rate=0.02):
    """
    동적 risk-free rate를 사용한 Sharpe ratio 계산 (편의 함수)
    
    Parameters:
    - portfolio_df: 포트폴리오 데이터프레임 ('value' 컬럼 필요)
    - rf_ticker: risk-free rate 티커
    - default_rate: 기본 금리
    
    Returns:
    - float: Sharpe ratio
    """
    try:
        # 수익률 계산
        returns = portfolio_df['value'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Risk-free rate 관리자
        rf_manager = RiskFreeRateManager(rf_ticker, default_rate)
        
        # Risk-free rate 다운로드
        start_date = portfolio_df.index[0]
        end_date = portfolio_df.index[-1]
        rf_manager.download_risk_free_rate(start_date, end_date)
        
        # Sharpe ratio 계산
        return rf_manager.calculate_sharpe_ratio(returns, portfolio_df.index)
        
    except Exception as e:
        print(f"동적 Sharpe ratio 계산 실패: {e}")
        return 0.0

def calculate_dynamic_sortino_ratio(portfolio_df, rf_ticker='^IRX', default_rate=0.02):
    """
    동적 risk-free rate를 사용한 Sortino ratio 계산 (편의 함수)
    
    Parameters:
    - portfolio_df: 포트폴리오 데이터프레임 ('value' 컬럼 필요)
    - rf_ticker: risk-free rate 티커
    - default_rate: 기본 금리
    
    Returns:
    - float: Sortino ratio
    """
    try:
        # 수익률 계산
        returns = portfolio_df['value'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Risk-free rate 관리자
        rf_manager = RiskFreeRateManager(rf_ticker, default_rate)
        
        # Risk-free rate 다운로드
        start_date = portfolio_df.index[0]
        end_date = portfolio_df.index[-1]
        rf_manager.download_risk_free_rate(start_date, end_date)
        
        # Sortino ratio 계산
        return rf_manager.calculate_sortino_ratio(returns, portfolio_df.index)
        
    except Exception as e:
        print(f"동적 Sortino ratio 계산 실패: {e}")
        return 0.0


# 사용 예시
if __name__ == "__main__":
    # 테스트 데이터 생성
    import matplotlib.pyplot as plt
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Risk-free rate 관리자 생성
    rf_manager = RiskFreeRateManager()
    
    # Risk-free rate 다운로드
    rf_data = rf_manager.download_risk_free_rate(start_date, end_date)
    
    print(f"\n=== Risk-Free Rate 통계 ===")
    stats = rf_manager.get_risk_free_rate_stats(start_date, end_date)
    for key, value in stats.items():
        print(f"{key}: {value:.3f}%")
    
    # 차트로 표시
    if rf_data is not None and not rf_data.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(rf_data.index, rf_data * 100, linewidth=2, label='3개월물 금리 (%)')
        plt.title('미국 3개월물 국채 금리 (^IRX)')
        plt.xlabel('날짜')
        plt.ylabel('금리 (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print(f"\n최신 3개월물 금리: {rf_data.iloc[-1]*100:.3f}%")
        print(f"일일 risk-free rate: {rf_data.iloc[-1]/252*100:.6f}%")
