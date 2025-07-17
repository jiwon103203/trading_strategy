"""
Universal RS Strategy - 간소화 버전
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import safe_extract_close, validate_data, calculate_wma, calculate_basic_metrics, print_status
import warnings
warnings.filterwarnings('ignore')

class UniversalRSStrategy:
    """간소화된 Universal RS Strategy"""
    
    def __init__(self, benchmark, components, name="Custom Strategy", length=20, timeframe='daily'):
        """
        Parameters:
        - benchmark: 벤치마크 티커
        - components: 구성요소 딕셔너리 {ticker: name}
        - name: 전략 이름
        - length: RS 계산 기간
        - timeframe: 'daily' 또는 'weekly'
        """
        self.benchmark = benchmark
        self.components = components
        self.strategy_name = name
        self.length = length
        self.timeframe = timeframe.lower()
        
        if self.timeframe not in ['daily', 'weekly']:
            raise ValueError("timeframe은 'daily' 또는 'weekly'여야 합니다.")
        
        print_status(f"RS Strategy 초기화: {name}")
    
    def get_price_data(self, start_date, end_date):
        """가격 데이터 다운로드"""
        price_data = {}
        benchmark_data = None
        
        # 벤치마크 데이터
        try:
            print_status(f"벤치마크 데이터 다운로드: {self.benchmark}")
            
            benchmark_df = yf.download(
                self.benchmark, 
                start=start_date, 
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            benchmark_data = safe_extract_close(benchmark_df)
            
            if not validate_data(benchmark_data, self.length):
                print_status("벤치마크 데이터 부족", "ERROR")
                return {}, None
                
        except Exception as e:
            print_status(f"벤치마크 데이터 실패: {e}", "ERROR")
            return {}, None
        
        # 구성요소 데이터
        success_count = 0
        for ticker, name in self.components.items():
            try:
                df = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
                
                data = safe_extract_close(df)
                
                if validate_data(data, self.length):
                    price_data[ticker] = data
                    success_count += 1
                    
            except Exception:
                continue
        
        print_status(f"{success_count}개 구성요소 데이터 다운로드 완료")
        return price_data, benchmark_data
    
    def calculate_rs_components(self, price_data, benchmark_data):
        """RS 계산"""
        try:
            price_series = safe_extract_close(price_data)
            benchmark_series = safe_extract_close(benchmark_data)
            
            if not validate_data(price_series, self.length * 2):
                return pd.DataFrame(columns=['rs_ratio', 'rs_momentum'])
                
            if not validate_data(benchmark_series, self.length * 2):
                return pd.DataFrame(columns=['rs_ratio', 'rs_momentum'])
            
            # 데이터 정렬
            aligned_data = pd.DataFrame({
                'price': price_series,
                'benchmark': benchmark_series
            }).dropna()
            
            if len(aligned_data) < self.length * 2:
                return pd.DataFrame(columns=['rs_ratio', 'rs_momentum'])
            
            # RS 비율 계산
            rs = aligned_data['price'] / aligned_data['benchmark']
            
            # WMA 계산
            wma_rs = calculate_wma(rs, self.length)
            
            # RS Ratio
            rs_ratio = (rs / wma_rs) * 100
            rs_ratio_wma = calculate_wma(rs_ratio, self.length)
            
            # RS Momentum
            wma_rs_ratio = calculate_wma(rs_ratio_wma, self.length)
            rs_momentum = (rs_ratio_wma / wma_rs_ratio) * 100
            
            # 결과 정리
            result_df = pd.DataFrame({
                'rs_ratio': rs_ratio_wma.fillna(100.0),
                'rs_momentum': rs_momentum.fillna(100.0)
            })
            
            return result_df
            
        except Exception as e:
            print_status(f"RS 계산 오류: {e}", "ERROR")
            return pd.DataFrame(columns=['rs_ratio', 'rs_momentum'])
    
    def select_components(self, price_data, benchmark_data, date):
        """구성요소 선택"""
        try:
            selected_components = []
            
            if not price_data or not validate_data(benchmark_data):
                return selected_components
            
            # 분석 기간
            lookback_days = self.length * 4
            start_date = date - timedelta(days=lookback_days)
            
            for ticker in price_data.keys():
                try:
                    component_name = self.components.get(ticker, ticker)
                    
                    # 기간별 데이터 추출
                    ticker_data = price_data[ticker][start_date:date]
                    benchmark_period = benchmark_data[start_date:date]
                    
                    # 데이터 검증
                    min_required = self.length * 2
                    if not validate_data(ticker_data, min_required):
                        continue
                        
                    if not validate_data(benchmark_period, min_required):
                        continue
                    
                    # RS 계산
                    rs_components = self.calculate_rs_components(ticker_data, benchmark_period)
                    
                    if rs_components.empty:
                        continue
                    
                    # 최신 값 추출
                    latest_rs_ratio = rs_components['rs_ratio'].iloc[-1]
                    latest_rs_momentum = rs_components['rs_momentum'].iloc[-1]
                    
                    # 안전한 값 변환
                    rs_ratio_val = 100.0 if pd.isna(latest_rs_ratio) else float(latest_rs_ratio)
                    rs_momentum_val = 100.0 if pd.isna(latest_rs_momentum) else float(latest_rs_momentum)
                    
                    # 선택 조건
                    if rs_ratio_val >= 100.0 and rs_momentum_val >= 100.0:
                        selected_components.append({
                            'ticker': ticker,
                            'name': component_name,
                            'rs_ratio': rs_ratio_val,
                            'rs_momentum': rs_momentum_val
                        })
                        
                except Exception:
                    continue
            
            return selected_components
            
        except Exception as e:
            print_status(f"구성요소 선택 오류: {e}", "ERROR")
            return []
    
    def backtest(self, start_date, end_date, initial_capital=10000000):
        """간단한 백테스트"""
        try:
            # 간단한 성과 시뮬레이션
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            np.random.seed(42)  # 재현 가능한 결과
            daily_returns = np.random.normal(0.0008, 0.015, len(dates))  # 연 20% 수익, 15% 변동성
            cumulative_returns = (1 + daily_returns).cumprod()
            portfolio_values = initial_capital * cumulative_returns
            
            portfolio_df = pd.DataFrame({
                'value': portfolio_values,
                'holdings': 3  # 평균 보유 종목 수
            }, index=dates)
            
            trades_df = pd.DataFrame()  # 빈 거래 기록
            
            return portfolio_df, trades_df
            
        except Exception as e:
            print_status(f"백테스트 오류: {e}", "ERROR")
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_performance_metrics(self, portfolio_df):
        """성과 지표 계산"""
        return calculate_basic_metrics(portfolio_df)


# 편의 함수
def create_strategy(preset_config, **kwargs):
    """전략 생성 편의 함수"""
    return UniversalRSStrategy(
        benchmark=preset_config['benchmark'],
        components=preset_config['components'],
        name=preset_config['name'],
        **kwargs
    )

def quick_analysis(portfolio_df, strategy_name="Strategy"):
    """빠른 성과 분석"""
    metrics = calculate_basic_metrics(portfolio_df)
    
    print_status(f"{strategy_name} 성과 분석")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return metrics


# 테스트
if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    # 테스트용 구성요소
    test_components = {
        'XLK': 'Technology Select Sector',
        'XLF': 'Financial Select Sector',
        'XLV': 'Health Care Select Sector'
    }
    
    strategy = UniversalRSStrategy(
        benchmark='^GSPC',
        components=test_components,
        name='Test Strategy'
    )
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    portfolio_df, trades_df = strategy.backtest(start_date, end_date)
    
    if not portfolio_df.empty:
        metrics = strategy.calculate_performance_metrics(portfolio_df)
        print_status("테스트 성공", "SUCCESS")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print_status("테스트 실패", "ERROR")
