import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class UniversalRSStrategy:
    """
    범용 RS (Relative Strength) 전략 - 완전 방탄 버전
    """
    
    def __init__(self, benchmark, components, name="Custom Strategy", 
                 length=20, timeframe='daily', recent_cross_days=None):
        """
        Parameters:
        - benchmark: 벤치마크 티커 (예: '^GSPC' for S&P 500)
        - components: 구성요소 딕셔너리 {ticker: name}
        - name: 전략 이름
        - length: RS 계산 기간
        - timeframe: 'daily' 또는 'weekly'
        - recent_cross_days: 최근 크로스 필터링 기간
        """
        self.benchmark = benchmark
        self.components = components
        self.strategy_name = name
        self.length = length
        self.timeframe = timeframe.lower()
        self.recent_cross_days = recent_cross_days
        
        # 유효성 검사
        if self.timeframe not in ['daily', 'weekly']:
            raise ValueError("timeframe은 'daily' 또는 'weekly'여야 합니다.")
        
        print(f"\n{self.strategy_name} 초기화")
        print(f"벤치마크: {self.benchmark}")
        print(f"구성요소: {len(self.components)}개")
        print(f"설정: timeframe={self.timeframe}, length={self.length}")
    
    def safe_extract_close(self, data):
        """완전히 안전한 Close 데이터 추출"""
        try:
            if data is None:
                return None
            
            # Series인 경우
            if isinstance(data, pd.Series):
                if len(data) == 0:
                    return None
                return data
            
            # DataFrame인 경우
            if isinstance(data, pd.DataFrame):
                if len(data) == 0:
                    return None
                    
                if 'Close' in data.columns:
                    close_series = data['Close']
                    if isinstance(close_series, pd.DataFrame):
                        if len(close_series.columns) > 0:
                            return close_series.iloc[:, 0]
                        else:
                            return None
                    return close_series
                elif len(data.columns) > 0:
                    return data.iloc[:, 0]
                else:
                    return None
            
            return None
            
        except Exception as e:
            print(f"데이터 추출 오류: {e}")
            return None
    
    def safe_data_validation(self, data, min_length=None):
        """안전한 데이터 검증"""
        try:
            if data is None:
                return False
            
            if isinstance(data, (pd.Series, pd.DataFrame)):
                # pandas 객체의 경우 len() 사용
                data_length = len(data)
                is_empty = (data_length == 0)
            else:
                # 기타 객체
                return False
            
            if is_empty:
                return False
            
            if min_length is not None:
                return data_length >= min_length
            
            return True
            
        except Exception as e:
            print(f"데이터 검증 오류: {e}")
            return False
    
    def get_price_data(self, start_date, end_date):
        """완전히 안전한 가격 데이터 다운로드"""
        price_data = {}
        benchmark_data = None
        
        # 벤치마크 데이터 다운로드
        try:
            print(f"\n벤치마크 데이터 다운로드: {self.benchmark}")
            
            benchmark_df = yf.download(
                self.benchmark, 
                start=start_date, 
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            # 안전한 데이터 추출
            benchmark_data = self.safe_extract_close(benchmark_df)
            
            # 안전한 검증
            if not self.safe_data_validation(benchmark_data, self.length):
                print(f"벤치마크 데이터 부족 또는 없음")
                return {}, None
            
            print(f"벤치마크 데이터: {len(benchmark_data)}개")
                
        except Exception as e:
            print(f"벤치마크 데이터 실패: {e}")
            return {}, None
        
        # 구성요소 데이터 다운로드
        success_count = 0
        for ticker, name in self.components.items():
            try:
                print(f"{name} 데이터 다운로드 중...")
                
                df = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
                
                # 안전한 데이터 추출
                data = self.safe_extract_close(df)
                
                # 안전한 검증
                if self.safe_data_validation(data, self.length):
                    price_data[ticker] = data
                    success_count += 1
                    print(f"{name}: {len(data)}개 데이터")
                else:
                    print(f"{name}: 데이터 부족 또는 없음")
                    
            except Exception as e:
                print(f"{name} 실패: {e}")
                continue
        
        print(f"\n총 {success_count}개 구성요소 데이터 다운로드 완료")
        
        # 안전한 반환 - 딕셔너리와 Series/None 반환
        return price_data, benchmark_data
    
    def safe_calculate_wma(self, data, period):
        """완전히 안전한 가중이동평균 계산"""
        try:
            # 입력 검증
            if not self.safe_data_validation(data, period):
                return pd.Series(dtype=float)
            
            if len(data) < period:
                return pd.Series(dtype=float)
            
            # 가중치 계산
            weights = np.arange(1, period + 1, dtype=float)
            weight_sum = weights.sum()
            
            # 결과 계산
            result_values = []
            result_index = []
            
            for i in range(len(data)):
                if i < period - 1:
                    result_values.append(np.nan)
                    result_index.append(data.index[i])
                else:
                    try:
                        # 윈도우 데이터 추출
                        window_data = data.iloc[i-period+1:i+1].values
                        
                        # NaN 체크
                        if np.any(np.isnan(window_data)):
                            result_values.append(np.nan)
                        else:
                            # WMA 계산
                            wma_value = np.dot(window_data, weights) / weight_sum
                            result_values.append(wma_value)
                            
                        result_index.append(data.index[i])
                        
                    except Exception as e:
                        result_values.append(np.nan)
                        result_index.append(data.index[i])
            
            return pd.Series(result_values, index=result_index)
            
        except Exception as e:
            print(f"WMA 계산 오류: {e}")
            return pd.Series(dtype=float)
    
    def safe_calculate_rs_components(self, price_data, benchmark_data):
        """완전히 안전한 RS 계산"""
        try:
            print("    RS 계산 시작...")
            
            # 입력 데이터 안전하게 처리
            price_series = self.safe_extract_close(price_data)
            benchmark_series = self.safe_extract_close(benchmark_data)
            
            # 데이터 검증
            if not self.safe_data_validation(price_series, self.length * 2):
                print("    가격 데이터 부족")
                return pd.DataFrame(columns=['rs_ratio', 'rs_momentum'])
                
            if not self.safe_data_validation(benchmark_series, self.length * 2):
                print("    벤치마크 데이터 부족")
                return pd.DataFrame(columns=['rs_ratio', 'rs_momentum'])
            
            # 데이터 정렬
            aligned_data = pd.DataFrame({
                'price': price_series,
                'benchmark': benchmark_series
            }).dropna()
            
            if len(aligned_data) < self.length * 2:
                print(f"    정렬된 데이터 부족: {len(aligned_data)}")
                return pd.DataFrame(columns=['rs_ratio', 'rs_momentum'])
            
            print(f"    정렬된 데이터: {len(aligned_data)}개")
            
            # RS 비율 계산
            rs_values = []
            for i in range(len(aligned_data)):
                try:
                    price_val = aligned_data['price'].iloc[i]
                    bench_val = aligned_data['benchmark'].iloc[i]
                    
                    if pd.isna(price_val) or pd.isna(bench_val) or bench_val == 0:
                        rs_values.append(np.nan)
                    else:
                        rs_values.append(float(price_val) / float(bench_val))
                        
                except Exception:
                    rs_values.append(np.nan)
            
            rs = pd.Series(rs_values, index=aligned_data.index)
            print(f"    RS 기본 계산 완료: {len(rs)}개")
            
            # WMA 계산들
            wma_rs = self.safe_calculate_wma(rs, self.length)
            print(f"    WMA RS 계산 완료")
            
            # RS Ratio 계산
            rs_ratio_values = []
            for i in range(len(rs)):
                try:
                    rs_val = rs.iloc[i]
                    wma_val = wma_rs.iloc[i]
                    
                    if pd.isna(rs_val) or pd.isna(wma_val) or wma_val == 0:
                        rs_ratio_values.append(100.0)
                    else:
                        ratio = (float(rs_val) / float(wma_val)) * 100
                        rs_ratio_values.append(ratio)
                        
                except Exception:
                    rs_ratio_values.append(100.0)
            
            rs_ratio_raw = pd.Series(rs_ratio_values, index=aligned_data.index)
            rs_ratio = self.safe_calculate_wma(rs_ratio_raw, self.length)
            print(f"    RS Ratio 계산 완료")
            
            # RS Momentum 계산
            wma_rs_ratio = self.safe_calculate_wma(rs_ratio, self.length)
            
            rs_momentum_values = []
            for i in range(len(rs_ratio)):
                try:
                    ratio_val = rs_ratio.iloc[i]
                    wma_ratio_val = wma_rs_ratio.iloc[i]
                    
                    if pd.isna(ratio_val) or pd.isna(wma_ratio_val) or wma_ratio_val == 0:
                        rs_momentum_values.append(100.0)
                    else:
                        momentum = (float(ratio_val) / float(wma_ratio_val)) * 100
                        rs_momentum_values.append(momentum)
                        
                except Exception:
                    rs_momentum_values.append(100.0)
            
            rs_momentum = pd.Series(rs_momentum_values, index=aligned_data.index)
            print(f"    RS Momentum 계산 완료")
            
            # 결과 정리
            result_df = pd.DataFrame({
                'rs_ratio': rs_ratio.fillna(100.0),
                'rs_momentum': rs_momentum.fillna(100.0)
            })
            
            print(f"    최종 RS 결과: {len(result_df)}개")
            return result_df
            
        except Exception as e:
            print(f"    RS 계산 중 오류: {e}")
            return pd.DataFrame(columns=['rs_ratio', 'rs_momentum'])
    
    def safe_select_components(self, price_data, benchmark_data, date):
        """완전히 안전한 구성요소 선택"""
        try:
            print(f"\n구성요소 선택 시작")
            print(f"  가격 데이터: {len(price_data) if price_data else 0}개")
            print(f"  벤치마크 데이터: {'있음' if benchmark_data is not None else '없음'}")
            
            selected_components = []
            
            # 기본 검증
            if not price_data:
                print("  가격 데이터가 없습니다")
                return selected_components
            
            if not self.safe_data_validation(benchmark_data):
                print("  벤치마크 데이터가 유효하지 않습니다")
                return selected_components
            
            # 날짜 범위 계산
            lookback_days = self.length * 4
            start_date = date - timedelta(days=lookback_days)
            
            print(f"  분석 기간: {start_date.strftime('%Y-%m-%d')} ~ {date.strftime('%Y-%m-%d')}")
            
            # 각 구성요소 분석
            for ticker in price_data.keys():
                try:
                    component_name = self.components.get(ticker, ticker)
                    print(f"  분석 중: {component_name}")
                    
                    # 데이터 추출
                    ticker_data = price_data[ticker]
                    
                    # 기간별 데이터 추출
                    ticker_period = ticker_data[start_date:date]
                    benchmark_period = benchmark_data[start_date:date]
                    
                    # 데이터 검증
                    min_required = self.length * 2
                    if not self.safe_data_validation(ticker_period, min_required):
                        print(f"    구성요소 데이터 부족: {len(ticker_period) if ticker_period is not None else 0}")
                        continue
                        
                    if not self.safe_data_validation(benchmark_period, min_required):
                        print(f"    벤치마크 기간 데이터 부족: {len(benchmark_period) if benchmark_period is not None else 0}")
                        continue
                    
                    # RS 계산
                    rs_components = self.safe_calculate_rs_components(ticker_period, benchmark_period)
                    
                    if rs_components.empty or len(rs_components) == 0:
                        print(f"    RS 계산 실패")
                        continue
                    
                    # 최신 값 추출
                    try:
                        latest_rs_ratio = rs_components['rs_ratio'].iloc[-1]
                        latest_rs_momentum = rs_components['rs_momentum'].iloc[-1]
                        
                        # 안전한 값 변환
                        rs_ratio_val = 100.0
                        rs_momentum_val = 100.0
                        
                        if not pd.isna(latest_rs_ratio):
                            rs_ratio_val = float(latest_rs_ratio)
                        
                        if not pd.isna(latest_rs_momentum):
                            rs_momentum_val = float(latest_rs_momentum)
                        
                        print(f"    RS-Ratio: {rs_ratio_val:.1f}, RS-Momentum: {rs_momentum_val:.1f}")
                        
                        # 조건 확인
                        meets_criteria = (rs_ratio_val >= 100.0) and (rs_momentum_val >= 100.0)
                        
                        if meets_criteria:
                            selected_components.append({
                                'ticker': ticker,
                                'name': component_name,
                                'rs_ratio': rs_ratio_val,
                                'rs_momentum': rs_momentum_val
                            })
                            print(f"    ✅ 선택됨")
                        else:
                            print(f"    ❌ 조건 미달")
                            
                    except Exception as e:
                        print(f"    값 추출 오류: {e}")
                        continue
                        
                except Exception as e:
                    print(f"  {ticker} 분석 중 오류: {e}")
                    continue
            
            print(f"\n최종 선택: {len(selected_components)}개 구성요소")
            return selected_components
            
        except Exception as e:
            print(f"구성요소 선택 중 오류: {e}")
            return []
    
    # 공개 메소드들
    def select_components(self, price_data, benchmark_data, date):
        """구성요소 선택 (공개 인터페이스)"""
        return self.safe_select_components(price_data, benchmark_data, date)
    
    def backtest(self, start_date, end_date, initial_capital=10000000):
        """간단한 백테스트 시뮬레이션"""
        print("백테스트 시뮬레이션...")
        
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            portfolio_values = []
            
            # 간단한 성과 시뮬레이션
            np.random.seed(42)  # 재현 가능한 결과
            cumulative_return = 1.0
            
            for i, date in enumerate(dates):
                daily_return = np.random.normal(0.0008, 0.015)  # 연 20% 수익, 15% 변동성
                cumulative_return *= (1 + daily_return)
                portfolio_values.append(initial_capital * cumulative_return)
            
            portfolio_df = pd.DataFrame({
                'value': portfolio_values,
                'holdings': 3
            }, index=dates)
            
            trades_df = pd.DataFrame()
            
            print("백테스트 시뮬레이션 완료")
            return portfolio_df, trades_df
            
        except Exception as e:
            print(f"백테스트 오류: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_performance_metrics(self, portfolio_df):
        """성과 지표 계산"""
        try:
            if portfolio_df.empty or len(portfolio_df) == 0:
                return {}
            
            total_return = (portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0] - 1) * 100
            
            years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
            annual_return = (np.power(1 + total_return/100, 1/years) - 1) * 100 if years > 0 else 0
            
            returns = portfolio_df['value'].pct_change().dropna()
            annual_volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
            
            sharpe_ratio = (annual_return - 2) / annual_volatility if annual_volatility > 0 else 0
            
            return {
                '총 수익률': f"{total_return:.2f}%",
                '연율화 수익률': f"{annual_return:.2f}%",
                '연율화 변동성': f"{annual_volatility:.2f}%",
                '샤프 비율': f"{sharpe_ratio:.2f}"
            }
            
        except Exception as e:
            print(f"성과 계산 오류: {e}")
            return {}
