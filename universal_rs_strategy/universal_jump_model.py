import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Risk-free rate 유틸리티 import
try:
    from risk_free_rate_utils import RiskFreeRateManager
    HAS_RF_UTILS = True
except ImportError:
    print("Warning: risk_free_rate_utils.py가 없습니다. 기본 risk-free rate (2%) 사용")
    HAS_RF_UTILS = False

def safe_float_conversion(value, default=0.0):
    """안전한 float 변환 함수"""
    try:
        if pd.isna(value):
            return default
        elif isinstance(value, pd.Series):
            if len(value) > 0:
                valid_values = value.dropna()
                if len(valid_values) > 0:
                    val = valid_values.iloc[-1]
                    return float(val) if not pd.isna(val) else default
                else:
                    return default
            else:
                return default
        elif isinstance(value, (list, np.ndarray)):
            if len(value) > 0:
                return float(value[-1]) if not pd.isna(value[-1]) else default
            else:
                return default
        else:
            return float(value)
    except (ValueError, TypeError, IndexError):
        return default

class UniversalJumpModel:
    """
    범용 Jump Model with EWM Features - realtime_dashboard.py 기준 통합 버전
    """
    
    def __init__(self, benchmark_ticker, benchmark_name="Market", 
                 n_states=2, jump_penalty=50.0, use_paper_features_only=True,  # 기본값 True로 변경
                 training_cutoff_date=None, rf_ticker='^IRX', default_rf_rate=0.02):
        """
        Parameters:
        - benchmark_ticker: 벤치마크 지수 티커
        - benchmark_name: 벤치마크 이름
        - n_states: 상태 수 (기본값: 2 - Bull/Bear)
        - jump_penalty: 체제 전환에 대한 페널티 (기본값: 50.0)
        - use_paper_features_only: True면 논문의 정확한 3가지 특징만 사용 (기본값: True)
        - training_cutoff_date: 학습 데이터 마지막 날짜 (기본값: 2024-12-31)
        - rf_ticker: Risk-free rate 티커 (기본값: ^IRX)
        - default_rf_rate: 기본 risk-free rate (기본값: 2%)
        """
        self.benchmark_ticker = benchmark_ticker
        self.benchmark_name = benchmark_name
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.use_paper_features_only = use_paper_features_only
        self.rf_ticker = rf_ticker
        self.default_rf_rate = default_rf_rate
        
        # 기본 학습 마감일을 2024년 12월 31일로 설정 (realtime_dashboard 기준)
        if training_cutoff_date is None:
            self.training_cutoff_date = datetime(2024, 12, 31)
        else:
            self.training_cutoff_date = training_cutoff_date
        
        # Risk-free rate 관리자 초기화
        if HAS_RF_UTILS:
            self.rf_manager = RiskFreeRateManager(rf_ticker, default_rf_rate)
        else:
            self.rf_manager = None
        
        # 모델 파라미터 (realtime_dashboard 기준)
        self.cluster_centers = None
        self.scaler = StandardScaler()
        self.current_regime = None
        self.state_mapping = None
        self.is_trained = False
        
        # 최소 데이터 요구량 (realtime_dashboard 기준)
        self.min_data_length = 300  # realtime_dashboard에서 사용하는 값
        
        feature_type = "논문 정확한 3특징" if use_paper_features_only else "논문 기반 + 추가 특징"
        print(f"EWM Jump Model 초기화 (통합): {feature_type}")
        print(f"학습 마감일: {self.training_cutoff_date.strftime('%Y-%m-%d')}")
        print(f"Risk-Free Rate: {self.rf_ticker} (기본값: {self.default_rf_rate*100:.1f}%)")
    
    def download_benchmark_data(self, start_date, end_date):
        """벤치마크 데이터 다운로드 - realtime_dashboard 기준 강화된 버전"""
        try:
            print(f"{self.benchmark_name} 데이터 다운로드 중...")
            
            extended_start = start_date - timedelta(days=100)
            
            # realtime_dashboard.py와 동일한 timeout 설정
            data = yf.download(
                self.benchmark_ticker, 
                start=extended_start, 
                end=end_date, 
                progress=False,
                auto_adjust=True,
                timeout=30  # realtime_dashboard 기준 timeout
            )
            
            if data.empty:
                raise ValueError(f"{self.benchmark_name} 데이터를 가져올 수 없습니다.")
            
            data = data.dropna()
            
            if len(data) < self.min_data_length:
                print(f"경고: {self.benchmark_name} 데이터가 부족합니다 ({len(data)} < {self.min_data_length})")
                return None  # realtime_dashboard 기준으로 None 반환
            
            print(f"{self.benchmark_name} 데이터: {len(data)}일")
            return data
            
        except Exception as e:
            print(f"데이터 다운로드 실패: {e}")
            return None
    
    def _safe_download_risk_free_rate(self, start_date, end_date):
        """안전한 Risk-free rate 다운로드 - realtime_dashboard 기준 에러 처리"""
        try:
            if not HAS_RF_UTILS or not self.rf_manager:
                return None
            
            print(f"Risk-free rate 데이터 다운로드 중... ({self.rf_ticker})")
            
            # realtime_dashboard 기준 직접 yfinance 사용
            try:
                rf_raw = yf.download(
                    self.rf_ticker,
                    start=start_date - timedelta(days=30),
                    end=end_date + timedelta(days=1),
                    progress=False,
                    auto_adjust=True,
                    timeout=30  # realtime_dashboard 기준
                )
                
                if rf_raw.empty:
                    print(f"Warning: {self.rf_ticker} 데이터가 없습니다.")
                    return None
                
                # Close 가격 추출하고 Series로 변환
                if 'Close' in rf_raw.columns:
                    rf_series = rf_raw['Close']
                else:
                    rf_series = rf_raw.iloc[:, 0]
                
                # DataFrame이면 Series로 변환
                if isinstance(rf_series, pd.DataFrame):
                    rf_series = rf_series.iloc[:, 0]
                
                # NaN 값 처리
                rf_series = rf_series.fillna(method='ffill').fillna(method='bfill')
                
                # 백분율을 소수점으로 변환
                rf_series = rf_series / 100.0
                
                # 요청 기간으로 제한
                rf_series = rf_series[start_date:end_date]
                
                if rf_series.empty:
                    print(f"Warning: 요청 기간의 {self.rf_ticker} 데이터가 없습니다.")
                    return None
                
                print(f"✅ Risk-free rate 다운로드 성공: {len(rf_series)}개 (평균: {rf_series.mean()*100:.3f}%)")
                return rf_series
                
            except Exception as e:
                print(f"Risk-free rate 다운로드 실패: {e}")
                return None
                
        except Exception as e:
            print(f"Risk-free rate 처리 오류: {e}")
            return None
    
    def calculate_features(self, price_data):
        """특징 계산 - realtime_dashboard 기준 안정화 버전"""
        try:
            if price_data is None or price_data.empty:
                print(f"❌ 가격 데이터가 없음")
                return pd.DataFrame()
            
            # 1단계: 수익률 계산 (realtime_dashboard 기준 강화된 검증)
            try:
                if 'Close' not in price_data.columns:
                    print(f"❌ Close 컬럼이 없음")
                    return pd.DataFrame()
                
                close_prices = price_data['Close']
                
                # Close가 DataFrame인 경우 Series로 변환
                if isinstance(close_prices, pd.DataFrame):
                    if len(close_prices.columns) > 0:
                        close_prices = close_prices.iloc[:, 0]
                    else:
                        print(f"❌ Close DataFrame이 비어있음")
                        return pd.DataFrame()
                
                # pct_change 결과도 Series로 보장
                returns = close_prices.pct_change().dropna()
                
                if isinstance(returns, pd.DataFrame):
                    if len(returns.columns) > 0:
                        returns = returns.iloc[:, 0]
                    else:
                        print(f"❌ pct_change DataFrame이 비어있음")
                        return pd.DataFrame()
                
                # 최종 확인 (realtime_dashboard 기준)
                if not isinstance(returns, pd.Series):
                    print(f"❌ returns가 Series가 아님: {type(returns)}")
                    return pd.DataFrame()
                
                if len(returns) < self.min_data_length:
                    print(f"❌ 수익률 데이터 부족: {len(returns)} < {self.min_data_length}")
                    return pd.DataFrame()
                
                print(f"✅ 수익률 계산: {len(returns)}일 (Series)")
                
            except Exception as e:
                print(f"❌ 수익률 계산 오류: {e}")
                return pd.DataFrame()
            
            # 2단계: 동적 Risk-free rate 처리 (realtime_dashboard 기준)
            try:
                start_date = returns.index[0]
                end_date = returns.index[-1]
                
                rf_data = self._safe_download_risk_free_rate(start_date, end_date)
                
                if rf_data is not None and isinstance(rf_data, pd.Series) and len(rf_data) > 0:
                    print(f"    동적 RF 데이터 병합 처리...")
                    
                    # returns 인덱스를 기준으로 RF 데이터 정렬
                    aligned_rf_data = rf_data.reindex(returns.index, method='ffill')
                    aligned_rf_data = aligned_rf_data.fillna(self.default_rf_rate)
                    
                    # 일일 risk-free rate로 변환
                    daily_rf_rates = aligned_rf_data / 252
                    
                    print(f"    RF 병합 완료: 평균 {daily_rf_rates.mean()*252*100:.3f}%")
                    use_dynamic_rf = True
                    
                else:
                    # 기본값 사용
                    daily_rf_rates = pd.Series(
                        self.default_rf_rate / 252, 
                        index=returns.index,
                        name='rf_rate'
                    )
                    use_dynamic_rf = False
                    print(f"    고정 Risk-Free Rate 사용: {self.default_rf_rate*100:.1f}%")
                
                # 초과수익률 계산
                excess_returns = returns - daily_rf_rates
                
                if not isinstance(excess_returns, pd.Series):
                    print(f"❌ excess_returns가 Series가 아님: {type(excess_returns)}")
                    return pd.DataFrame()
                
                print(f"✅ 초과수익률 계산 완료: 평균 {excess_returns.mean()*252*100:.3f}%")
                
            except Exception as e:
                print(f"❌ Risk-free rate 처리 오류: {e}")
                # Fallback (realtime_dashboard 기준)
                daily_rf_rates = pd.Series(
                    self.default_rf_rate / 252, 
                    index=returns.index,
                    name='rf_rate'
                )
                excess_returns = returns - daily_rf_rates
                use_dynamic_rf = False
            
            # 3단계: 특징 계산 (realtime_dashboard 기준 - 주로 논문 특징 사용)
            try:
                if self.use_paper_features_only:
                    features_df = self._calculate_paper_features_unified(excess_returns, use_dynamic_rf)
                else:
                    features_df = self._calculate_enhanced_features_unified(excess_returns, returns, use_dynamic_rf)
                
                if features_df is None or features_df.empty:
                    print(f"❌ 특징 계산 결과가 비어있음")
                    return pd.DataFrame()
                
                print(f"✅ 특징 계산 완료: {len(features_df)}개")
                return features_df
                
            except Exception as e:
                print(f"❌ 특징 계산 오류: {e}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ 특징 계산 치명적 오류: {e}")
            return pd.DataFrame()
    
    def _calculate_paper_features_unified(self, excess_returns, use_dynamic_rf):
        """논문 특징 계산 - realtime_dashboard 기준 통합 버전"""
        try:
            print(f"    논문 특징 계산 시작... (동적 RF: {use_dynamic_rf})")
            
            # 입력 검증 (realtime_dashboard 기준)
            if not isinstance(excess_returns, pd.Series):
                print(f"    ❌ excess_returns가 Series가 아님: {type(excess_returns)}")
                return pd.DataFrame()
            
            # 하방 수익률 계산
            negative_excess_returns = excess_returns.where(excess_returns < 0, 0)
            
            # Feature 1: Downside Deviation (halflife=10) - realtime_dashboard 기준 안정화
            try:
                negative_squared = negative_excess_returns * negative_excess_returns
                
                ewm_dd_var_10 = negative_squared.ewm(
                    halflife=10, 
                    min_periods=20, 
                    adjust=False
                ).mean()
                
                # NaN 값 처리 강화 (realtime_dashboard 기준)
                ewm_dd_var_10 = ewm_dd_var_10.fillna(method='ffill').fillna(0)
                
                downside_deviation_10 = np.sqrt(ewm_dd_var_10.abs()) * np.sqrt(252)
                downside_deviation_10 = downside_deviation_10.fillna(0.1)  # 기본값 0.1
                
                print(f"    Feature 1 완료: 평균={downside_deviation_10.mean():.6f}")
                
            except Exception as e:
                print(f"    Feature 1 실패: {e}, 기본값 사용")
                downside_deviation_10 = pd.Series(0.1, index=excess_returns.index)
            
            # Feature 2: Sortino Ratio (halflife=20) - realtime_dashboard 기준
            try:
                ewm_mean_20 = excess_returns.ewm(
                    halflife=20, 
                    min_periods=40, 
                    adjust=False
                ).mean() * 252
                
                ewm_dd_var_20 = negative_squared.ewm(
                    halflife=20, 
                    min_periods=40, 
                    adjust=False
                ).mean()
                
                # NaN 처리 강화
                ewm_mean_20 = ewm_mean_20.fillna(method='ffill').fillna(0)
                ewm_dd_var_20 = ewm_dd_var_20.fillna(method='ffill').fillna(1e-8)
                
                ewm_dd_20 = np.sqrt(ewm_dd_var_20.abs()) * np.sqrt(252)
                
                # 0으로 나누기 방지 (realtime_dashboard 기준)
                sortino_ratio_20 = ewm_mean_20 / (ewm_dd_20 + 1e-8)
                sortino_ratio_20 = sortino_ratio_20.replace([np.inf, -np.inf], 1.0).fillna(1.0)
                
                print(f"    Feature 2 완료: 평균={sortino_ratio_20.mean():.6f}")
                
            except Exception as e:
                print(f"    Feature 2 실패: {e}, 기본값 사용")
                sortino_ratio_20 = pd.Series(1.0, index=excess_returns.index)
            
            # Feature 3: Sortino Ratio (halflife=60) - realtime_dashboard 기준
            try:
                ewm_mean_60 = excess_returns.ewm(
                    halflife=60, 
                    min_periods=120, 
                    adjust=False
                ).mean() * 252
                
                ewm_dd_var_60 = negative_squared.ewm(
                    halflife=60, 
                    min_periods=120, 
                    adjust=False
                ).mean()
                
                # NaN 처리 강화
                ewm_mean_60 = ewm_mean_60.fillna(method='ffill').fillna(0)
                ewm_dd_var_60 = ewm_dd_var_60.fillna(method='ffill').fillna(1e-8)
                
                ewm_dd_60 = np.sqrt(ewm_dd_var_60.abs()) * np.sqrt(252)
                
                # 0으로 나누기 방지
                sortino_ratio_60 = ewm_mean_60 / (ewm_dd_60 + 1e-8)
                sortino_ratio_60 = sortino_ratio_60.replace([np.inf, -np.inf], 1.0).fillna(1.0)
                
                print(f"    Feature 3 완료: 평균={sortino_ratio_60.mean():.6f}")
                
            except Exception as e:
                print(f"    Feature 3 실패: {e}, 기본값 사용")
                sortino_ratio_60 = pd.Series(1.0, index=excess_returns.index)
            
            # DataFrame 생성
            features_df = pd.DataFrame({
                'downside_deviation_10': downside_deviation_10,
                'sortino_ratio_20': sortino_ratio_20,
                'sortino_ratio_60': sortino_ratio_60
            }, index=excess_returns.index)
            
            # realtime_dashboard 기준 강화된 데이터 정리
            features_df = self._clean_features_dataframe_unified(features_df)
            
            print(f"    ✅ 논문 특징 계산 완료: {len(features_df)}개")
            return features_df
            
        except Exception as e:
            print(f"    ❌ 논문 특징 계산 실패: {e}")
            return pd.DataFrame()
    
    def _clean_features_dataframe_unified(self, features_df):
        """특징 데이터프레임 정리 - realtime_dashboard 기준 통합 버전"""
        try:
            if features_df is None or features_df.empty:
                return pd.DataFrame()
            
            print(f"    데이터 정리 시작: {features_df.shape}")
            
            # 1. 무한대 및 NaN 처리 (realtime_dashboard 기준)
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            
            # 2. NaN 값 처리
            features_df = features_df.fillna(method='ffill')
            features_df = features_df.fillna(method='bfill')
            
            # 3. 컬럼별 기본값 설정 (realtime_dashboard 기준)
            default_values = {
                'downside_deviation_10': 0.1,  # 10% 기본 변동성
                'sortino_ratio_20': 1.0,       # 중립적 Sortino ratio
                'sortino_ratio_60': 1.0        # 중립적 Sortino ratio
            }
            
            for col, default_val in default_values.items():
                if col in features_df.columns:
                    features_df[col] = features_df[col].fillna(default_val)
            
            # 4. 이상값 처리 (realtime_dashboard 기준 보수적 범위)
            for col in features_df.columns:
                if col.startswith('downside_deviation'):
                    # 하방변동성: 0~100% 범위로 제한
                    features_df[col] = features_df[col].clip(lower=0, upper=1.0)
                elif col.startswith('sortino_ratio'):
                    # Sortino ratio: -10~10 범위로 제한
                    features_df[col] = features_df[col].clip(lower=-10, upper=10)
            
            # 5. 초기 불안정한 값들 제거 (realtime_dashboard 기준)
            stable_start = max(120, len(features_df) // 4)
            if len(features_df) > stable_start:
                features_df = features_df.iloc[stable_start:].copy()
            
            # 6. 최종 유효성 검사 (realtime_dashboard 기준)
            if len(features_df) < 50:
                print(f"    ⚠️ 최종 데이터가 부족: {len(features_df)}")
                return pd.DataFrame()
            
            # 7. 최종 NaN 체크
            nan_counts = features_df.isna().sum()
            total_nans = nan_counts.sum()
            
            if total_nans > 0:
                print(f"    ⚠️ 남은 NaN: {total_nans}개")
                features_df = features_df.fillna(0)
            
            print(f"    ✅ 데이터 정리 완료: {features_df.shape}")
            
            # 8. 품질 확인
            for col in features_df.columns:
                avg_val = features_df[col].mean()
                std_val = features_df[col].std()
                print(f"      {col}: 평균={avg_val:.6f}, 표준편차={std_val:.6f}")
            
            return features_df
            
        except Exception as e:
            print(f"    ❌ 데이터 정리 실패: {e}")
            return pd.DataFrame()
    
    def _calculate_enhanced_features_unified(self, excess_returns, returns, use_dynamic_rf):
        """논문 기반 + 추가 특징들 - realtime_dashboard 기준"""
        try:
            # 먼저 논문 특징 계산
            features_df = self._calculate_paper_features_unified(excess_returns, use_dynamic_rf)
            
            if features_df.empty:
                return pd.DataFrame()
            
            print(f"    추가 특징 계산...")
            
            # 추가 특징들 (realtime_dashboard 기준 안전한 계산)
            try:
                # 변동성
                variance = (excess_returns * excess_returns).ewm(halflife=20, min_periods=20).mean()
                realized_vol = np.sqrt(variance.abs()) * np.sqrt(252)
                features_df['realized_vol'] = realized_vol.fillna(0.15)  # 15% 기본값
                
                # 평균 초과수익률
                mean_excess_return = excess_returns.ewm(halflife=20, min_periods=20).mean() * 252
                features_df['mean_excess_return_20'] = mean_excess_return.fillna(0.05)  # 5% 기본값
                
                # 왜도 (안전한 계산)
                skewness = excess_returns.rolling(window=20, min_periods=10).skew()
                features_df['skewness'] = skewness.fillna(0).clip(lower=-3, upper=3)
                
                # 상승일 비율
                up_days = (excess_returns > 0).astype(float)
                up_days_ratio = up_days.ewm(halflife=20, min_periods=10).mean()
                features_df['up_days_ratio'] = up_days_ratio.fillna(0.5)  # 50% 기본값
                
            except Exception as e:
                print(f"    추가 특징 계산 일부 실패: {e}")
            
            print(f"    추가 특징 계산 완료")
            return features_df
            
        except Exception as e:
            print(f"    확장 특징 계산 오류: {e}")
            return pd.DataFrame()
    
    def fit_jump_model(self, features_df):
        """Jump Model 학습 - realtime_dashboard 기준 에러 처리 강화"""
        try:
            if features_df.empty or len(features_df) < 50:
                print(f"학습용 특징 데이터 부족: {len(features_df)}")
                return None
            
            X = features_df.values
            
            # 무효한 값 확인 및 정리 (realtime_dashboard 기준)
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print("특징 데이터 정리 중...")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 데이터 스케일링
            try:
                X_scaled = self.scaler.fit_transform(X)
            except Exception as e:
                print(f"스케일링 실패: {e}")
                return None
            
            # 초기 클러스터링 (realtime_dashboard 기준 파라미터)
            try:
                kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10, max_iter=300)
                initial_states = kmeans.fit_predict(X_scaled)
                self.cluster_centers = kmeans.cluster_centers_
            except Exception as e:
                print(f"클러스터링 실패: {e}")
                return None
            
            # Jump penalty 적용 최적화
            try:
                optimized_states = self.optimize_with_jump_penalty(X_scaled, initial_states)
            except Exception as e:
                print(f"최적화 실패: {e}")
                optimized_states = initial_states
            
            # 상태별 특성 분석
            try:
                self.analyze_regimes(features_df, optimized_states)
            except Exception as e:
                print(f"체제 분석 실패: {e}")
            
            self.is_trained = True
            print("Jump Model 학습 완료")
            return optimized_states
            
        except Exception as e:
            print(f"모델 학습 중 오류: {e}")
            return None
    
    def optimize_with_jump_penalty(self, X, initial_states):
        """Jump penalty 최적화 - realtime_dashboard 기준"""
        try:
            n_samples = len(X)
            states = initial_states.copy()
            
            for iteration in range(10):
                converged = True
                
                for i in range(1, n_samples - 1):
                    current_state = states[i]
                    min_cost = float('inf')
                    best_state = current_state
                    
                    for new_state in range(self.n_states):
                        try:
                            cluster_cost = np.linalg.norm(X[i] - self.cluster_centers[new_state]) ** 2
                            
                            jump_cost = 0
                            if new_state != states[i-1]:
                                jump_cost += self.jump_penalty
                            if i < n_samples - 1 and new_state != states[i+1]:
                                jump_cost += self.jump_penalty
                            
                            total_cost = cluster_cost + jump_cost
                            
                            if total_cost < min_cost:
                                min_cost = total_cost
                                best_state = new_state
                        except:
                            continue
                    
                    if best_state != current_state:
                        states[i] = best_state
                        converged = False
                
                if converged:
                    break
            
            return states
            
        except Exception as e:
            print(f"최적화 오류: {e}")
            return initial_states
    
    def analyze_regimes(self, features_df, states):
        """체제별 특성 분석 - realtime_dashboard 기준"""
        try:
            regime_stats = {}
            
            for state in range(self.n_states):
                state_mask = (states == state)
                state_features = features_df[state_mask]
                
                if len(state_features) > 0:
                    avg_dd = state_features['downside_deviation_10'].mean()
                    avg_sr20 = state_features['sortino_ratio_20'].mean()
                    avg_sr60 = state_features['sortino_ratio_60'].mean()
                    
                    regime_stats[state] = {
                        'count': len(state_features),
                        'avg_downside_dev': avg_dd if not pd.isna(avg_dd) else 0.0,
                        'avg_sortino_20': avg_sr20 if not pd.isna(avg_sr20) else 0.0,
                        'avg_sortino_60': avg_sr60 if not pd.isna(avg_sr60) else 0.0
                    }
                else:
                    regime_stats[state] = {
                        'count': 0,
                        'avg_downside_dev': 0.0,
                        'avg_sortino_20': 0.0,
                        'avg_sortino_60': 0.0
                    }
            
            # Bear 상태 식별 (realtime_dashboard 기준)
            state_scores = {}
            for state in range(self.n_states):
                bear_score = (
                    regime_stats[state]['avg_downside_dev'] * 3 -
                    regime_stats[state]['avg_sortino_20'] * 2 -
                    regime_stats[state]['avg_sortino_60'] * 2
                )
                state_scores[state] = bear_score
            
            bear_state = max(state_scores.keys(), key=lambda x: state_scores[x])
            
            # 상태 매핑
            self.state_mapping = {}
            for state in range(self.n_states):
                if state == bear_state:
                    self.state_mapping[state] = 'BEAR'
                else:
                    self.state_mapping[state] = 'BULL'
            
            # 통계 출력
            print(f"\n=== {self.benchmark_name} EWM 체제별 특성 (통합) ===")
            feature_info = "논문 정확한 3특징" if self.use_paper_features_only else "논문 기반 + 추가"
            print(f"특징: {feature_info}")
            
            for state, stats in regime_stats.items():
                regime_type = self.state_mapping[state]
                if stats['count'] > 0:
                    print(f"\n{regime_type} 체제 (State {state}):")
                    print(f"  - 기간 비율: {stats['count'] / len(features_df) * 100:.1f}%")
                    print(f"  - 하방변동성 (hl=10): {stats['avg_downside_dev']*100:.1f}%")
                    print(f"  - Sortino Ratio (hl=20): {stats['avg_sortino_20']:.3f}")
                    print(f"  - Sortino Ratio (hl=60): {stats['avg_sortino_60']:.3f}")
            
            return regime_stats
            
        except Exception as e:
            print(f"체제 분석 오류: {e}")
            self.state_mapping = {0: 'BULL', 1: 'BEAR'}
            return {}
    
    def predict_regime(self, current_features):
        """현재 시장 체제 예측 - realtime_dashboard 기준 안정화"""
        if not self.is_trained or self.cluster_centers is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            if isinstance(current_features, pd.Series):
                X = current_features.values.reshape(1, -1)
            else:
                X = np.array(current_features).reshape(1, -1)
            
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            try:
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                print(f"예측 시 스케일링 오류: {e}")
                predicted_state = 0
                confidence = 0.5
                regime = self.state_mapping.get(predicted_state, 'BULL')
                return regime, confidence
            
            distances = []
            for center in self.cluster_centers:
                try:
                    dist = np.linalg.norm(X_scaled - center)
                    distances.append(float(dist))
                except:
                    distances.append(float('inf'))
            
            if not distances or all(d == float('inf') for d in distances):
                predicted_state = 0
                confidence = 0.5
            else:
                predicted_state = np.argmin(distances)
                
                # Jump penalty 고려 (realtime_dashboard 기준)
                if (self.current_regime is not None and 
                    predicted_state != self.current_regime and
                    len(distances) > self.current_regime):
                    
                    current_distance = distances[self.current_regime]
                    new_distance = distances[predicted_state]
                    
                    if (current_distance - new_distance) < self.jump_penalty / 100:
                        predicted_state = self.current_regime
                
                # 신뢰도 계산
                try:
                    min_dist = min(distances)
                    max_dist = max(distances)
                    if max_dist > min_dist and max_dist > 0:
                        confidence = 1 - (min_dist / max_dist)
                        confidence = max(0.0, min(1.0, confidence))
                    else:
                        confidence = 0.5
                except:
                    confidence = 0.5
            
            self.current_regime = predicted_state
            regime = self.state_mapping.get(predicted_state, 'BULL')
            
            return regime, confidence
            
        except Exception as e:
            print(f"체제 예측 중 오류: {e}")
            return 'BULL', 0.5
    
    def train_model_with_cutoff(self, start_date=None, end_date=None):
        """특정 기간의 데이터로만 모델 학습 - realtime_dashboard 기준"""
        if end_date is None:
            end_date = self.training_cutoff_date
        
        if start_date is None:
            start_date = end_date - timedelta(days=365*20)
        
        print(f"모델 학습: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        try:
            extended_start = start_date - timedelta(days=200)
            
            price_data = self.download_benchmark_data(extended_start, end_date)
            
            if price_data is None or price_data.empty:
                print(f"❌ {self.benchmark_name} 학습 데이터 다운로드 실패")
                return False
            
            features_df = self.calculate_features(price_data)
            
            if features_df is None or features_df.empty:
                print(f"❌ {self.benchmark_name} 학습용 특징 계산 실패")
                return False
            
            training_features = features_df[start_date:end_date]
            
            if len(training_features) < 50:
                print(f"❌ 학습 기간 데이터 부족: {len(training_features)} < 50")
                return False
            
            result = self.fit_jump_model(training_features)
            
            if result is None:
                print(f"❌ 모델 학습 실패")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 학습 오류: {e}")
            return False
    
    def get_current_regime_with_training_cutoff(self):
        """학습 마감일까지만 학습하고 현재 체제 예측 - realtime_dashboard 기준 통합"""
        try:
            if not self.is_trained:
                print(f"모델 학습 시작: {self.benchmark_name}")
                success = self.train_model_with_cutoff()
                if not success:
                    print(f"❌ {self.benchmark_name} 모델 학습 실패")
                    return None
            
            current_date = datetime.now()
            inference_start = self.training_cutoff_date - timedelta(days=200)
            
            print(f"추론 데이터 다운로드: {inference_start.strftime('%Y-%m-%d')} ~ {current_date.strftime('%Y-%m-%d')}")
            
            price_data = self.download_benchmark_data(inference_start, current_date)
            
            if price_data is None or price_data.empty:
                print(f"❌ {self.benchmark_name} 추론 데이터 다운로드 실패")
                return None
            
            features_df = self.calculate_features(price_data)
            
            if features_df is None or features_df.empty:
                print(f"❌ {self.benchmark_name} 특징 계산 실패")
                return None
            
            if len(features_df) == 0:
                print(f"❌ 특징 데이터가 비어있음")
                return None
            
            latest_features = features_df.iloc[-1]
            latest_date = features_df.index[-1]
            
            if latest_features.isna().all():
                print(f"❌ 최신 특징이 모두 NaN")
                return None
            
            print(f"✅ 최신 특징 추출: {latest_date.strftime('%Y-%m-%d')}")
            
            current_regime, confidence = self.predict_regime(latest_features)
            
            if current_regime is None:
                print(f"❌ 체제 예측 실패")
                return None
            
            is_out_of_sample = latest_date > self.training_cutoff_date
            
            # 안전한 confidence 변환 (realtime_dashboard 기준)
            safe_confidence = safe_float_conversion(confidence, 0.5)
            
            # RF 정보 추가 (realtime_dashboard 기준)
            try:
                if HAS_RF_UTILS and self.rf_manager:
                    rf_data = self._safe_download_risk_free_rate(
                        latest_date - timedelta(days=30), latest_date
                    )
                    if rf_data is not None and len(rf_data) > 0:
                        current_rf_rate = rf_data.iloc[-1] * 100
                        avg_rf_rate_30d = rf_data.mean() * 100
                    else:
                        current_rf_rate = self.default_rf_rate * 100
                        avg_rf_rate_30d = self.default_rf_rate * 100
                else:
                    current_rf_rate = self.default_rf_rate * 100
                    avg_rf_rate_30d = self.default_rf_rate * 100
            except:
                current_rf_rate = self.default_rf_rate * 100
                avg_rf_rate_30d = self.default_rf_rate * 100
            
            # realtime_dashboard 기준 분석 정보
            analysis_info = {
                'regime': str(current_regime),
                'confidence': safe_confidence,
                'date': latest_date,
                'features': {k: safe_float_conversion(v, 0.0) for k, v in latest_features.items()},
                'is_out_of_sample': bool(is_out_of_sample),
                'training_cutoff': self.training_cutoff_date.strftime('%Y-%m-%d'),
                'feature_type': "논문 정확한 3특징" if self.use_paper_features_only else "논문 기반 + 추가",
                'rf_ticker': self.rf_ticker,
                'dynamic_rf_used': HAS_RF_UTILS and self.rf_manager is not None,
                'current_rf_rate': current_rf_rate,
                'avg_rf_rate_30d': avg_rf_rate_30d
            }
            
            print(f"✅ 체제 분석 완료: {current_regime} (신뢰도: {safe_confidence:.2%})")
            return analysis_info
            
        except Exception as e:
            print(f"❌ 체제 분석 치명적 오류: {e}")
            return None
    
    def get_current_regime(self):
        """현재 시장 체제 확인 (기존 호환성 유지)"""
        return self.get_current_regime_with_training_cutoff()


# realtime_dashboard.py 기준 테스트 함수
def test_unified_jump_model():
    """통합 Jump Model 테스트 - realtime_dashboard 기준"""
    print("=== 통합 Jump Model 테스트 (realtime_dashboard 기준) ===")
    
    # SPY 테스트 (realtime_dashboard 기준 파라미터)
    print("\n1. SPY 테스트 (통합 버전)")
    spy_model = UniversalJumpModel(
        benchmark_ticker='SPY',
        benchmark_name='SPDR S&P 500 ETF',
        use_paper_features_only=True,  # realtime_dashboard 기준
        jump_penalty=50.0,  # realtime_dashboard 기준
        rf_ticker='^IRX'
    )
    
    spy_result = spy_model.get_current_regime_with_training_cutoff()
    if spy_result:
        print(f"\n✅ SPY 분석 성공!")
        print(f"체제: {spy_result['regime']} (신뢰도: {spy_result['confidence']:.2%})")
        print(f"동적 RF: {spy_result.get('dynamic_rf_used', False)}")
        print(f"현재 RF: {spy_result.get('current_rf_rate', 0):.3f}%")
        
        features = spy_result.get('features', {})
        if features:
            print(f"특징값:")
            for key, value in features.items():
                print(f"  {key}: {value:.6f}")
    else:
        print("❌ SPY 분석 실패")
    
    # QQQ 테스트 (realtime_dashboard 기준)
    print(f"\n2. QQQ 테스트 (통합 버전)")
    try:
        qqq_model = UniversalJumpModel(
            benchmark_ticker='QQQ',
            benchmark_name='Nasdaq 100',
            use_paper_features_only=True,  # realtime_dashboard 기준
            rf_ticker='^IRX'
        )
        
        result = qqq_model.get_current_regime_with_training_cutoff()
        if result:
            print(f"✅ QQQ: {result['regime']} (신뢰도: {result['confidence']:.2%})")
        else:
            print(f"❌ QQQ 분석 실패")
    except Exception as e:
        print(f"❌ QQQ 오류: {str(e)[:50]}...")

if __name__ == "__main__":
    test_unified_jump_model()
