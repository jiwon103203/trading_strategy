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
    """안전한 float 변환 함수 - 강화된 버전"""
    try:
        if pd.isna(value):
            return default
        elif isinstance(value, pd.Series):
            if len(value) > 0:
                # Series의 마지막 유효한 값 추출
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

def safe_mean(series, default=0.0):
    """안전한 평균 계산"""
    try:
        if isinstance(series, pd.Series) and len(series) > 0:
            valid_series = series.dropna()
            if len(valid_series) > 0:
                result = valid_series.mean()
                return safe_float_conversion(result, default)
            else:
                return default
        else:
            return default
    except:
        return default

def safe_std(series, default=0.0):
    """안전한 표준편차 계산"""
    try:
        if isinstance(series, pd.Series) and len(series) > 1:
            valid_series = series.dropna()
            if len(valid_series) > 1:
                result = valid_series.std()
                return safe_float_conversion(result, default)
            else:
                return default
        else:
            return default
    except:
        return default

def safe_ewm_calculation(series, halflife, min_periods=None):
    """안전한 EWM 계산"""
    try:
        if series is None or len(series) == 0:
            return pd.Series(dtype=float)
        
        # 최소 기간 설정 (halflife의 2배)
        if min_periods is None:
            min_periods = max(int(halflife * 2), 10)
        
        if len(series) < min_periods:
            return pd.Series(dtype=float, index=series.index)
        
        # NaN 처리
        clean_series = series.fillna(0.0)
        
        # EWM 계산
        ewm_result = clean_series.ewm(halflife=halflife, adjust=False, min_periods=min_periods).mean()
        
        # 무한대 및 NaN 처리
        ewm_result = ewm_result.replace([np.inf, -np.inf], np.nan)
        ewm_result = ewm_result.fillna(method='ffill').fillna(0.0)
        
        return ewm_result
        
    except Exception as e:
        print(f"EWM 계산 오류 (halflife={halflife}): {e}")
        return pd.Series(dtype=float, index=series.index if series is not None else [])

class UniversalJumpModel:
    """
    범용 Jump Model with EWM Features (논문 Table 2 기준) - 안정화된 버전
    """
    
    def __init__(self, benchmark_ticker, benchmark_name="Market", 
                 n_states=2, jump_penalty=50.0, use_paper_features_only=False,
                 training_cutoff_date=None, rf_ticker='^IRX', default_rf_rate=0.02):
        """
        Parameters:
        - benchmark_ticker: 벤치마크 지수 티커
        - benchmark_name: 벤치마크 이름
        - n_states: 상태 수 (기본값: 2 - Bull/Bear)
        - jump_penalty: 체제 전환에 대한 페널티
        - use_paper_features_only: True면 논문의 정확한 3가지 특징만 사용
        - training_cutoff_date: 학습 데이터 마지막 날짜
        - rf_ticker: Risk-free rate 티커
        - default_rf_rate: 기본 risk-free rate
        """
        self.benchmark_ticker = benchmark_ticker
        self.benchmark_name = benchmark_name
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.use_paper_features_only = use_paper_features_only
        self.rf_ticker = rf_ticker
        self.default_rf_rate = default_rf_rate
        
        # 기본 학습 마감일을 2024년 12월 31일로 설정
        if training_cutoff_date is None:
            self.training_cutoff_date = datetime(2024, 12, 31)
        else:
            self.training_cutoff_date = training_cutoff_date
        
        # Risk-free rate 관리자 초기화
        if HAS_RF_UTILS:
            self.rf_manager = RiskFreeRateManager(rf_ticker, default_rf_rate)
        else:
            self.rf_manager = None
        
        # 모델 파라미터
        self.cluster_centers = None
        self.scaler = StandardScaler()
        self.current_regime = None
        self.state_mapping = None
        self.is_trained = False
        
        # EWM 계산을 위한 최소 데이터 요구량 증가
        self.min_data_length = 200  # 기존 60 → 200
        
        feature_type = "논문 정확한 3특징" if use_paper_features_only else "논문 기반 + 추가 특징"
        print(f"EWM Jump Model 초기화: {feature_type}")
        print(f"학습 마감일: {self.training_cutoff_date.strftime('%Y-%m-%d')}")
        print(f"Risk-Free Rate: {self.rf_ticker} (기본값: {self.default_rf_rate*100:.1f}%)")
        print(f"최소 데이터 요구량: {self.min_data_length}일")
    
    def download_benchmark_data(self, start_date, end_date):
        """벤치마크 데이터 다운로드 - 안정화된 버전"""
        try:
            print(f"{self.benchmark_name} 데이터 다운로드 중...")
            
            # 더 많은 데이터 다운로드 (EWM을 위해)
            extended_start = start_date - timedelta(days=100)
            
            data = yf.download(
                self.benchmark_ticker, 
                start=extended_start, 
                end=end_date, 
                progress=False,
                auto_adjust=True,
                timeout=30  # 타임아웃 추가
            )
            
            if data.empty:
                raise ValueError(f"{self.benchmark_name} 데이터를 가져올 수 없습니다.")
            
            # 데이터 품질 검사
            data = data.dropna()
            
            if len(data) < self.min_data_length:
                print(f"경고: {self.benchmark_name} 데이터가 부족합니다 ({len(data)} < {self.min_data_length})")
            
            print(f"{self.benchmark_name} 데이터: {len(data)}일")
            return data
            
        except Exception as e:
            print(f"데이터 다운로드 실패: {e}")
            return None
    
    def calculate_features(self, price_data):
        """특징 계산 - 범용적 오류 처리 강화"""
        try:
            if price_data is None or price_data.empty:
                print(f"❌ 가격 데이터가 없음")
                return pd.DataFrame()
            
            # 1단계: 수익률 계산
            try:
                if 'Close' not in price_data.columns:
                    print(f"❌ Close 컬럼이 없음")
                    return pd.DataFrame()
                
                returns = price_data['Close'].pct_change().dropna()
                
                if len(returns) < self.min_data_length:
                    print(f"❌ 수익률 데이터 부족: {len(returns)} < {self.min_data_length}")
                    return pd.DataFrame()
                
                print(f"✅ 수익률 계산: {len(returns)}일")
                
            except Exception as e:
                print(f"❌ 수익률 계산 오류: {e}")
                return pd.DataFrame()
            
            # 2단계: Risk-free rate 처리
            try:
                rf_data = None
                if HAS_RF_UTILS and self.rf_manager:
                    start_date = returns.index[0]
                    end_date = returns.index[-1]
                    rf_data = self.rf_manager.download_risk_free_rate(start_date, end_date)
                    
                    if rf_data is not None and len(rf_data) > 0:
                        print(f"✅ Risk-free rate 데이터: {len(rf_data)}개")
                    else:
                        print(f"⚠️ Risk-free rate 데이터 없음 - 기본값 사용")
                        rf_data = None
                
                # Daily risk-free rate 계산
                if rf_data is not None:
                    try:
                        daily_rf_rates = rf_data.reindex(returns.index, method='ffill').fillna(self.default_rf_rate)
                        daily_rf_rates = daily_rf_rates / 252
                        print(f"✅ 동적 Risk-Free Rate 사용")
                    except Exception as e:
                        print(f"⚠️ RF 데이터 정렬 실패: {e} - 기본값 사용")
                        daily_rf_rates = pd.Series(self.default_rf_rate / 252, index=returns.index)
                else:
                    daily_rf_rates = pd.Series(self.default_rf_rate / 252, index=returns.index)
                    print(f"✅ 고정 Risk-Free Rate 사용")
                
                # 초과수익률 계산
                excess_returns = returns - daily_rf_rates
                
            except Exception as e:
                print(f"❌ Risk-free rate 처리 오류: {e}")
                excess_returns = returns - self.default_rf_rate / 252
            
            # 3단계: 특징 계산
            try:
                if self.use_paper_features_only:
                    features_df = self._calculate_paper_features_only_robust(excess_returns, rf_data)
                else:
                    features_df = self._calculate_enhanced_features_robust(excess_returns, returns, rf_data)
                
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
    
    def _calculate_paper_features_only_robust(self, excess_returns, rf_data):
        """논문 특징 계산 - EWM 차원 문제 해결"""
        try:
            print(f"    특징 계산 시작... (데이터: {len(excess_returns)}개)")
            
            # 하방 수익률 계산
            negative_excess_returns = excess_returns.where(excess_returns < 0, 0)
            
            print(f"    하방 수익률 계산 완료")
            
            # Feature 1: Downside Deviation (halflife=10) - 차원 문제 해결
            try:
                print(f"    Feature 1 계산 시작...")
                
                # 제곱 계산 (1차원 유지)
                negative_squared = negative_excess_returns ** 2
                print(f"    negative_squared shape: {negative_squared.shape if hasattr(negative_squared, 'shape') else 'Series'}")
                
                # EWM 계산
                ewm_dd_var_10 = self._robust_ewm_calculation_fixed(negative_squared, halflife=10, min_periods=20)
                
                if ewm_dd_var_10 is None or ewm_dd_var_10.empty:
                    print(f"    ❌ Feature 1 EWM 계산 실패")
                    return pd.DataFrame()
                
                print(f"    EWM 결과 shape: {ewm_dd_var_10.shape if hasattr(ewm_dd_var_10, 'shape') else 'Series'}")
                
                # 제곱근 계산 (연율화)
                downside_deviation_10 = np.sqrt(ewm_dd_var_10.abs()) * np.sqrt(252)
                print(f"    Feature 1 계산 완료")
                
            except Exception as e:
                print(f"    ❌ Feature 1 계산 오류: {e}")
                print(f"    오류 상세: {type(e).__name__}")
                return pd.DataFrame()
            
            # Feature 2: Sortino Ratio (halflife=20) - 차원 문제 해결
            try:
                print(f"    Feature 2 계산 시작...")
                
                # 평균 초과수익률 계산
                ewm_mean_20 = self._robust_ewm_calculation_fixed(excess_returns, halflife=20, min_periods=40)
                if ewm_mean_20 is None or ewm_mean_20.empty:
                    print(f"    ❌ Feature 2 평균 계산 실패")
                    return pd.DataFrame()
                
                ewm_mean_20_annual = ewm_mean_20 * 252
                
                # 하방 분산 계산
                ewm_dd_var_20 = self._robust_ewm_calculation_fixed(negative_squared, halflife=20, min_periods=40)
                if ewm_dd_var_20 is None or ewm_dd_var_20.empty:
                    print(f"    ❌ Feature 2 하방분산 계산 실패")
                    return pd.DataFrame()
                
                # 하방 표준편차 계산
                ewm_dd_20 = np.sqrt(ewm_dd_var_20.abs()) * np.sqrt(252)
                
                # Sortino ratio 계산 (0으로 나누기 방지)
                denominator_20 = ewm_dd_20 + 1e-8
                sortino_ratio_20 = ewm_mean_20_annual / denominator_20
                
                print(f"    Feature 2 계산 완료")
                
            except Exception as e:
                print(f"    ❌ Feature 2 계산 오류: {e}")
                return pd.DataFrame()
            
            # Feature 3: Sortino Ratio (halflife=60) - 차원 문제 해결
            try:
                print(f"    Feature 3 계산 시작...")
                
                # 평균 초과수익률 계산
                ewm_mean_60 = self._robust_ewm_calculation_fixed(excess_returns, halflife=60, min_periods=120)
                if ewm_mean_60 is None or ewm_mean_60.empty:
                    print(f"    ❌ Feature 3 평균 계산 실패")
                    return pd.DataFrame()
                
                ewm_mean_60_annual = ewm_mean_60 * 252
                
                # 하방 분산 계산
                ewm_dd_var_60 = self._robust_ewm_calculation_fixed(negative_squared, halflife=60, min_periods=120)
                if ewm_dd_var_60 is None or ewm_dd_var_60.empty:
                    print(f"    ❌ Feature 3 하방분산 계산 실패")
                    return pd.DataFrame()
                
                # 하방 표준편차 계산
                ewm_dd_60 = np.sqrt(ewm_dd_var_60.abs()) * np.sqrt(252)
                
                # Sortino ratio 계산 (0으로 나누기 방지)
                denominator_60 = ewm_dd_60 + 1e-8
                sortino_ratio_60 = ewm_mean_60_annual / denominator_60
                
                print(f"    Feature 3 계산 완료")
                
            except Exception as e:
                print(f"    ❌ Feature 3 계산 오류: {e}")
                return pd.DataFrame()
            
            # DataFrame 생성 - 차원 검증 강화
            try:
                print(f"    DataFrame 생성 시작...")
                
                # 각 특징의 차원 확인
                print(f"    downside_deviation_10 type: {type(downside_deviation_10)}")
                print(f"    sortino_ratio_20 type: {type(sortino_ratio_20)}")
                print(f"    sortino_ratio_60 type: {type(sortino_ratio_60)}")
                
                # 인덱스 확인
                common_index = excess_returns.index
                print(f"    공통 인덱스 길이: {len(common_index)}")
                
                # Series 검증 및 정렬
                dd_10_series = self._ensure_series_aligned(downside_deviation_10, common_index, 'downside_deviation_10')
                sr_20_series = self._ensure_series_aligned(sortino_ratio_20, common_index, 'sortino_ratio_20')
                sr_60_series = self._ensure_series_aligned(sortino_ratio_60, common_index, 'sortino_ratio_60')
                
                if dd_10_series is None or sr_20_series is None or sr_60_series is None:
                    print(f"    ❌ Series 정렬 실패")
                    return pd.DataFrame()
                
                # DataFrame 생성
                features_df = pd.DataFrame({
                    'downside_deviation_10': dd_10_series,
                    'sortino_ratio_20': sr_20_series,
                    'sortino_ratio_60': sr_60_series
                }, index=common_index)
                
                print(f"    DataFrame 생성 완료: {features_df.shape}")
                
            except Exception as e:
                print(f"    ❌ DataFrame 생성 오류: {e}")
                print(f"    오류 상세: {type(e).__name__}")
                return pd.DataFrame()
            
            # 데이터 정리
            try:
                print(f"    데이터 정리 시작...")
                features_df = self._clean_features_dataframe(features_df)
                
                if features_df.empty:
                    print(f"    ❌ 데이터 정리 후 빈 DataFrame")
                    return pd.DataFrame()
                
                print(f"    ✅ 논문 특징 계산 완료: {len(features_df)}개")
                return features_df
                
            except Exception as e:
                print(f"    ❌ 데이터 정리 오류: {e}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"    ❌ 논문 특징 계산 치명적 오류: {e}")
            return pd.DataFrame()
    
    def _robust_ewm_calculation_fixed(self, series, halflife, min_periods=None):
        """EWM 계산 - 차원 문제 완전 해결"""
        try:
            if series is None:
                print(f"        EWM 입력 series가 None")
                return pd.Series(dtype=float)
            
            # 입력 타입 확인
            print(f"        EWM 입력 타입: {type(series)}")
            
            if hasattr(series, 'shape'):
                print(f"        EWM 입력 shape: {series.shape}")
                
                # 2차원 배열이면 1차원으로 변환
                if len(series.shape) > 1:
                    print(f"        ⚠️ 2차원 배열 감지, 1차원으로 변환")
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]  # 첫 번째 컬럼 사용
                    elif isinstance(series, np.ndarray):
                        series = pd.Series(series.flatten())
                    else:
                        print(f"        ❌ 알 수 없는 2차원 타입: {type(series)}")
                        return pd.Series(dtype=float)
            
            # pandas Series인지 확인
            if not isinstance(series, pd.Series):
                print(f"        Series가 아닌 타입, 변환 시도: {type(series)}")
                try:
                    series = pd.Series(series)
                except Exception as e:
                    print(f"        ❌ Series 변환 실패: {e}")
                    return pd.Series(dtype=float)
            
            if len(series) == 0:
                print(f"        빈 Series")
                return pd.Series(dtype=float)
            
            print(f"        EWM 계산 시작: 길이={len(series)}, halflife={halflife}")
            
            # 최소 기간 설정
            if min_periods is None:
                min_periods = max(int(halflife * 2), 10)
            
            if len(series) < min_periods:
                print(f"        데이터 부족: {len(series)} < {min_periods}")
                return pd.Series(dtype=float, index=series.index)
            
            # 데이터 전처리
            clean_series = series.copy()
            
            # 무한대 제거
            clean_series = clean_series.replace([np.inf, -np.inf], np.nan)
            
            # NaN 처리
            clean_series = clean_series.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
            
            print(f"        데이터 전처리 완료")
            
            # EWM 계산
            try:
                ewm_result = clean_series.ewm(
                    halflife=halflife, 
                    adjust=False, 
                    min_periods=min_periods
                ).mean()
                
                print(f"        EWM 계산 성공: {type(ewm_result)}")
                
                # 결과 검증
                if hasattr(ewm_result, 'shape'):
                    print(f"        EWM 결과 shape: {ewm_result.shape}")
                    
                    # 2차원 결과면 1차원으로 변환
                    if len(ewm_result.shape) > 1:
                        print(f"        ⚠️ EWM 결과가 2차원, 1차원으로 변환")
                        if isinstance(ewm_result, pd.DataFrame):
                            ewm_result = ewm_result.iloc[:, 0]
                        else:
                            ewm_result = pd.Series(ewm_result.flatten(), index=series.index)
                
                # 결과 정리
                ewm_result = ewm_result.replace([np.inf, -np.inf], np.nan)
                ewm_result = ewm_result.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
                
                print(f"        EWM 계산 완료")
                return ewm_result
                
            except Exception as e:
                print(f"        EWM 계산 실패: {e}")
                print(f"        오류 타입: {type(e).__name__}")
                
                # 실패시 rolling mean으로 대체
                try:
                    print(f"        Rolling mean으로 대체 시도...")
                    window = max(int(halflife * 2), 5)
                    rolling_result = clean_series.rolling(window=window, min_periods=min_periods//2).mean()
                    rolling_result = rolling_result.fillna(0.0)
                    print(f"        Rolling mean 성공")
                    return rolling_result
                except Exception as e2:
                    print(f"        Rolling mean도 실패: {e2}")
                    return pd.Series(dtype=float, index=series.index)
            
        except Exception as e:
            print(f"        EWM 전처리 치명적 실패: {e}")
            if hasattr(series, 'index'):
                return pd.Series(dtype=float, index=series.index)
            else:
                return pd.Series(dtype=float)
    
    def _ensure_series_aligned(self, data, target_index, feature_name):
        """Series 정렬 및 차원 검증"""
        try:
            print(f"        {feature_name} 정렬 시작...")
            
            if data is None:
                print(f"        {feature_name} 데이터가 None")
                return None
            
            # 타입 확인
            print(f"        {feature_name} 타입: {type(data)}")
            
            if hasattr(data, 'shape'):
                print(f"        {feature_name} shape: {data.shape}")
                
                # 2차원이면 1차원으로 변환
                if len(data.shape) > 1:
                    print(f"        ⚠️ {feature_name} 2차원 데이터 감지")
                    if isinstance(data, pd.DataFrame):
                        data = data.iloc[:, 0]
                    elif isinstance(data, np.ndarray):
                        data = pd.Series(data.flatten())
                    else:
                        print(f"        ❌ {feature_name} 알 수 없는 2차원 타입")
                        return None
            
            # Series로 변환
            if not isinstance(data, pd.Series):
                try:
                    data = pd.Series(data)
                except Exception as e:
                    print(f"        ❌ {feature_name} Series 변환 실패: {e}")
                    return None
            
            # 인덱스 정렬
            try:
                # 공통 인덱스로 정렬
                aligned_data = data.reindex(target_index)
                aligned_data = aligned_data.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
                
                print(f"        {feature_name} 정렬 완료: {len(aligned_data)}개")
                return aligned_data
                
            except Exception as e:
                print(f"        ❌ {feature_name} 인덱스 정렬 실패: {e}")
                return None
            
        except Exception as e:
            print(f"        ❌ {feature_name} 정렬 치명적 실패: {e}")
            return None  

    def _clean_features_dataframe(self, features_df):
        """특징 데이터프레임 정리"""
        try:
            if features_df is None or features_df.empty:
                return pd.DataFrame()
            
            # 초기 불안정한 값들 제거
            stable_start = max(120, len(features_df) // 4)
            if len(features_df) > stable_start:
                features_df = features_df.iloc[stable_start:].copy()
            else:
                print(f"⚠️ 안정화 후 데이터 부족: {len(features_df)} <= {stable_start}")
                return features_df  # 그대로 반환
            
            # 데이터 정리
            features_df = features_df.fillna(method='ffill').fillna(0)
            features_df = features_df.replace([np.inf, -np.inf], 0)
            
            # 이상값 처리
            for col in features_df.columns:
                try:
                    q99 = features_df[col].quantile(0.99)
                    q01 = features_df[col].quantile(0.01)
                    features_df[col] = features_df[col].clip(lower=q01, upper=q99)
                except:
                    continue
            
            # 최종 유효성 검사
            if len(features_df) < 50:
                print(f"⚠️ 최종 특징 데이터가 부족: {len(features_df)}")
                return pd.DataFrame()
            
            return features_df
            
        except Exception as e:
            print(f"특징 데이터프레임 정리 실패: {e}")
            return pd.DataFrame()
    
    def _calculate_enhanced_features(self, excess_returns, returns, rf_data):
        """논문 기반 + 추가 특징들 - 안정화된 버전"""
        try:
            negative_excess_returns = excess_returns.where(excess_returns < 0, 0)
            
            # 논문 Table 2의 핵심 특징들 (EWM) - 안전한 계산
            ewm_downside_var_10 = safe_ewm_calculation((negative_excess_returns ** 2), halflife=10)
            ewm_downside_deviation_10 = np.sqrt(ewm_downside_var_10.abs()) * np.sqrt(252)
            
            ewm_mean_excess_20 = safe_ewm_calculation(excess_returns, halflife=20) * 252
            ewm_mean_excess_60 = safe_ewm_calculation(excess_returns, halflife=60) * 252
            
            ewm_downside_var_20 = safe_ewm_calculation((negative_excess_returns ** 2), halflife=20)
            ewm_downside_deviation_20 = np.sqrt(ewm_downside_var_20.abs()) * np.sqrt(252)
            
            ewm_downside_var_60 = safe_ewm_calculation((negative_excess_returns ** 2), halflife=60)
            ewm_downside_deviation_60 = np.sqrt(ewm_downside_var_60.abs()) * np.sqrt(252)
            
            # 안전한 나눗셈
            ewm_sortino_20 = ewm_mean_excess_20 / (ewm_downside_deviation_20 + 1e-6)
            ewm_sortino_60 = ewm_mean_excess_60 / (ewm_downside_deviation_60 + 1e-6)
            
            # 추가 특징들 (EWM 적용) - 안전한 계산
            ewm_variance_20 = safe_ewm_calculation((excess_returns ** 2), halflife=20)
            ewm_realized_vol = np.sqrt(ewm_variance_20.abs()) * np.sqrt(252)
            
            # Skewness - rolling으로 대체 (더 안정적)
            ewm_skewness = excess_returns.rolling(window=20, min_periods=10).skew()
            ewm_skewness = ewm_skewness.fillna(0)
            
            # Maximum Drawdown - 안전한 계산
            try:
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown_20 = drawdown.rolling(window=20, min_periods=10).min()
                max_drawdown_20 = max_drawdown_20.fillna(0)
            except:
                max_drawdown_20 = pd.Series(0, index=excess_returns.index)
            
            # Up Days Ratio - 안전한 계산
            up_days = (excess_returns > 0).astype(float)
            ewm_up_days_ratio = safe_ewm_calculation(up_days, halflife=20)
            
            # Volatility Ratio - 안전한 계산
            ewm_vol_ratio = ewm_downside_deviation_20 / (ewm_realized_vol + 1e-6)
            
            # RF Level - 안전한 계산
            if rf_data is not None:
                try:
                    ewm_rf_level = safe_ewm_calculation(
                        rf_data.reindex(excess_returns.index, method='ffill').fillna(self.default_rf_rate), 
                        halflife=20
                    )
                except:
                    ewm_rf_level = pd.Series(self.default_rf_rate, index=excess_returns.index)
            else:
                ewm_rf_level = pd.Series(self.default_rf_rate, index=excess_returns.index)
            
            # DataFrame 생성
            features_df = pd.DataFrame({
                # 논문 Table 2의 핵심 특징들
                'downside_deviation_10': ewm_downside_deviation_10,
                'sortino_ratio_20': ewm_sortino_20,
                'sortino_ratio_60': ewm_sortino_60,
                
                # 추가 특징들
                'realized_vol': ewm_realized_vol,
                'mean_excess_return_20': ewm_mean_excess_20,
                'skewness': ewm_skewness,
                'max_drawdown': max_drawdown_20,
                'up_days_ratio': ewm_up_days_ratio,
                'vol_ratio': ewm_vol_ratio,
                'rf_level': ewm_rf_level
            }, index=excess_returns.index)
            
            # 초기 불안정한 값들 제거
            stable_start = max(120, len(features_df) // 4)
            if len(features_df) > stable_start:
                features_df = features_df.iloc[stable_start:].copy()
            else:
                print(f"경고: 안정화 후 데이터 부족 ({len(features_df)} <= {stable_start})")
                return pd.DataFrame()
            
            # 데이터 정리
            features_df = features_df.fillna(method='ffill').fillna(0)
            features_df = features_df.replace([np.inf, -np.inf], 0)
            
            # 이상값 처리
            for col in features_df.columns:
                q99 = features_df[col].quantile(0.99)
                q01 = features_df[col].quantile(0.01)
                features_df[col] = features_df[col].clip(lower=q01, upper=q99)
            
            print(f"EWM 특징 계산 완료: {len(features_df)}개")
            print(f"핵심 특징: Downside Deviation (hl=10), Sortino Ratio (hl=20,60)")
            
            # 최종 데이터 품질 확인
            if len(features_df) < 50:
                print(f"경고: 최종 특징 데이터가 너무 적습니다 ({len(features_df)})")
                return pd.DataFrame()
            
            return features_df
            
        except Exception as e:
            print(f"확장 특징 계산 오류: {e}")
            return pd.DataFrame()
    
    def fit_jump_model(self, features_df):
        """Jump Model 학습 - 안정화된 버전"""
        try:
            if features_df.empty or len(features_df) < 50:
                print(f"학습용 특징 데이터 부족: {len(features_df)}")
                return None
            
            X = features_df.values
            
            # 무효한 값 확인 및 정리
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print("특징 데이터 정리 중...")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 데이터 스케일링
            try:
                X_scaled = self.scaler.fit_transform(X)
            except Exception as e:
                print(f"스케일링 실패: {e}")
                return None
            
            # 초기 클러스터링
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
        """Jump penalty를 적용하여 상태 시퀀스 최적화 - 안정화된 버전"""
        try:
            n_samples = len(X)
            states = initial_states.copy()
            
            # 반복적 최적화
            for iteration in range(10):
                converged = True
                
                for i in range(1, n_samples - 1):
                    current_state = states[i]
                    min_cost = float('inf')
                    best_state = current_state
                    
                    for new_state in range(self.n_states):
                        try:
                            # 클러스터링 비용
                            cluster_cost = np.linalg.norm(X[i] - self.cluster_centers[new_state]) ** 2
                            
                            # Jump penalty
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
        """체제별 특성 분석 및 Bull/Bear 레이블링 - 안정화된 버전"""
        try:
            regime_stats = {}
            
            for state in range(self.n_states):
                state_mask = (states == state)
                state_features = features_df[state_mask]
                
                if len(state_features) > 0:
                    regime_stats[state] = {
                        'count': len(state_features),
                        'avg_downside_dev': safe_mean(state_features['downside_deviation_10'], 0.0),
                        'avg_sortino_20': safe_mean(state_features['sortino_ratio_20'], 0.0),
                        'avg_sortino_60': safe_mean(state_features['sortino_ratio_60'], 0.0)
                    }
                    
                    # 추가 특징들이 있다면
                    if 'mean_excess_return_20' in state_features.columns:
                        regime_stats[state].update({
                            'avg_excess_return': safe_mean(state_features['mean_excess_return_20'], 0.0),
                            'avg_volatility': safe_mean(state_features['realized_vol'], 0.0),
                            'avg_up_days': safe_mean(state_features['up_days_ratio'], 0.0),
                            'avg_rf_level': safe_mean(state_features['rf_level'], self.default_rf_rate)
                        })
                else:
                    regime_stats[state] = {
                        'count': 0,
                        'avg_downside_dev': 0.0,
                        'avg_sortino_20': 0.0,
                        'avg_sortino_60': 0.0
                    }
            
            # Bear 상태 식별 (높은 하방변동성, 낮은 Sortino ratio)
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
            print(f"\n=== {self.benchmark_name} EWM 체제별 특성 ===")
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
                    
                    if 'avg_excess_return' in stats:
                        print(f"  - 평균 초과수익률: {stats['avg_excess_return']*100:.2f}%")
                        print(f"  - 평균 변동성: {stats['avg_volatility']*100:.1f}%")
            
            return regime_stats
            
        except Exception as e:
            print(f"체제 분석 오류: {e}")
            # 기본 매핑 생성
            self.state_mapping = {0: 'BULL', 1: 'BEAR'}
            return {}
    
    def predict_regime(self, current_features):
        """현재 시장 체제 예측 - 안정화된 버전"""
        if not self.is_trained or self.cluster_centers is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            # 입력 특징 안전하게 처리
            if isinstance(current_features, pd.Series):
                X = current_features.values.reshape(1, -1)
            else:
                X = np.array(current_features).reshape(1, -1)
            
            # NaN 및 무한대 처리
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 스케일링
            try:
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                print(f"예측 시 스케일링 오류: {e}")
                predicted_state = 0
                confidence = 0.5
                regime = self.state_mapping.get(predicted_state, 'BULL')
                return regime, confidence
            
            # 가장 가까운 클러스터 찾기
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
                
                # Jump penalty 고려
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

    def _safe_confidence_conversion(self, confidence):
        """안전한 신뢰도 변환"""
        try:
            if confidence is None:
                return 0.5
            
            if isinstance(confidence, pd.Series):
                if len(confidence) > 0:
                    # Series의 마지막 유효값 사용
                    valid_values = confidence.dropna()
                    if len(valid_values) > 0:
                        conf_val = valid_values.iloc[-1]
                        return max(0.0, min(1.0, float(conf_val)))
                    else:
                        return 0.5
                else:
                    return 0.5
            
            elif isinstance(confidence, (int, float)):
                if np.isfinite(confidence):
                    return max(0.0, min(1.0, float(confidence)))
                else:
                    return 0.5
            
            else:
                # 기타 타입은 float 변환 시도
                conf_val = float(confidence)
                return max(0.0, min(1.0, conf_val))
                
        except Exception as e:
            print(f"신뢰도 변환 실패: {e}")
            return 0.5
    
    def _safe_features_conversion(self, features):
        """안전한 특징 변환"""
        try:
            if features is None:
                return {}
            
            if isinstance(features, pd.Series):
                result = {}
                for key, value in features.items():
                    try:
                        if pd.isna(value):
                            result[key] = 0.0
                        else:
                            result[key] = float(value)
                    except:
                        result[key] = 0.0
                return result
            
            elif isinstance(features, dict):
                result = {}
                for key, value in features.items():
                    try:
                        if pd.isna(value):
                            result[key] = 0.0
                        else:
                            result[key] = float(value)
                    except:
                        result[key] = 0.0
                return result
            
            else:
                return {}
                
        except Exception as e:
            print(f"특징 변환 실패: {e}")
            return {}
    
    def train_model_with_cutoff(self, start_date=None, end_date=None):
        """특정 기간의 데이터로만 모델 학습 - 범용적 오류 처리 강화"""
        if end_date is None:
            end_date = self.training_cutoff_date
        
        if start_date is None:
            start_date = end_date - timedelta(days=365*20)
        
        print(f"모델 학습: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # 1단계: 학습용 데이터 다운로드
            extended_start = start_date - timedelta(days=200)
            
            try:
                price_data = self.download_benchmark_data(extended_start, end_date)
                
                if price_data is None or price_data.empty:
                    print(f"❌ {self.benchmark_name} 학습 데이터 다운로드 실패")
                    return False
                
                print(f"✅ 학습 데이터: {len(price_data)}일")
                
            except Exception as e:
                print(f"❌ 학습 데이터 다운로드 오류: {e}")
                return False
            
            # 2단계: 데이터 충분성 검사
            if len(price_data) < self.min_data_length:
                print(f"❌ {self.benchmark_name} 학습 데이터 부족: {len(price_data)} < {self.min_data_length}")
                return False
            
            # 3단계: 특징 계산
            try:
                print(f"학습용 특징 계산 중...")
                features_df = self.calculate_features(price_data)
                
                if features_df is None or features_df.empty:
                    print(f"❌ {self.benchmark_name} 학습용 특징 계산 실패")
                    return False
                
                print(f"✅ 학습용 특징: {len(features_df)}개")
                
            except Exception as e:
                print(f"❌ 학습용 특징 계산 오류: {e}")
                return False
            
            # 4단계: 학습 기간으로 제한
            try:
                training_features = features_df[start_date:end_date]
                
                if len(training_features) < 50:
                    print(f"❌ 학습 기간 데이터 부족: {len(training_features)} < 50")
                    return False
                
                print(f"✅ 학습 기간 특징: {len(training_features)}개")
                
            except Exception as e:
                print(f"❌ 학습 기간 제한 오류: {e}")
                return False
            
            # 5단계: 모델 학습
            try:
                print(f"모델 학습 시작...")
                result = self.fit_jump_model(training_features)
                
                if result is None:
                    print(f"❌ 모델 학습 실패")
                    return False
                
                print(f"✅ 모델 학습 완료")
                return True
                
            except Exception as e:
                print(f"❌ 모델 학습 오류: {e}")
                return False
                
        except Exception as e:
            print(f"❌ 모델 학습 치명적 오류: {e}")
            return False
    
    def get_current_regime_with_training_cutoff(self):
        """학습 마감일까지만 학습하고 현재 체제 예측 - 범용적 오류 처리 강화"""
        try:
            # 1단계: 모델 학습 상태 확인
            if not self.is_trained:
                print(f"모델 학습 시작: {self.benchmark_name}")
                success = self.train_model_with_cutoff()
                if not success:
                    print(f"❌ {self.benchmark_name} 모델 학습 실패")
                    return None
            
            # 2단계: 현재 시점까지의 데이터 가져오기
            current_date = datetime.now()
            inference_start = self.training_cutoff_date - timedelta(days=200)
            
            print(f"추론 데이터 다운로드: {inference_start.strftime('%Y-%m-%d')} ~ {current_date.strftime('%Y-%m-%d')}")
            
            try:
                price_data = self.download_benchmark_data(inference_start, current_date)
                
                if price_data is None or price_data.empty:
                    print(f"❌ {self.benchmark_name} 추론 데이터 다운로드 실패")
                    return None
                
                print(f"✅ 추론 데이터: {len(price_data)}일")
                
            except Exception as e:
                print(f"❌ 데이터 다운로드 오류: {e}")
                return None
            
            # 3단계: 특징 계산
            try:
                print(f"특징 계산 시작...")
                features_df = self.calculate_features(price_data)
                
                if features_df is None or features_df.empty:
                    print(f"❌ {self.benchmark_name} 특징 계산 실패 - 빈 결과")
                    return None
                
                print(f"✅ 특징 계산 완료: {len(features_df)}개")
                
            except Exception as e:
                print(f"❌ 특징 계산 오류: {e}")
                return None
            
            # 4단계: 최신 특징 추출 및 검증
            try:
                if len(features_df) == 0:
                    print(f"❌ 특징 데이터가 비어있음")
                    return None
                
                latest_features = features_df.iloc[-1]
                latest_date = features_df.index[-1]
                
                # 특징 값 검증
                if latest_features.isna().all():
                    print(f"❌ 최신 특징이 모두 NaN")
                    return None
                
                print(f"✅ 최신 특징 추출: {latest_date.strftime('%Y-%m-%d')}")
                
            except Exception as e:
                print(f"❌ 최신 특징 추출 오류: {e}")
                return None
            
            # 5단계: 체제 예측 (핵심 부분)
            try:
                print(f"체제 예측 시작...")
                current_regime, confidence = self.predict_regime(latest_features)
                
                if current_regime is None:
                    print(f"❌ 체제 예측 실패 - None 반환")
                    return None
                
                print(f"✅ 체제 예측 완료: {current_regime}")
                
            except Exception as e:
                print(f"❌ 체제 예측 오류: {e}")
                return None
            
            # 6단계: 결과 구성 (안전한 변환)
            try:
                is_out_of_sample = latest_date > self.training_cutoff_date
                
                # 안전한 confidence 변환
                safe_confidence = self._safe_confidence_conversion(confidence)
                
                # 분석 정보 구성
                analysis_info = {
                    'regime': str(current_regime),
                    'confidence': safe_confidence,
                    'date': latest_date,
                    'features': self._safe_features_conversion(latest_features),
                    'is_out_of_sample': bool(is_out_of_sample),
                    'training_cutoff': self.training_cutoff_date.strftime('%Y-%m-%d'),
                    'feature_type': "논문 정확한 3특징" if self.use_paper_features_only else "논문 기반 + 추가",
                    'ewm_applied': True,
                    'rf_ticker': self.rf_ticker,
                    'dynamic_rf_used': HAS_RF_UTILS and self.rf_manager is not None
                }
                
                # Risk-free rate 정보 추가
                try:
                    if HAS_RF_UTILS and self.rf_manager:
                        rf_stats = self.rf_manager.get_risk_free_rate_stats(
                            latest_date - timedelta(days=30), latest_date
                        )
                        analysis_info['current_rf_rate'] = safe_float_conversion(rf_stats['end_rate'], self.default_rf_rate * 100)
                        analysis_info['avg_rf_rate_30d'] = safe_float_conversion(rf_stats['mean_rate'], self.default_rf_rate * 100)
                    else:
                        analysis_info['current_rf_rate'] = self.default_rf_rate * 100
                        analysis_info['avg_rf_rate_30d'] = self.default_rf_rate * 100
                except Exception as e:
                    print(f"⚠️ RF 정보 추가 실패: {e}")
                    analysis_info['current_rf_rate'] = self.default_rf_rate * 100
                    analysis_info['avg_rf_rate_30d'] = self.default_rf_rate * 100
                
                print(f"✅ 체제 분석 완료: {current_regime} (신뢰도: {safe_confidence:.2%})")
                return analysis_info
                
            except Exception as e:
                print(f"❌ 결과 구성 오류: {e}")
                return None
                
        except Exception as e:
            print(f"❌ 체제 분석 치명적 오류: {e}")
            return None
    
    def get_current_regime(self):
        """현재 시장 체제 확인 (기존 호환성 유지)"""
        return self.get_current_regime_with_training_cutoff()


# 편의 함수들 (EWM 버전)
def create_paper_exact_jump_model(benchmark_ticker, benchmark_name, rf_ticker='^IRX', **kwargs):
    """논문 Table 2의 정확한 3가지 특징만 사용하는 Jump Model"""
    return UniversalJumpModel(
        benchmark_ticker=benchmark_ticker,
        benchmark_name=benchmark_name,
        rf_ticker=rf_ticker,
        use_paper_features_only=True,
        **kwargs
    )

def create_enhanced_ewm_jump_model(benchmark_ticker, benchmark_name, rf_ticker='^IRX', **kwargs):
    """논문 기반 + 추가 EWM 특징을 사용하는 Jump Model"""
    return UniversalJumpModel(
        benchmark_ticker=benchmark_ticker,
        benchmark_name=benchmark_name,
        rf_ticker=rf_ticker,
        use_paper_features_only=False,
        **kwargs
    )

def compare_feature_approaches(benchmark_ticker, benchmark_name):
    """논문 정확한 특징 vs 추가 특징 비교"""
    print(f"\n=== {benchmark_name} EWM 특징 비교 ===")
    
    # 논문 정확한 특징만
    paper_model = create_paper_exact_jump_model(benchmark_ticker, benchmark_name)
    paper_result = paper_model.get_current_regime_with_training_cutoff()
    
    # 논문 기반 + 추가 특징
    enhanced_model = create_enhanced_ewm_jump_model(benchmark_ticker, benchmark_name)
    enhanced_result = enhanced_model.get_current_regime_with_training_cutoff()
    
    if paper_result and enhanced_result:
        print(f"\n논문 정확한 특징 (3개): {paper_result['regime']} (신뢰도: {paper_result['confidence']:.2%})")
        print(f"논문 기반 + 추가 특징: {enhanced_result['regime']} (신뢰도: {enhanced_result['confidence']:.2%})")
        print(f"일치 여부: {'✓' if paper_result['regime'] == enhanced_result['regime'] else '✗'}")
    
    return paper_result, enhanced_result


# 사용 예시
if __name__ == "__main__":
    print("=== EWM Jump Model (안정화된 버전) 테스트 ===")
    
    # GLD 특별 테스트
    print("\n1. GLD (Gold ETF) 안정화된 분석")
    gld_model = create_paper_exact_jump_model(
        benchmark_ticker='GLD',
        benchmark_name='SPDR Gold Trust (Stabilized)',
        jump_penalty=30.0  # 더 관대한 페널티
    )
    
    gld_result = gld_model.get_current_regime_with_training_cutoff()
    if gld_result:
        print(f"GLD 체제: {gld_result['regime']} (신뢰도: {gld_result['confidence']:.2%})")
        print(f"EWM 적용: {gld_result['ewm_applied']}")
        print(f"특징 유형: {gld_result['feature_type']}")
    else:
        print("GLD 분석 실패")
    
    print(f"\n=== EWM Jump Model 안정화 완성 ===")
