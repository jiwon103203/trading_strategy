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
                # Series의 마지막 값 또는 유일한 값 사용
                val = value.iloc[-1]
                return float(val) if not pd.isna(val) else default
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
            result = series.mean()
            return safe_float_conversion(result, default)
        else:
            return default
    except:
        return default

def safe_std(series, default=0.0):
    """안전한 표준편차 계산"""
    try:
        if isinstance(series, pd.Series) and len(series) > 1:
            result = series.std()
            return safe_float_conversion(result, default)
        else:
            return default
    except:
        return default

def safe_skew(series, default=0.0):
    """안전한 왜도 계산"""
    try:
        if isinstance(series, pd.Series) and len(series) > 2:
            result = series.skew()
            return safe_float_conversion(result, default)
        else:
            return default
    except:
        return default

def safe_min(series, default=0.0):
    """안전한 최솟값 계산"""
    try:
        if isinstance(series, pd.Series) and len(series) > 0:
            result = series.min()
            return safe_float_conversion(result, default)
        else:
            return default
    except:
        return default

def safe_sum(series, default=0.0):
    """안전한 합계 계산"""
    try:
        if isinstance(series, pd.Series):
            result = series.sum()
            return safe_float_conversion(result, default)
        else:
            return default
    except:
        return default

class UniversalJumpModel:
    """
    범용 Jump Model with Training Cutoff Support + 동적 Risk-Free Rate 지원
    다양한 지수에 적용 가능한 시장 체제(Bull/Bear) 감지
    2024년까지 학습, 2025년은 추론용
    동적 risk-free rate를 사용한 위험조정 수익률 계산
    Series → float 변환 오류 수정 버전
    """
    
    def __init__(self, benchmark_ticker, benchmark_name="Market", 
                 n_states=2, lookback_window=20, jump_penalty=50.0,
                 training_cutoff_date=None, rf_ticker='^IRX', default_rf_rate=0.02):
        """
        Parameters:
        - benchmark_ticker: 벤치마크 지수 티커 (예: '^GSPC', '069500.KS', 'URTH')
        - benchmark_name: 벤치마크 이름
        - n_states: 상태 수 (기본값: 2 - Bull/Bear)
        - lookback_window: 특징 계산을 위한 lookback 기간
        - jump_penalty: 체제 전환에 대한 페널티
        - training_cutoff_date: 학습 데이터 마지막 날짜 (None이면 전체 사용)
        - rf_ticker: Risk-free rate 티커 (기본: ^IRX)
        - default_rf_rate: 기본 risk-free rate (기본: 2%)
        """
        self.benchmark_ticker = benchmark_ticker
        self.benchmark_name = benchmark_name
        self.n_states = n_states
        self.lookback_window = lookback_window
        self.jump_penalty = jump_penalty
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
        
        print(f"Jump Model 초기화: 학습 마감일 = {self.training_cutoff_date.strftime('%Y-%m-%d')}")
        print(f"Risk-Free Rate: {self.rf_ticker} (기본값: {self.default_rf_rate*100:.1f}%)")
    
    def download_benchmark_data(self, start_date, end_date):
        """벤치마크 데이터 다운로드"""
        try:
            print(f"{self.benchmark_name} 데이터 다운로드 중...")
            data = yf.download(self.benchmark_ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"{self.benchmark_name} 데이터를 가져올 수 없습니다.")
                
            # 데이터 정리 (NaN 제거)
            data = data.dropna()
            
            return data
            
        except Exception as e:
            print(f"데이터 다운로드 실패: {e}")
            return None
    
    def calculate_features(self, price_data):
        """
        Jump Model을 위한 특징 계산 (Series → float 변환 오류 수정)
        """
        features_list = []
        
        # 일일 수익률
        returns = price_data['Close'].pct_change().dropna()
        
        if len(returns) == 0:
            print("가격 데이터에서 수익률을 계산할 수 없습니다.")
            return pd.DataFrame()
        
        # Risk-free rate 다운로드 (특징 계산 기간에 맞춰)
        rf_data = None
        if HAS_RF_UTILS and self.rf_manager:
            try:
                start_date = returns.index[0]
                end_date = returns.index[-1]
                rf_data = self.rf_manager.download_risk_free_rate(start_date, end_date)
                print(f"Risk-free rate 데이터 사용: {len(rf_data) if rf_data is not None else 0}개")
            except Exception as e:
                print(f"Risk-free rate 다운로드 실패: {e}")
                rf_data = None
        
        # Rolling window로 특징 계산
        for i in range(self.lookback_window, len(returns)):
            try:
                window_returns = returns.iloc[i-self.lookback_window:i]
                window_dates = returns.index[i-self.lookback_window:i]
                
                # 윈도우 데이터 유효성 검사
                if len(window_returns) < self.lookback_window // 2:
                    continue
                
                # 해당 기간의 risk-free rate
                avg_rf_rate = self.default_rf_rate
                if rf_data is not None:
                    try:
                        window_rf = rf_data.reindex(window_dates, method='ffill').fillna(self.default_rf_rate)
                        daily_rf = window_rf / 252  # 일일 risk-free rate
                        excess_returns = window_returns - daily_rf
                        avg_rf_rate = safe_mean(window_rf, self.default_rf_rate)
                    except:
                        excess_returns = window_returns - (self.default_rf_rate / 252)
                        avg_rf_rate = self.default_rf_rate
                else:
                    excess_returns = window_returns - (self.default_rf_rate / 252)
                    avg_rf_rate = self.default_rf_rate
                
                # 1. 평균 초과 수익률 (위험조정) - 안전한 변환
                mean_excess_return = safe_mean(excess_returns, 0.0)
                
                # 2. 실현 변동성 (Realized Volatility) - 안전한 변환
                realized_vol = safe_std(window_returns, 0.0) * np.sqrt(252)
                
                # 3. 하방 변동성 (Downside Volatility) - 초과수익률 기준
                downside_excess = excess_returns[excess_returns < 0]
                if len(downside_excess) > 0:
                    downside_vol = safe_std(downside_excess, 0.0) * np.sqrt(252)
                else:
                    downside_vol = 0.0
                
                # 4. 왜도 (Skewness) - 초과수익률 기준
                skewness = safe_skew(excess_returns, 0.0)
                
                # 5. 최대 낙폭 (Maximum Drawdown) - 안전한 계산
                try:
                    cumulative = (1 + window_returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = safe_min(drawdown, 0.0)
                except:
                    max_drawdown = 0.0
                
                # 6. 상승/하락 일수 비율 (초과수익률 기준) - 안전한 계산
                try:
                    up_days_count = safe_sum(excess_returns > 0, 0)
                    up_days_ratio = up_days_count / len(excess_returns) if len(excess_returns) > 0 else 0.0
                except:
                    up_days_ratio = 0.0
                
                # 7. 변동성 비율 (Volatility Ratio)
                if realized_vol > 0:
                    vol_ratio = downside_vol / realized_vol
                else:
                    vol_ratio = 1.0
                
                # 8. Sharpe-like 비율 (위험조정 성과)
                if realized_vol > 0:
                    risk_adjusted_return = mean_excess_return * 252 / realized_vol
                else:
                    risk_adjusted_return = 0.0
                
                # 9. 현재 risk-free rate 수준 - 안전한 변환
                current_rf_level = safe_float_conversion(avg_rf_rate, self.default_rf_rate)
                
                features_list.append({
                    'date': returns.index[i],
                    'mean_excess_return': mean_excess_return,
                    'realized_vol': realized_vol,
                    'downside_vol': downside_vol,
                    'skewness': skewness,
                    'max_drawdown': max_drawdown,
                    'up_days_ratio': up_days_ratio,
                    'vol_ratio': vol_ratio,
                    'risk_adjusted_return': risk_adjusted_return,
                    'rf_level': current_rf_level
                })
                
            except Exception as e:
                print(f"특징 계산 중 오류 (인덱스 {i}): {e}")
                continue
        
        if not features_list:
            print("특징 계산 결과가 비어 있습니다.")
            return pd.DataFrame()
        
        features_df = pd.DataFrame(features_list).set_index('date')
        
        # NaN 값 처리
        features_df = features_df.fillna(0)
        
        # 무한대 값 처리
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        print(f"특징 계산 완료: {len(features_df)}개 (동적 RF 적용: {rf_data is not None})")
        
        return features_df
    
    def fit_jump_model(self, features_df):
        """Jump Model 학습 - 안전한 데이터 처리"""
        try:
            # 특징 정규화
            X = features_df.values
            
            # 무효한 값 확인
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print("특징 데이터에 NaN 또는 무한대 값이 있습니다. 정리 중...")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            X_scaled = self.scaler.fit_transform(X)
            
            # 초기 클러스터링 (K-means)
            kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            initial_states = kmeans.fit_predict(X_scaled)
            self.cluster_centers = kmeans.cluster_centers_
            
            # Jump penalty를 적용한 최적화
            optimized_states = self.optimize_with_jump_penalty(X_scaled, initial_states)
            
            # 상태별 특성 분석
            self.analyze_regimes(features_df, optimized_states)
            
            self.is_trained = True
            return optimized_states
            
        except Exception as e:
            print(f"모델 학습 중 오류: {e}")
            return None
    
    def optimize_with_jump_penalty(self, X, initial_states):
        """Jump penalty를 적용하여 상태 시퀀스 최적화"""
        n_samples = len(X)
        states = initial_states.copy()
        
        # 반복적 최적화
        for iteration in range(10):
            converged = True
            
            for i in range(1, n_samples - 1):
                current_state = states[i]
                
                # 각 상태로 변경할 때의 비용 계산
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
    
    def analyze_regimes(self, features_df, states):
        """체제별 특성 분석 및 Bull/Bear 레이블링 (안전한 통계 계산)"""
        regime_stats = {}
        
        for state in range(self.n_states):
            state_mask = (states == state)
            state_features = features_df[state_mask]
            
            if len(state_features) > 0:
                regime_stats[state] = {
                    'count': len(state_features),
                    'avg_excess_return': safe_mean(state_features['mean_excess_return'], 0.0),
                    'avg_volatility': safe_mean(state_features['realized_vol'], 0.0),
                    'avg_downside_vol': safe_mean(state_features['downside_vol'], 0.0),
                    'avg_drawdown': safe_mean(state_features['max_drawdown'], 0.0),
                    'avg_up_days': safe_mean(state_features['up_days_ratio'], 0.0),
                    'avg_vol_ratio': safe_mean(state_features['vol_ratio'], 1.0),
                    'avg_risk_adjusted': safe_mean(state_features['risk_adjusted_return'], 0.0),
                    'avg_rf_level': safe_mean(state_features['rf_level'], self.default_rf_rate)
                }
            else:
                regime_stats[state] = {
                    'count': 0,
                    'avg_excess_return': 0.0,
                    'avg_volatility': 0.0,
                    'avg_downside_vol': 0.0,
                    'avg_drawdown': 0.0,
                    'avg_up_days': 0.0,
                    'avg_vol_ratio': 1.0,
                    'avg_risk_adjusted': 0.0,
                    'avg_rf_level': self.default_rf_rate
                }
        
        # Bear 상태 식별 (위험조정 점수 기반)
        state_scores = {}
        for state in range(self.n_states):
            # Bear 점수: 낮은 위험조정 수익률, 높은 하방 변동성, 낮은 초과수익률
            bear_score = (
                regime_stats[state]['avg_downside_vol'] * 3 +
                abs(regime_stats[state]['avg_drawdown']) * 4 -
                regime_stats[state]['avg_excess_return'] * 1000 -  # 초과수익률이 중요
                regime_stats[state]['avg_risk_adjusted'] * 2 -
                regime_stats[state]['avg_up_days'] * 2
            )
            state_scores[state] = bear_score
        
        # 가장 높은 Bear 점수를 가진 상태를 Bear로 지정
        bear_state = max(state_scores.keys(), key=lambda x: state_scores[x])
        
        # 상태 매핑
        self.state_mapping = {}
        for state in range(self.n_states):
            if state == bear_state:
                self.state_mapping[state] = 'BEAR'
            else:
                self.state_mapping[state] = 'BULL'
        
        # 통계 출력 (동적 RF 정보 포함)
        print(f"\n=== {self.benchmark_name} 체제별 특성 (학습기간: ~{self.training_cutoff_date.strftime('%Y-%m-%d')}) ===")
        rf_info = f"동적 RF ({self.rf_ticker})" if HAS_RF_UTILS and self.rf_manager else f"고정 RF ({self.default_rf_rate*100:.1f}%)"
        print(f"Risk-Free Rate: {rf_info}")
        
        for state, stats in regime_stats.items():
            regime_type = self.state_mapping[state]
            if stats['count'] > 0:
                print(f"\n{regime_type} 체제 (State {state}):")
                print(f"  - 기간 비율: {stats['count'] / len(features_df) * 100:.1f}%")
                print(f"  - 평균 초과수익률: {stats['avg_excess_return']*252*100:.2f}%")
                print(f"  - 평균 위험조정 수익률: {stats['avg_risk_adjusted']:.3f}")
                print(f"  - 평균 변동성: {stats['avg_volatility']*100:.1f}%")
                print(f"  - 평균 하방 변동성: {stats['avg_downside_vol']*100:.1f}%")
                print(f"  - 평균 최대 낙폭: {stats['avg_drawdown']*100:.1f}%")
                print(f"  - 평균 상승일 비율: {stats['avg_up_days']*100:.1f}%")
                print(f"  - 평균 RF 수준: {stats['avg_rf_level']*100:.3f}%")
        
        return regime_stats
    
    def predict_regime(self, current_features):
        """현재 시장 체제 예측 - 안전한 신뢰도 계산"""
        if not self.is_trained or self.cluster_centers is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            # 특징 정규화
            X = current_features.values.reshape(1, -1)
            
            # 무효한 값 처리
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            X_scaled = self.scaler.transform(X)
            
            # 가장 가까운 클러스터 찾기
            distances = []
            for center in self.cluster_centers:
                try:
                    dist = np.linalg.norm(X_scaled - center)
                    distances.append(float(dist))
                except:
                    distances.append(float('inf'))
            
            if not distances or all(d == float('inf') for d in distances):
                # 거리 계산 실패 시 기본값
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
                
                # 신뢰도 계산 - 안전한 버전
                try:
                    min_dist = min(distances)
                    max_dist = max(distances)
                    if max_dist > min_dist and max_dist > 0:
                        confidence = 1 - (min_dist / max_dist)
                        confidence = max(0.0, min(1.0, confidence))  # 0-1 범위로 제한
                    else:
                        confidence = 0.5
                except:
                    confidence = 0.5
            
            self.current_regime = predicted_state
            
            return self.state_mapping[predicted_state], confidence
            
        except Exception as e:
            print(f"체제 예측 중 오류: {e}")
            return 'BULL', 0.5  # 기본값 반환
    
    def train_model_with_cutoff(self, start_date=None, end_date=None):
        """
        특정 기간의 데이터로만 모델 학습
        end_date가 None이면 training_cutoff_date 사용
        """
        if end_date is None:
            end_date = self.training_cutoff_date
        
        if start_date is None:
            # 20년 전부터 학습
            start_date = end_date - timedelta(days=365*20)
        
        print(f"\n모델 학습 시작: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # 학습용 데이터 다운로드
            price_data = self.download_benchmark_data(
                start_date - timedelta(days=self.lookback_window * 2),
                end_date
            )
            
            if price_data is None or price_data.empty:
                print(f"{self.benchmark_name} 학습 데이터를 가져올 수 없습니다.")
                return False
            
            # 특징 계산 (동적 RF 포함)
            features_df = self.calculate_features(price_data)
            
            if features_df.empty:
                print(f"{self.benchmark_name} 특징 계산 실패")
                return False
            
            # 학습 기간으로 제한
            training_features = features_df[start_date:end_date]
            
            if len(training_features) < self.lookback_window * 2:
                print(f"학습 데이터 부족: {len(training_features)} < {self.lookback_window * 2}")
                return False
            
            print(f"학습 특징 수: {len(training_features)}")
            
            # 모델 학습
            result = self.fit_jump_model(training_features)
            
            return result is not None
            
        except Exception as e:
            print(f"모델 학습 중 예외 발생: {e}")
            return False
    
    def get_current_regime_with_training_cutoff(self):
        """
        학습 마감일까지만 학습하고 현재 체제 예측
        2024년까지 학습, 2025년은 추론용
        안전한 오류 처리 포함
        """
        try:
            # 모델이 학습되지 않았으면 먼저 학습
            if not self.is_trained:
                success = self.train_model_with_cutoff()
                if not success:
                    return None
            
            # 현재 시점까지의 데이터 가져오기 (추론용)
            current_date = datetime.now()
            inference_start = self.training_cutoff_date - timedelta(days=self.lookback_window * 2)
            
            print(f"\n추론 데이터: {inference_start.strftime('%Y-%m-%d')} ~ {current_date.strftime('%Y-%m-%d')}")
            
            # 추론용 데이터 다운로드
            price_data = self.download_benchmark_data(inference_start, current_date)
            
            if price_data is None or price_data.empty:
                print(f"{self.benchmark_name} 추론 데이터를 가져올 수 없습니다.")
                return None
            
            # 특징 계산 (동적 RF 포함)
            features_df = self.calculate_features(price_data)
            
            if features_df.empty:
                print(f"{self.benchmark_name} 추론 특징 계산 실패")
                return None
            
            # 최신 특징으로 예측
            latest_features = features_df.iloc[-1]
            current_regime, confidence = self.predict_regime(latest_features)
            
            # 2025년 데이터 사용 여부 확인
            latest_date = features_df.index[-1]
            is_out_of_sample = latest_date > self.training_cutoff_date
            
            # 추가 분석 정보
            analysis_info = {
                'regime': current_regime,
                'confidence': safe_float_conversion(confidence, 0.5),
                'date': latest_date,
                'features': latest_features.to_dict(),
                'is_out_of_sample': is_out_of_sample,
                'training_cutoff': self.training_cutoff_date.strftime('%Y-%m-%d'),
                'rf_ticker': self.rf_ticker,
                'dynamic_rf_used': HAS_RF_UTILS and self.rf_manager is not None
            }
            
            # Risk-free rate 정보 추가
            if HAS_RF_UTILS and self.rf_manager:
                try:
                    rf_stats = self.rf_manager.get_risk_free_rate_stats(
                        latest_date - timedelta(days=30), latest_date
                    )
                    analysis_info['current_rf_rate'] = safe_float_conversion(rf_stats['end_rate'], self.default_rf_rate * 100)
                    analysis_info['avg_rf_rate_30d'] = safe_float_conversion(rf_stats['mean_rate'], self.default_rf_rate * 100)
                except:
                    analysis_info['current_rf_rate'] = self.default_rf_rate * 100
                    analysis_info['avg_rf_rate_30d'] = self.default_rf_rate * 100
            else:
                analysis_info['current_rf_rate'] = self.default_rf_rate * 100
                analysis_info['avg_rf_rate_30d'] = self.default_rf_rate * 100
            
            return analysis_info
            
        except Exception as e:
            print(f"체제 분석 중 예외 발생: {e}")
            return None
    
    def get_regime_history(self, start_date, end_date):
        """과거 체제 이력 계산 (동적 RF 지원) - 안전한 버전"""
        try:
            # 데이터 다운로드
            price_data = self.download_benchmark_data(
                start_date - timedelta(days=self.lookback_window * 2),
                end_date
            )
            
            if price_data is None or price_data.empty:
                print(f"{self.benchmark_name} 데이터를 가져올 수 없습니다.")
                return None
            
            # 특징 계산 (동적 RF 포함)
            features_df = self.calculate_features(price_data)
            
            if features_df.empty:
                print(f"{self.benchmark_name} 특징 계산 실패")
                return None
            
            # 모델 학습
            states = self.fit_jump_model(features_df)
            
            if states is None:
                return None
            
            # 체제 이력 생성
            regime_history = pd.DataFrame({
                'state': states,
                'regime': [self.state_mapping[s] for s in states]
            }, index=features_df.index)
            
            return regime_history[start_date:end_date]
            
        except Exception as e:
            print(f"체제 이력 계산 중 오류: {e}")
            return None
    
    def get_current_regime(self):
        """현재 시장 체제 확인 (기존 호환성 유지)"""
        return self.get_current_regime_with_training_cutoff()
    
    def get_regime_statistics(self, start_date, end_date):
        """체제별 상세 통계 (동적 RF 정보 포함) - 안전한 버전"""
        try:
            regime_history = self.get_regime_history(start_date, end_date)
            
            if regime_history is None or regime_history.empty:
                return None
            
            # 체제 전환 분석
            regime_changes = regime_history[regime_history['regime'] != regime_history['regime'].shift()]
            
            # 체제별 지속 기간
            regime_durations = []
            for i in range(len(regime_changes) - 1):
                start = regime_changes.index[i]
                end = regime_changes.index[i + 1]
                duration = (end - start).days
                regime = regime_changes.iloc[i]['regime']
                regime_durations.append({'regime': regime, 'duration': duration})
            
            # 통계 계산 - 안전한 버전
            stats = {}
            for regime in ['BULL', 'BEAR']:
                regime_data = regime_history[regime_history['regime'] == regime]
                durations = [d['duration'] for d in regime_durations if d['regime'] == regime]
                
                stats[regime] = {
                    'total_days': len(regime_data),
                    'percentage': len(regime_data) / len(regime_history) * 100 if len(regime_history) > 0 else 0,
                    'avg_duration': safe_float_conversion(np.mean(durations) if durations else 0, 0),
                    'max_duration': safe_float_conversion(max(durations) if durations else 0, 0),
                    'min_duration': safe_float_conversion(min(durations) if durations else 0, 0),
                    'transitions': len([d for d in regime_durations if d['regime'] == regime])
                }
            
            # Risk-free rate 통계 추가
            if HAS_RF_UTILS and self.rf_manager:
                try:
                    rf_stats = self.rf_manager.get_risk_free_rate_stats(start_date, end_date)
                    stats['risk_free_rate'] = {
                        'ticker': self.rf_ticker,
                        'avg_rate': safe_float_conversion(rf_stats['mean_rate'], self.default_rf_rate * 100),
                        'min_rate': safe_float_conversion(rf_stats['min_rate'], self.default_rf_rate * 100),
                        'max_rate': safe_float_conversion(rf_stats['max_rate'], self.default_rf_rate * 100),
                        'std_rate': safe_float_conversion(rf_stats['std_rate'], 0),
                        'dynamic_used': True
                    }
                except:
                    stats['risk_free_rate'] = {
                        'ticker': self.rf_ticker,
                        'avg_rate': self.default_rf_rate * 100,
                        'dynamic_used': False
                    }
            else:
                stats['risk_free_rate'] = {
                    'ticker': 'Fixed',
                    'avg_rate': self.default_rf_rate * 100,
                    'dynamic_used': False
                }
            
            return stats
            
        except Exception as e:
            print(f"체제 통계 계산 중 오류: {e}")
            return None


# 편의 함수들
def create_jump_model_with_dynamic_rf(benchmark_ticker, benchmark_name, rf_ticker='^IRX', **kwargs):
    """동적 Risk-Free Rate를 사용하는 Jump Model 생성 편의 함수"""
    return UniversalJumpModel(
        benchmark_ticker=benchmark_ticker,
        benchmark_name=benchmark_name,
        rf_ticker=rf_ticker,
        **kwargs
    )

def analyze_multiple_markets_with_dynamic_rf(markets, rf_ticker='^IRX'):
    """여러 시장의 체제를 동적 Risk-Free Rate로 분석"""
    results = {}
    
    print(f"\n=== 다중 시장 분석 (동적 RF: {rf_ticker}) ===")
    
    for ticker, name in markets:
        try:
            jump_model = UniversalJumpModel(
                benchmark_ticker=ticker,
                benchmark_name=name,
                rf_ticker=rf_ticker,
                training_cutoff_date=datetime(2024, 12, 31)
            )
            
            current = jump_model.get_current_regime_with_training_cutoff()
            
            if current:
                results[name] = current
                
                oos_status = "🔮 Out-of-Sample" if current['is_out_of_sample'] else "📚 In-Sample"
                rf_status = "📊 Dynamic" if current['dynamic_rf_used'] else "📌 Fixed"
                
                print(f"\n{name}:")
                print(f"  체제: {current['regime']} (신뢰도: {current['confidence']:.2%})")
                print(f"  상태: {oos_status}")
                print(f"  RF: {rf_status} ({current['current_rf_rate']:.3f}%)")
                print(f"  날짜: {current['date'].strftime('%Y-%m-%d')}")
            else:
                print(f"\n{name}: 분석 실패")
                
        except Exception as e:
            print(f"\n{name}: 오류 - {e}")
    
    return results


# 사용 예시
if __name__ == "__main__":
    # S&P 500에 대한 Jump Model (2024년까지 학습, 동적 RF 사용)
    sp500_jump = UniversalJumpModel(
        benchmark_ticker='^GSPC',
        benchmark_name='S&P 500',
        jump_penalty=50.0,
        training_cutoff_date=datetime(2024, 12, 31),
        rf_ticker='^IRX'  # 미국 3개월물 금리
    )
    
    # 현재 체제 확인 (2024년까지 학습, 2025년은 추론, 동적 RF 사용)
    current = sp500_jump.get_current_regime_with_training_cutoff()
    if current:
        print(f"\nS&P 500 현재 체제: {current['regime']} (신뢰도: {current['confidence']:.2%})")
        print(f"분석 날짜: {current['date'].strftime('%Y-%m-%d')}")
        print(f"Out-of-Sample 예측: {current['is_out_of_sample']}")
        print(f"학습 마감일: {current['training_cutoff']}")
        print(f"Risk-Free Rate: {current['rf_ticker']} (현재: {current['current_rf_rate']:.3f}%)")
        print(f"동적 RF 사용: {current['dynamic_rf_used']}")
    
    # GLD 테스트 (금 ETF - 문제가 자주 발생하는 케이스)
    print("\n=== GLD (Gold ETF) 테스트 ===")
    gld_jump = UniversalJumpModel(
        benchmark_ticker='GLD',
        benchmark_name='SPDR Gold Trust',
        jump_penalty=30.0,  # 더 낮은 패널티
        training_cutoff_date=datetime(2024, 12, 31),
        rf_ticker='^IRX'
    )
    
    current_gld = gld_jump.get_current_regime_with_training_cutoff()
    if current_gld:
        print(f"GLD 현재 체제: {current_gld['regime']} (신뢰도: {current_gld['confidence']:.2%})")
        print(f"분석 날짜: {current_gld['date'].strftime('%Y-%m-%d')}")
        print(f"Out-of-Sample 예측: {current_gld['is_out_of_sample']}")
        print(f"Risk-Free Rate: {current_gld['rf_ticker']} (현재: {current_gld['current_rf_rate']:.3f}%)")
    else:
        print("GLD 분석 실패")
    
    print(f"\n=== Series Conversion 오류 수정 버전 테스트 완료 ===")
