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

class UniversalJumpModel:
    """
    범용 Jump Model with EWM Features (논문 Table 2 기준)
    논문: "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach"
    
    핵심 특징 (Table 2):
    1. Downside Deviation (halflife=10 days)
    2. Sortino Ratio (halflife=20 days) 
    3. Sortino Ratio (halflife=60 days)
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
        
        feature_type = "논문 정확한 3특징" if use_paper_features_only else "논문 기반 + 추가 특징"
        print(f"EWM Jump Model 초기화: {feature_type}")
        print(f"학습 마감일: {self.training_cutoff_date.strftime('%Y-%m-%d')}")
        print(f"Risk-Free Rate: {self.rf_ticker} (기본값: {self.default_rf_rate*100:.1f}%)")
    
    def download_benchmark_data(self, start_date, end_date):
        """벤치마크 데이터 다운로드"""
        try:
            print(f"{self.benchmark_name} 데이터 다운로드 중...")
            data = yf.download(self.benchmark_ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"{self.benchmark_name} 데이터를 가져올 수 없습니다.")
                
            data = data.dropna()
            return data
            
        except Exception as e:
            print(f"데이터 다운로드 실패: {e}")
            return None
    
    def calculate_features(self, price_data):
        """
        논문 기반 EWM 특징 계산
        Table 2: Downside Deviation (hl=10), Sortino Ratio (hl=20, 60)
        """
        returns = price_data['Close'].pct_change().dropna()
        
        if len(returns) == 0:
            print("가격 데이터에서 수익률을 계산할 수 없습니다.")
            return pd.DataFrame()
        
        # Risk-free rate 다운로드
        rf_data = None
        if HAS_RF_UTILS and self.rf_manager:
            try:
                start_date = returns.index[0]
                end_date = returns.index[-1]
                rf_data = self.rf_manager.download_risk_free_rate(start_date, end_date)
                print(f"Risk-free rate 데이터: {len(rf_data) if rf_data is not None else 0}개")
            except Exception as e:
                print(f"Risk-free rate 다운로드 실패: {e}")
                rf_data = None
        
        # Daily risk-free rate 계산
        if rf_data is not None:
            try:
                daily_rf_rates = rf_data.reindex(returns.index, method='ffill').fillna(self.default_rf_rate)
                daily_rf_rates = daily_rf_rates / 252  # 연율을 일별로 변환
            except:
                daily_rf_rates = pd.Series(self.default_rf_rate / 252, index=returns.index)
        else:
            daily_rf_rates = pd.Series(self.default_rf_rate / 252, index=returns.index)
        
        # 초과수익률 계산
        excess_returns = returns - daily_rf_rates
        
        if self.use_paper_features_only:
            return self._calculate_paper_features_only(excess_returns, rf_data)
        else:
            return self._calculate_enhanced_features(excess_returns, returns, rf_data)
    
    def _calculate_paper_features_only(self, excess_returns, rf_data):
        """논문 Table 2의 정확한 3가지 특징만 계산"""
        
        # 하방 수익률 (음수인 경우만)
        negative_excess_returns = excess_returns.where(excess_returns < 0, 0)
        
        # Feature 1: Downside Deviation (halflife=10)
        ewm_dd_var_10 = (negative_excess_returns ** 2).ewm(halflife=10, adjust=False).mean()
        downside_deviation_10 = np.sqrt(ewm_dd_var_10) * np.sqrt(252)
        
        # Feature 2: Sortino Ratio (halflife=20)
        ewm_mean_20 = excess_returns.ewm(halflife=20, adjust=False).mean() * 252
        ewm_dd_var_20 = (negative_excess_returns ** 2).ewm(halflife=20, adjust=False).mean()
        ewm_dd_20 = np.sqrt(ewm_dd_var_20) * np.sqrt(252)
        sortino_ratio_20 = ewm_mean_20 / (ewm_dd_20 + 1e-8)
        
        # Feature 3: Sortino Ratio (halflife=60)  
        ewm_mean_60 = excess_returns.ewm(halflife=60, adjust=False).mean() * 252
        ewm_dd_var_60 = (negative_excess_returns ** 2).ewm(halflife=60, adjust=False).mean()
        ewm_dd_60 = np.sqrt(ewm_dd_var_60) * np.sqrt(252)
        sortino_ratio_60 = ewm_mean_60 / (ewm_dd_60 + 1e-8)
        
        # DataFrame 생성
        features_df = pd.DataFrame({
            'downside_deviation_10': downside_deviation_10,
            'sortino_ratio_20': sortino_ratio_20,
            'sortino_ratio_60': sortino_ratio_60
        }, index=excess_returns.index)
        
        # 초기 NaN 제거 (60일 이후부터 안정)
        features_df = features_df.iloc[60:].copy()
        features_df = features_df.fillna(method='ffill').fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        print(f"논문 정확한 EWM 특징 계산 완료: {len(features_df)}개 (3 features)")
        return features_df
    
    def _calculate_enhanced_features(self, excess_returns, returns, rf_data):
        """논문 기반 + 추가 특징들 (EWM 적용)"""
        
        negative_excess_returns = excess_returns.where(excess_returns < 0, 0)
        
        # 논문 Table 2의 핵심 특징들 (EWM)
        
        # 1. EWM Downside Deviation (halflife=10)
        ewm_downside_var_10 = (negative_excess_returns ** 2).ewm(halflife=10, adjust=False).mean()
        ewm_downside_deviation_10 = np.sqrt(ewm_downside_var_10) * np.sqrt(252)
        
        # 2. EWM Sortino Ratios (halflife=20, 60)
        ewm_mean_excess_20 = excess_returns.ewm(halflife=20, adjust=False).mean() * 252
        ewm_mean_excess_60 = excess_returns.ewm(halflife=60, adjust=False).mean() * 252
        
        ewm_downside_var_20 = (negative_excess_returns ** 2).ewm(halflife=20, adjust=False).mean()
        ewm_downside_deviation_20 = np.sqrt(ewm_downside_var_20) * np.sqrt(252)
        
        ewm_downside_var_60 = (negative_excess_returns ** 2).ewm(halflife=60, adjust=False).mean()
        ewm_downside_deviation_60 = np.sqrt(ewm_downside_var_60) * np.sqrt(252)
        
        ewm_sortino_20 = ewm_mean_excess_20 / (ewm_downside_deviation_20 + 1e-8)
        ewm_sortino_60 = ewm_mean_excess_60 / (ewm_downside_deviation_60 + 1e-8)
        
        # 추가 특징들 (EWM 적용)
        
        # 3. EWM Realized Volatility
        ewm_variance_20 = (excess_returns ** 2).ewm(halflife=20, adjust=False).mean()
        ewm_realized_vol = np.sqrt(ewm_variance_20) * np.sqrt(252)
        
        # 4. EWM Skewness (rolling 적용)
        ewm_skewness = excess_returns.rolling(window=20, min_periods=10).skew()
        
        # 5. Maximum Drawdown (rolling)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown_20 = drawdown.rolling(window=20, min_periods=10).min()
        
        # 6. EWM Up Days Ratio
        up_days = (excess_returns > 0).astype(float)
        ewm_up_days_ratio = up_days.ewm(halflife=20, adjust=False).mean()
        
        # 7. EWM Volatility Ratio
        ewm_vol_ratio = ewm_downside_deviation_20 / (ewm_realized_vol + 1e-8)
        
        # 8. EWM RF Level
        if rf_data is not None:
            ewm_rf_level = rf_data.reindex(excess_returns.index, method='ffill').ewm(halflife=20, adjust=False).mean()
        else:
            ewm_rf_level = pd.Series(self.default_rf_rate, index=excess_returns.index)
        
        # DataFrame 생성
        features_df = pd.DataFrame({
            # 논문 Table 2의 핵심 특징들
            'downside_deviation_10': ewm_downside_deviation_10,      # Feature 1
            'sortino_ratio_20': ewm_sortino_20,                      # Feature 2
            'sortino_ratio_60': ewm_sortino_60,                      # Feature 3
            
            # 추가 특징들
            'realized_vol': ewm_realized_vol,
            'mean_excess_return_20': ewm_mean_excess_20,
            'skewness': ewm_skewness,
            'max_drawdown': max_drawdown_20,
            'up_days_ratio': ewm_up_days_ratio,
            'vol_ratio': ewm_vol_ratio,
            'rf_level': ewm_rf_level
        }, index=excess_returns.index)
        
        # 초기 NaN 제거 (60일 이후부터)
        features_df = features_df.iloc[60:].copy()
        features_df = features_df.fillna(method='ffill').fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        print(f"EWM 특징 계산 완료: {len(features_df)}개")
        print(f"핵심 특징: Downside Deviation (hl=10), Sortino Ratio (hl=20,60)")
        return features_df
    
    def fit_jump_model(self, features_df):
        """Jump Model 학습"""
        try:
            X = features_df.values
            
            # 무효한 값 확인 및 정리
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print("특징 데이터 정리 중...")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            X_scaled = self.scaler.fit_transform(X)
            
            # 초기 클러스터링
            kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            initial_states = kmeans.fit_predict(X_scaled)
            self.cluster_centers = kmeans.cluster_centers_
            
            # Jump penalty 적용 최적화
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
        """체제별 특성 분석 및 Bull/Bear 레이블링"""
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
    
    def predict_regime(self, current_features):
        """현재 시장 체제 예측"""
        if not self.is_trained or self.cluster_centers is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X = current_features.values.reshape(1, -1)
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
            return self.state_mapping[predicted_state], confidence
            
        except Exception as e:
            print(f"체제 예측 중 오류: {e}")
            return 'BULL', 0.5
    
    def train_model_with_cutoff(self, start_date=None, end_date=None):
        """특정 기간의 데이터로만 모델 학습"""
        if end_date is None:
            end_date = self.training_cutoff_date
        
        if start_date is None:
            start_date = end_date - timedelta(days=365*20)
        
        print(f"\nEWM 모델 학습: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # 학습용 데이터 다운로드 (EWM 안정화를 위해 더 많은 데이터 필요)
            price_data = self.download_benchmark_data(
                start_date - timedelta(days=100),  # EWM을 위한 추가 기간
                end_date
            )
            
            if price_data is None or price_data.empty:
                print(f"{self.benchmark_name} 학습 데이터를 가져올 수 없습니다.")
                return False
            
            # EWM 특징 계산
            features_df = self.calculate_features(price_data)
            
            if features_df.empty:
                print(f"{self.benchmark_name} EWM 특징 계산 실패")
                return False
            
            # 학습 기간으로 제한
            training_features = features_df[start_date:end_date]
            
            if len(training_features) < 100:  # EWM을 위한 최소 데이터
                print(f"학습 데이터 부족: {len(training_features)} < 100")
                return False
            
            print(f"EWM 학습 특징 수: {len(training_features)}")
            
            # 모델 학습
            result = self.fit_jump_model(training_features)
            return result is not None
            
        except Exception as e:
            print(f"EWM 모델 학습 중 예외: {e}")
            return False
    
    def get_current_regime_with_training_cutoff(self):
        """학습 마감일까지만 학습하고 현재 체제 예측"""
        try:
            if not self.is_trained:
                success = self.train_model_with_cutoff()
                if not success:
                    return None
            
            # 현재 시점까지의 데이터 가져오기
            current_date = datetime.now()
            inference_start = self.training_cutoff_date - timedelta(days=100)
            
            print(f"\nEWM 추론: {inference_start.strftime('%Y-%m-%d')} ~ {current_date.strftime('%Y-%m-%d')}")
            
            price_data = self.download_benchmark_data(inference_start, current_date)
            
            if price_data is None or price_data.empty:
                print(f"{self.benchmark_name} 추론 데이터를 가져올 수 없습니다.")
                return None
            
            # EWM 특징 계산
            features_df = self.calculate_features(price_data)
            
            if features_df.empty:
                print(f"{self.benchmark_name} EWM 추론 특징 계산 실패")
                return None
            
            # 최신 특징으로 예측
            latest_features = features_df.iloc[-1]
            current_regime, confidence = self.predict_regime(latest_features)
            
            latest_date = features_df.index[-1]
            is_out_of_sample = latest_date > self.training_cutoff_date
            
            # 분석 정보
            analysis_info = {
                'regime': current_regime,
                'confidence': safe_float_conversion(confidence, 0.5),
                'date': latest_date,
                'features': latest_features.to_dict(),
                'is_out_of_sample': is_out_of_sample,
                'training_cutoff': self.training_cutoff_date.strftime('%Y-%m-%d'),
                'feature_type': "논문 정확한 3특징" if self.use_paper_features_only else "논문 기반 + 추가",
                'ewm_applied': True,
                'rf_ticker': self.rf_ticker,
                'dynamic_rf_used': HAS_RF_UTILS and self.rf_manager is not None
            }
            
            # Risk-free rate 정보
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
            print(f"EWM 체제 분석 중 예외: {e}")
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
    print("=== EWM Jump Model (논문 버전) 테스트 ===")
    
    # 1. 논문 정확한 특징만 사용
    print("\n1. 논문 Table 2 정확한 특징 (3개)")
    sp500_paper = create_paper_exact_jump_model(
        benchmark_ticker='^GSPC',
        benchmark_name='S&P 500 (Paper Exact)',
        jump_penalty=50.0
    )
    
    paper_result = sp500_paper.get_current_regime_with_training_cutoff()
    if paper_result:
        print(f"체제: {paper_result['regime']} (신뢰도: {paper_result['confidence']:.2%})")
        print(f"EWM 적용: {paper_result['ewm_applied']}")
        print(f"특징 유형: {paper_result['feature_type']}")
    
    # 2. 논문 기반 + 추가 특징
    print("\n2. 논문 기반 + 추가 EWM 특징")
    sp500_enhanced = create_enhanced_ewm_jump_model(
        benchmark_ticker='^GSPC',
        benchmark_name='S&P 500 (Enhanced)',
        jump_penalty=50.0
    )
    
    enhanced_result = sp500_enhanced.get_current_regime_with_training_cutoff()
    if enhanced_result:
        print(f"체제: {enhanced_result['regime']} (신뢰도: {enhanced_result['confidence']:.2%})")
        print(f"EWM 적용: {enhanced_result['ewm_applied']}")
        print(f"특징 유형: {enhanced_result['feature_type']}")
    
    # 3. 비교 분석
    print("\n3. 특징 접근법 비교")
    compare_feature_approaches('^GSPC', 'S&P 500')
    
    print(f"\n=== EWM Jump Model 완성 ===")
    print("논문의 EWM 특징 (Downside Deviation, Sortino Ratio)이 정확히 적용되었습니다.")
