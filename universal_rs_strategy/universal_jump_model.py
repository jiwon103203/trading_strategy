"""
Jump Model - 간소화 버전
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from utils import safe_float, safe_extract_close, validate_data, clean_dataframe, print_status
import warnings
warnings.filterwarnings('ignore')

class UniversalJumpModel:
    """간소화된 Jump Model"""
    
    def __init__(self, benchmark_ticker, benchmark_name="Market", 
                 jump_penalty=50.0, training_cutoff_date=None):
        self.benchmark_ticker = benchmark_ticker
        self.benchmark_name = benchmark_name
        self.jump_penalty = jump_penalty
        self.training_cutoff_date = training_cutoff_date or datetime(2024, 12, 31)
        
        # 모델 상태
        self.cluster_centers = None
        self.scaler = StandardScaler()
        self.current_regime = None
        self.state_mapping = {0: 'BULL', 1: 'BEAR'}
        self.is_trained = False
        
        print_status(f"Jump Model 초기화: {benchmark_name}")
    
    def download_data(self, start_date, end_date):
        """데이터 다운로드"""
        try:
            data = yf.download(
                self.benchmark_ticker, 
                start=start_date - timedelta(days=100), 
                end=end_date,
                progress=False,
                auto_adjust=True,
                timeout=30
            )
            
            close_data = safe_extract_close(data)
            
            if not validate_data(close_data, 300):
                print_status(f"데이터 부족: {self.benchmark_ticker}", "ERROR")
                return None
            
            return close_data[start_date:end_date]
            
        except Exception as e:
            print_status(f"데이터 다운로드 실패: {e}", "ERROR")
            return None
    
    def calculate_features(self, price_data):
        """특징 계산 - 3가지 핵심 특징"""
        try:
            if not validate_data(price_data, 300):
                return pd.DataFrame()
            
            # 수익률 계산
            returns = price_data.pct_change().dropna()
            excess_returns = returns - 0.02/252  # 2% 고정 RF
            
            # 하방 수익률
            negative_returns = excess_returns.where(excess_returns < 0, 0)
            negative_squared = negative_returns * negative_returns
            
            # Feature 1: Downside Deviation (halflife=10)
            dd_var_10 = negative_squared.ewm(halflife=10, min_periods=20).mean()
            downside_dev_10 = np.sqrt(dd_var_10) * np.sqrt(252)
            
            # Feature 2: Sortino Ratio (halflife=20)
            mean_20 = excess_returns.ewm(halflife=20, min_periods=40).mean() * 252
            dd_20 = np.sqrt(negative_squared.ewm(halflife=20, min_periods=40).mean()) * np.sqrt(252)
            sortino_20 = mean_20 / (dd_20 + 1e-8)
            
            # Feature 3: Sortino Ratio (halflife=60)  
            mean_60 = excess_returns.ewm(halflife=60, min_periods=120).mean() * 252
            dd_60 = np.sqrt(negative_squared.ewm(halflife=60, min_periods=120).mean()) * np.sqrt(252)
            sortino_60 = mean_60 / (dd_60 + 1e-8)
            
            # DataFrame 생성
            features = pd.DataFrame({
                'downside_dev': downside_dev_10,
                'sortino_20': sortino_20,
                'sortino_60': sortino_60
            }, index=excess_returns.index)
            
            # 정리
            features = clean_dataframe(features, {
                'downside_dev': 0.1,
                'sortino_20': 1.0,
                'sortino_60': 1.0
            })
            
            # 이상값 제거
            features['downside_dev'] = features['downside_dev'].clip(0, 1.0)
            features['sortino_20'] = features['sortino_20'].clip(-10, 10)
            features['sortino_60'] = features['sortino_60'].clip(-10, 10)
            
            # 초기 불안정 기간 제거
            stable_start = max(120, len(features) // 4)
            if len(features) > stable_start:
                features = features.iloc[stable_start:]
            
            return features
            
        except Exception as e:
            print_status(f"특징 계산 실패: {e}", "ERROR")
            return pd.DataFrame()
    
    def fit_model(self, features_df):
        """모델 학습"""
        try:
            if features_df.empty or len(features_df) < 50:
                return False
            
            X = features_df.values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 스케일링
            X_scaled = self.scaler.fit_transform(X)
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            states = kmeans.fit_predict(X_scaled)
            self.cluster_centers = kmeans.cluster_centers_
            
            # Jump penalty 최적화
            states = self.optimize_states(X_scaled, states)
            
            # 체제 분석
            self.analyze_regimes(features_df, states)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print_status(f"모델 학습 실패: {e}", "ERROR")
            return False
    
    def optimize_states(self, X, initial_states):
        """Jump penalty 최적화"""
        try:
            states = initial_states.copy()
            n_samples = len(X)
            
            for _ in range(5):  # 간소화: 반복 횟수 감소
                for i in range(1, n_samples - 1):
                    best_state = states[i]
                    min_cost = float('inf')
                    
                    for new_state in range(2):
                        cluster_cost = np.linalg.norm(X[i] - self.cluster_centers[new_state]) ** 2
                        
                        jump_cost = 0
                        if new_state != states[i-1]:
                            jump_cost += self.jump_penalty
                        if new_state != states[i+1]:
                            jump_cost += self.jump_penalty
                        
                        total_cost = cluster_cost + jump_cost
                        
                        if total_cost < min_cost:
                            min_cost = total_cost
                            best_state = new_state
                    
                    states[i] = best_state
            
            return states
        except:
            return initial_states
    
    def analyze_regimes(self, features_df, states):
        """체제 분석"""
        try:
            regime_stats = {}
            
            for state in range(2):
                mask = (states == state)
                if mask.any():
                    state_features = features_df[mask]
                    regime_stats[state] = {
                        'count': len(state_features),
                        'downside_dev': state_features['downside_dev'].mean(),
                        'sortino_20': state_features['sortino_20'].mean(),
                        'sortino_60': state_features['sortino_60'].mean()
                    }
            
            # Bear 상태 식별 (높은 하방변동성, 낮은 Sortino)
            bear_scores = {}
            for state in regime_stats:
                score = (regime_stats[state]['downside_dev'] * 3 - 
                        regime_stats[state]['sortino_20'] - 
                        regime_stats[state]['sortino_60'])
                bear_scores[state] = score
            
            bear_state = max(bear_scores, key=bear_scores.get)
            
            self.state_mapping = {
                bear_state: 'BEAR',
                1 - bear_state: 'BULL'
            }
            
        except Exception as e:
            print_status(f"체제 분석 실패: {e}", "ERROR")
    
    def predict_regime(self, features):
        """체제 예측"""
        if not self.is_trained:
            return 'BULL', 0.5
        
        try:
            if isinstance(features, pd.Series):
                X = features.values.reshape(1, -1)
            else:
                X = np.array(features).reshape(1, -1)
            
            X = np.nan_to_num(X)
            X_scaled = self.scaler.transform(X)
            
            # 거리 계산
            distances = [np.linalg.norm(X_scaled - center) for center in self.cluster_centers]
            predicted_state = np.argmin(distances)
            
            # Jump penalty 고려
            if (self.current_regime is not None and 
                predicted_state != self.current_regime):
                
                current_dist = distances[self.current_regime]
                new_dist = distances[predicted_state]
                
                if (current_dist - new_dist) < self.jump_penalty / 100:
                    predicted_state = self.current_regime
            
            # 신뢰도 계산
            min_dist, max_dist = min(distances), max(distances)
            confidence = 1 - (min_dist / max_dist) if max_dist > min_dist else 0.5
            
            self.current_regime = predicted_state
            regime = self.state_mapping.get(predicted_state, 'BULL')
            
            return regime, max(0.0, min(1.0, confidence))
            
        except Exception as e:
            print_status(f"예측 실패: {e}", "ERROR")
            return 'BULL', 0.5
    
    def train_and_predict(self, start_date=None, end_date=None):
        """학습 후 현재 체제 예측"""
        try:
            # 학습 데이터
            if start_date is None:
                start_date = self.training_cutoff_date - timedelta(days=365*5)
            if end_date is None:
                end_date = self.training_cutoff_date
            
            # 학습
            train_data = self.download_data(start_date, end_date)
            if train_data is None:
                return None
            
            features = self.calculate_features(train_data)
            if features.empty:
                return None
            
            if not self.fit_model(features):
                return None
            
            # 현재 예측
            current_date = datetime.now()
            current_data = self.download_data(
                current_date - timedelta(days=200), 
                current_date
            )
            
            if current_data is None:
                return None
            
            current_features = self.calculate_features(current_data)
            if current_features.empty:
                return None
            
            latest_features = current_features.iloc[-1]
            regime, confidence = self.predict_regime(latest_features)
            
            return {
                'regime': regime,
                'confidence': confidence,
                'date': current_features.index[-1],
                'is_out_of_sample': current_features.index[-1] > self.training_cutoff_date,
                'features': {k: safe_float(v) for k, v in latest_features.items()}
            }
            
        except Exception as e:
            print_status(f"학습/예측 실패: {e}", "ERROR")
            return None
    
    def get_current_regime_with_training_cutoff(self):
        """호환성을 위한 래퍼 함수"""
        return self.train_and_predict()


# 테스트 함수
def test_jump_model():
    """간단한 테스트"""
    print_status("Jump Model 테스트 시작")
    
    model = UniversalJumpModel('SPY', 'S&P 500')
    result = model.train_and_predict()
    
    if result:
        print_status(f"테스트 성공: {result['regime']} ({result['confidence']:.2%})", "SUCCESS")
    else:
        print_status("테스트 실패", "ERROR")

if __name__ == "__main__":
    test_jump_model()
