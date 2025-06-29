import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class KospiJumpModel:
    """
    KOSPI 200 지수에 적용하는 Jump Model
    시장 체제(Bull/Bear)를 감지하여 투자 신호 생성
    """
    
    def __init__(self, n_states=2, lookback_window=20, jump_penalty=50.0):
        """
        Parameters:
        - n_states: 상태 수 (기본값: 2 - Bull/Bear)
        - lookback_window: 특징 계산을 위한 lookback 기간
        - jump_penalty: 체제 전환에 대한 페널티 (높을수록 지속성 증가)
        """
        self.n_states = n_states
        self.lookback_window = lookback_window
        self.jump_penalty = jump_penalty
        self.kospi200_ticker = '^GSPC'  # KOSPI 200 지수
        
        # 모델 파라미터
        self.cluster_centers = None
        self.scaler = StandardScaler()
        self.current_regime = None
        
    def download_kospi200_data(self, start_date, end_date):
        """KOSPI 200 데이터 다운로드"""
        try:
            print("KOSPI 200 데이터 다운로드 중...")
            data = yf.download(self.kospi200_ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                # 대안: KODEX 200 ETF 사용
                print("KOSPI 200 지수 데이터 없음. KODEX 200 ETF 사용...")
                data = yf.download('069500.KS', start=start_date, end=end_date, progress=False)
                
            return data
            
        except Exception as e:
            print(f"데이터 다운로드 실패: {e}")
            return None
    
    def calculate_features(self, price_data):
        """
        Jump Model을 위한 특징 계산
        - 수익률
        - 실현 변동성
        - 하방 변동성
        - 왜도
        """
        features_list = []
        
        # 일일 수익률
        returns = price_data['Close'].pct_change().dropna()
        
        # Rolling window로 특징 계산
        for i in range(self.lookback_window, len(returns)):
            window_returns = returns.iloc[i-self.lookback_window:i]
            
            # 1. 평균 수익률
            mean_return = window_returns.mean()
            
            # 2. 실현 변동성 (Realized Volatility)
            realized_vol = window_returns.std() * np.sqrt(252)
            
            # 3. 하방 변동성 (Downside Volatility)
            downside_returns = window_returns[window_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # 4. 왜도 (Skewness)
            skewness = window_returns.skew()
            
            # 5. 최대 낙폭 (Maximum Drawdown)
            cumulative = (1 + window_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 6. 상승/하락 일수 비율
            up_days_ratio = (window_returns > 0).sum() / len(window_returns)
            
            features_list.append({
                'date': returns.index[i],
                'mean_return': mean_return,
                'realized_vol': realized_vol,
                'downside_vol': downside_vol,
                'skewness': skewness,
                'max_drawdown': max_drawdown,
                'up_days_ratio': up_days_ratio
            })
        
        features_df = pd.DataFrame(features_list).set_index('date')
        return features_df
    
    def fit_jump_model(self, features_df):
        """Jump Model 학습"""
        # 특징 정규화
        X = features_df.values
        X_scaled = self.scaler.fit_transform(X)
        
        # 초기 클러스터링 (K-means)
        kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
        initial_states = kmeans.fit_predict(X_scaled)
        self.cluster_centers = kmeans.cluster_centers_
        
        # Jump penalty를 적용한 최적화
        optimized_states = self.optimize_with_jump_penalty(X_scaled, initial_states)
        
        # 상태별 특성 분석
        self.analyze_regimes(features_df, optimized_states)
        
        return optimized_states
    
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
            
            regime_stats[state] = {
                'count': len(state_features),
                'avg_return': state_features['mean_return'].mean(),
                'avg_volatility': state_features['realized_vol'].mean(),
                'avg_downside_vol': state_features['downside_vol'].mean(),
                'avg_drawdown': state_features['max_drawdown'].mean()
            }
        
        # Bear 상태 식별 (높은 변동성, 낮은 수익률)
        bear_state = max(regime_stats.keys(), 
                        key=lambda x: regime_stats[x]['avg_volatility'] - regime_stats[x]['avg_return'])
        
        # 상태 매핑 (0: Bull, 1: Bear)
        self.state_mapping = {}
        for state in range(self.n_states):
            if state == bear_state:
                self.state_mapping[state] = 'BEAR'
            else:
                self.state_mapping[state] = 'BULL'
        
        # 통계 출력
        print("\n=== 체제별 특성 ===")
        for state, stats in regime_stats.items():
            regime_type = self.state_mapping[state]
            print(f"\n{regime_type} 체제 (State {state}):")
            print(f"  - 기간 비율: {stats['count'] / len(features_df) * 100:.1f}%")
            print(f"  - 평균 수익률: {stats['avg_return']*252*100:.2f}%")
            print(f"  - 평균 변동성: {stats['avg_volatility']*100:.1f}%")
            print(f"  - 평균 하방 변동성: {stats['avg_downside_vol']*100:.1f}%")
            print(f"  - 평균 최대 낙폭: {stats['avg_drawdown']*100:.1f}%")
        
        return regime_stats
    
    def predict_regime(self, current_features):
        """현재 시장 체제 예측"""
        if self.cluster_centers is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 특징 정규화
        X = current_features.values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # 가장 가까운 클러스터 찾기
        distances = [np.linalg.norm(X_scaled - center) for center in self.cluster_centers]
        predicted_state = np.argmin(distances)
        
        # Jump penalty 고려 (이전 상태가 있는 경우)
        if self.current_regime is not None and predicted_state != self.current_regime:
            # 전환 비용이 충분히 큰 경우에만 체제 변경
            current_distance = distances[self.current_regime]
            new_distance = distances[predicted_state]
            
            if (current_distance - new_distance) < self.jump_penalty / 100:
                predicted_state = self.current_regime
        
        self.current_regime = predicted_state
        
        return self.state_mapping[predicted_state]
    
    def get_regime_history(self, start_date, end_date):
        """과거 체제 이력 계산"""
        # 데이터 다운로드
        price_data = self.download_kospi200_data(
            start_date - timedelta(days=self.lookback_window * 2),
            end_date
        )
        
        if price_data is None:
            return None
        
        # 특징 계산
        features_df = self.calculate_features(price_data)
        
        # 모델 학습
        states = self.fit_jump_model(features_df)
        
        # 체제 이력 생성
        regime_history = pd.DataFrame({
            'state': states,
            'regime': [self.state_mapping[s] for s in states]
        }, index=features_df.index)
        
        return regime_history[start_date:end_date]
    
    def get_current_regime(self):
        """현재 시장 체제 확인"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_window * 4)
        
        # 데이터 다운로드
        price_data = self.download_kospi200_data(start_date, end_date)
        
        if price_data is None:
            return None
        
        # 특징 계산
        features_df = self.calculate_features(price_data)
        
        if features_df.empty:
            return None
        
        # 모델이 학습되지 않은 경우 학습
        if self.cluster_centers is None:
            self.fit_jump_model(features_df)
        
        # 최신 특징으로 예측
        latest_features = features_df.iloc[-1]
        current_regime = self.predict_regime(latest_features)
        
        return {
            'regime': current_regime,
            'date': features_df.index[-1],
            'features': latest_features.to_dict()
        }

# 사용 예시
if __name__ == "__main__":
    # Jump Model 생성
    jump_model = KospiJumpModel(
        n_states=2,
        lookback_window=20,
        jump_penalty=50.0
    )
    
    # 1. 과거 체제 분석
    print("=== KOSPI 200 체제 분석 ===\n")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3년
    
    regime_history = jump_model.get_regime_history(start_date, end_date)
    
    if regime_history is not None:
        # 체제 전환 분석
        regime_changes = regime_history[regime_history['regime'] != regime_history['regime'].shift()]
        
        print(f"\n=== 체제 전환 이력 (최근 10개) ===")
        for date, row in regime_changes.tail(10).iterrows():
            print(f"{date.strftime('%Y-%m-%d')}: {row['regime']} 체제 시작")
        
        # 체제별 기간
        bull_days = (regime_history['regime'] == 'BULL').sum()
        bear_days = (regime_history['regime'] == 'BEAR').sum()
        
        print(f"\n체제별 기간:")
        print(f"- BULL: {bull_days}일 ({bull_days/len(regime_history)*100:.1f}%)")
        print(f"- BEAR: {bear_days}일 ({bear_days/len(regime_history)*100:.1f}%)")
    
    # 2. 현재 체제 확인
    print("\n=== 현재 시장 체제 ===")
    current = jump_model.get_current_regime()
    
    if current:
        print(f"현재 체제: {current['regime']}")
        print(f"기준일: {current['date'].strftime('%Y-%m-%d')}")
        print(f"현재 지표:")
        for key, value in current['features'].items():
            if 'vol' in key or 'drawdown' in key:
                print(f"  - {key}: {value*100:.2f}%")
            else:
                print(f"  - {key}: {value:.4f}")