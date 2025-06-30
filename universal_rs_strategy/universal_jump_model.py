import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UniversalJumpModel:
    """
    범용 Jump Model with Training Cutoff Support
    다양한 지수에 적용 가능한 시장 체제(Bull/Bear) 감지
    2024년까지 학습, 2025년은 추론용
    """
    
    def __init__(self, benchmark_ticker, benchmark_name="Market", 
                 n_states=2, lookback_window=20, jump_penalty=50.0,
                 training_cutoff_date=None):
        """
        Parameters:
        - benchmark_ticker: 벤치마크 지수 티커 (예: '^GSPC', '069500.KS', 'URTH')
        - benchmark_name: 벤치마크 이름
        - n_states: 상태 수 (기본값: 2 - Bull/Bear)
        - lookback_window: 특징 계산을 위한 lookback 기간
        - jump_penalty: 체제 전환에 대한 페널티
        - training_cutoff_date: 학습 데이터 마지막 날짜 (None이면 전체 사용)
        """
        self.benchmark_ticker = benchmark_ticker
        self.benchmark_name = benchmark_name
        self.n_states = n_states
        self.lookback_window = lookback_window
        self.jump_penalty = jump_penalty
        self.training_cutoff_date = training_cutoff_date
        
        # 기본 학습 마감일을 2024년 12월 31일로 설정
        if training_cutoff_date is None:
            self.training_cutoff_date = datetime(2024, 12, 31)
        
        # 모델 파라미터
        self.cluster_centers = None
        self.scaler = StandardScaler()
        self.current_regime = None
        self.state_mapping = None
        self.is_trained = False
        
        print(f"Jump Model 초기화: 학습 마감일 = {self.training_cutoff_date.strftime('%Y-%m-%d')}")
    
    def download_benchmark_data(self, start_date, end_date):
        """벤치마크 데이터 다운로드"""
        try:
            print(f"{self.benchmark_name} 데이터 다운로드 중...")
            data = yf.download(self.benchmark_ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"{self.benchmark_name} 데이터를 가져올 수 없습니다.")
                
            return data
            
        except Exception as e:
            print(f"데이터 다운로드 실패: {e}")
            return None
    
    def calculate_features(self, price_data):
        """
        Jump Model을 위한 특징 계산
        """
        features_list = []
        
        # 일일 수익률
        returns = price_data['Close'].pct_change().dropna()
        
        # Rolling window로 특징 계산
        for i in range(self.lookback_window, len(returns)):
            window_returns = returns.iloc[i-self.lookback_window:i]
            
            # 1. 평균 수익률
            mean_return = float(window_returns.mean())
            
            # 2. 실현 변동성 (Realized Volatility) - 명시적으로 float로 변환
            realized_vol = float(window_returns.std()) * np.sqrt(252)
            
            # 3. 하방 변동성 (Downside Volatility)
            downside_returns = window_returns[window_returns < 0]
            if len(downside_returns) > 0:
                downside_vol = float(downside_returns.std()) * np.sqrt(252)
            else:
                downside_vol = 0.0
            
            # 4. 왜도 (Skewness)
            try:
                skewness = float(window_returns.skew())
                if pd.isna(skewness):
                    skewness = 0.0
            except:
                skewness = 0.0
            
            # 5. 최대 낙폭 (Maximum Drawdown)
            try:
                cumulative = (1 + window_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = float(drawdown.min())
            except:
                max_drawdown = 0.0
            
            # 6. 상승/하락 일수 비율
            up_days_ratio = float((window_returns > 0).sum()) / len(window_returns)
            
            # 7. 변동성 비율 (Volatility Ratio) - 수정된 부분
            if realized_vol > 0:
                vol_ratio = downside_vol / realized_vol
            else:
                vol_ratio = 1.0
            
            features_list.append({
                'date': returns.index[i],
                'mean_return': mean_return,
                'realized_vol': realized_vol,
                'downside_vol': downside_vol,
                'skewness': skewness,
                'max_drawdown': max_drawdown,
                'up_days_ratio': up_days_ratio,
                'vol_ratio': vol_ratio
            })
        
        features_df = pd.DataFrame(features_list).set_index('date')
        
        # NaN 값 처리
        features_df = features_df.fillna(0)
        
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
        
        self.is_trained = True
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
            
            if len(state_features) > 0:
                regime_stats[state] = {
                    'count': len(state_features),
                    'avg_return': float(state_features['mean_return'].mean()),
                    'avg_volatility': float(state_features['realized_vol'].mean()),
                    'avg_downside_vol': float(state_features['downside_vol'].mean()),
                    'avg_drawdown': float(state_features['max_drawdown'].mean()),
                    'avg_up_days': float(state_features['up_days_ratio'].mean()),
                    'avg_vol_ratio': float(state_features['vol_ratio'].mean())
                }
            else:
                regime_stats[state] = {
                    'count': 0,
                    'avg_return': 0.0,
                    'avg_volatility': 0.0,
                    'avg_downside_vol': 0.0,
                    'avg_drawdown': 0.0,
                    'avg_up_days': 0.0,
                    'avg_vol_ratio': 1.0
                }
        
        # Bear 상태 식별 (점수 기반)
        state_scores = {}
        for state in range(self.n_states):
            # Bear 점수: 높은 변동성, 낮은 수익률, 높은 하방 변동성
            bear_score = (
                regime_stats[state]['avg_volatility'] * 2 +
                regime_stats[state]['avg_downside_vol'] * 2 +
                abs(regime_stats[state]['avg_drawdown']) * 3 -
                regime_stats[state]['avg_return'] * 100 -
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
        
        # 통계 출력
        print(f"\n=== {self.benchmark_name} 체제별 특성 (학습기간: ~{self.training_cutoff_date.strftime('%Y-%m-%d')}) ===")
        for state, stats in regime_stats.items():
            regime_type = self.state_mapping[state]
            if stats['count'] > 0:
                print(f"\n{regime_type} 체제 (State {state}):")
                print(f"  - 기간 비율: {stats['count'] / len(features_df) * 100:.1f}%")
                print(f"  - 평균 수익률: {stats['avg_return']*252*100:.2f}%")
                print(f"  - 평균 변동성: {stats['avg_volatility']*100:.1f}%")
                print(f"  - 평균 하방 변동성: {stats['avg_downside_vol']*100:.1f}%")
                print(f"  - 평균 최대 낙폭: {stats['avg_drawdown']*100:.1f}%")
                print(f"  - 평균 상승일 비율: {stats['avg_up_days']*100:.1f}%")
        
        return regime_stats
    
    def predict_regime(self, current_features):
        """현재 시장 체제 예측"""
        if not self.is_trained or self.cluster_centers is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 특징 정규화
        X = current_features.values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # 가장 가까운 클러스터 찾기
        distances = [np.linalg.norm(X_scaled - center) for center in self.cluster_centers]
        predicted_state = np.argmin(distances)
        
        # Jump penalty 고려
        if self.current_regime is not None and predicted_state != self.current_regime:
            current_distance = distances[self.current_regime]
            new_distance = distances[predicted_state]
            
            if (current_distance - new_distance) < self.jump_penalty / 100:
                predicted_state = self.current_regime
        
        self.current_regime = predicted_state
        
        # 신뢰도 계산
        if max(distances) > 0:
            confidence = 1 - (min(distances) / max(distances))
        else:
            confidence = 1.0
        
        return self.state_mapping[predicted_state], confidence
    
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
        
        # 학습용 데이터 다운로드
        price_data = self.download_benchmark_data(
            start_date - timedelta(days=self.lookback_window * 2),
            end_date
        )
        
        if price_data is None or price_data.empty:
            print(f"{self.benchmark_name} 학습 데이터를 가져올 수 없습니다.")
            return False
        
        # 특징 계산
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
        self.fit_jump_model(training_features)
        
        return True
    
    def get_current_regime_with_training_cutoff(self):
        """
        학습 마감일까지만 학습하고 현재 체제 예측
        2024년까지 학습, 2025년은 추론용
        """
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
        
        # 특징 계산
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
        
        return {
            'regime': current_regime,
            'confidence': confidence,
            'date': latest_date,
            'features': latest_features.to_dict(),
            'is_out_of_sample': is_out_of_sample,
            'training_cutoff': self.training_cutoff_date.strftime('%Y-%m-%d')
        }
    
    def get_regime_history(self, start_date, end_date):
        """과거 체제 이력 계산"""
        # 데이터 다운로드
        price_data = self.download_benchmark_data(
            start_date - timedelta(days=self.lookback_window * 2),
            end_date
        )
        
        if price_data is None or price_data.empty:
            print(f"{self.benchmark_name} 데이터를 가져올 수 없습니다.")
            return None
        
        # 특징 계산
        features_df = self.calculate_features(price_data)
        
        if features_df.empty:
            print(f"{self.benchmark_name} 특징 계산 실패")
            return None
        
        # 모델 학습
        states = self.fit_jump_model(features_df)
        
        # 체제 이력 생성
        regime_history = pd.DataFrame({
            'state': states,
            'regime': [self.state_mapping[s] for s in states]
        }, index=features_df.index)
        
        return regime_history[start_date:end_date]
    
    def get_current_regime(self):
        """현재 시장 체제 확인 (기존 호환성 유지)"""
        return self.get_current_regime_with_training_cutoff()
    
    def get_regime_statistics(self, start_date, end_date):
        """체제별 상세 통계"""
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
        
        # 통계 계산
        stats = {}
        for regime in ['BULL', 'BEAR']:
            regime_data = regime_history[regime_history['regime'] == regime]
            durations = [d['duration'] for d in regime_durations if d['regime'] == regime]
            
            stats[regime] = {
                'total_days': len(regime_data),
                'percentage': len(regime_data) / len(regime_history) * 100,
                'avg_duration': np.mean(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'transitions': len([d for d in regime_durations if d['regime'] == regime])
            }
        
        return stats


# 사용 예시
if __name__ == "__main__":
    # S&P 500에 대한 Jump Model (2024년까지 학습)
    sp500_jump = UniversalJumpModel(
        benchmark_ticker='^GSPC',
        benchmark_name='S&P 500',
        jump_penalty=50.0,
        training_cutoff_date=datetime(2024, 12, 31)
    )
    
    # 현재 체제 확인 (2024년까지 학습, 2025년은 추론)
    current = sp500_jump.get_current_regime_with_training_cutoff()
    if current:
        print(f"\nS&P 500 현재 체제: {current['regime']} (신뢰도: {current['confidence']:.2%})")
        print(f"분석 날짜: {current['date'].strftime('%Y-%m-%d')}")
        print(f"Out-of-Sample 예측: {current['is_out_of_sample']}")
        print(f"학습 마감일: {current['training_cutoff']}")
    
    # KOSPI에 대한 Jump Model
    kospi_jump = UniversalJumpModel(
        benchmark_ticker='069500.KS',
        benchmark_name='KOSPI 200',
        jump_penalty=50.0,
        training_cutoff_date=datetime(2024, 12, 31)
    )
    
    # 현재 체제 확인
    current_kospi = kospi_jump.get_current_regime_with_training_cutoff()
    if current_kospi:
        print(f"\nKOSPI 200 현재 체제: {current_kospi['regime']} (신뢰도: {current_kospi['confidence']:.2%})")
        print(f"분석 날짜: {current_kospi['date'].strftime('%Y-%m-%d')}")
        print(f"Out-of-Sample 예측: {current_kospi['is_out_of_sample']}")
        print(f"학습 마감일: {current_kospi['training_cutoff']}")
