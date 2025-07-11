"""
최적화된 Jump Model
효율적인 체제 감지 모델
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any
import warnings
warnings.filterwarnings('ignore')

from data_cache import data_cache
from parallel_downloader import SmartDataDownloader
from data_utils import data_validator, safe_float
from optimized_calculations import OptimizedCalculations
from config_manager import config_manager
from logger import get_logger
from memory_optimization import MemoryOptimizer

class OptimizedJumpModel:
    """최적화된 Jump Model"""
    
    def __init__(self, benchmark_ticker: str, benchmark_name: str = "Market",
                 n_states: int = 2, jump_penalty: float = 50.0,
                 use_paper_features_only: bool = True,
                 training_cutoff_date: Optional[datetime] = None,
                 cache_features: bool = True):
        """
        Parameters:
        - benchmark_ticker: 벤치마크 티커
        - benchmark_name: 벤치마크 이름
        - n_states: 상태 수 (기본 2: Bull/Bear)
        - jump_penalty: 체제 전환 페널티
        - use_paper_features_only: 논문 특징만 사용
        - training_cutoff_date: 학습 데이터 마감일
        - cache_features: 특징 계산 캐싱 여부
        """
        self.benchmark_ticker = benchmark_ticker
        self.benchmark_name = benchmark_name
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.use_paper_features_only = use_paper_features_only
        self.cache_features = cache_features
        
        # 학습 마감일 설정
        if training_cutoff_date is None:
            self.training_cutoff_date = datetime(2024, 12, 31)
        else:
            self.training_cutoff_date = training_cutoff_date
        
        # 모듈 초기화
        self.logger = get_logger(f"JumpModel_{benchmark_name}")
        self.downloader = SmartDataDownloader()
        self.calculator = OptimizedCalculations()
        self.memory_optimizer = MemoryOptimizer()
        
        # 모델 파라미터
        self.scaler = StandardScaler()
        self.cluster_centers = None
        self.state_mapping = None
        self.is_trained = False
        
        # 캐시
        self._feature_cache = {}
        self._regime_cache = {}
        
        self.logger.info(f"Jump Model 초기화: {benchmark_name}")
        self.logger.info(f"학습 마감일: {self.training_cutoff_date.strftime('%Y-%m-%d')}")
    
    @data_cache.disk_cache(expiry_hours=24, cache_subdir='jump_features')
    def calculate_features(self, price_data: pd.DataFrame, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """특징 계산 (캐시 적용)"""
        if price_data is None or price_data.empty:
            return pd.DataFrame()
        
        # 날짜 범위 설정
        if start_date:
            price_data = price_data[start_date:]
        if end_date:
            price_data = price_data[:end_date]
        
        self.logger.debug(f"Calculating features for {len(price_data)} days")
        
        # Close 가격 추출
        close_prices = data_validator.safe_extract_series(price_data, 'Close')
        if close_prices is None or len(close_prices) < 200:
            self.logger.warning("Insufficient data for feature calculation")
            return pd.DataFrame()
        
        # 수익률 계산
        returns = close_prices.pct_change().dropna()
        
        # Risk-free rate 차감 (config에서 가져오기)
        rf_rate = config_manager.get('default_rf_rate', 0.02)
        daily_rf = rf_rate / 252
        excess_returns = returns - daily_rf
        
        if self.use_paper_features_only:
            features_df = self._calculate_paper_features_optimized(excess_returns)
        else:
            features_df = self._calculate_enhanced_features_optimized(excess_returns, returns)
        
        # 메모리 최적화
        if not features_df.empty:
            features_df = self.memory_optimizer.optimize_dataframe(features_df, verbose=False)
        
        return features_df
    
    def _calculate_paper_features_optimized(self, excess_returns: pd.Series) -> pd.DataFrame:
        """논문 특징 계산 (최적화)"""
        # 하방 수익률
        negative_returns = excess_returns.where(excess_returns < 0, 0)
        negative_squared = negative_returns ** 2
        
        # 벡터화된 EWM 계산
        features_dict = {}
        
        # Feature 1: Downside Deviation (halflife=10)
        ewm_dd_var_10 = negative_squared.ewm(halflife=10, min_periods=20, adjust=False).mean()
        features_dict['downside_deviation_10'] = np.sqrt(ewm_dd_var_10) * np.sqrt(252)
        
        # Feature 2: Sortino Ratio (halflife=20)
        ewm_mean_20 = excess_returns.ewm(halflife=20, min_periods=40, adjust=False).mean() * 252
        ewm_dd_var_20 = negative_squared.ewm(halflife=20, min_periods=40, adjust=False).mean()
        ewm_dd_20 = np.sqrt(ewm_dd_var_20) * np.sqrt(252)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            features_dict['sortino_ratio_20'] = np.where(
                ewm_dd_20 > 0, 
                ewm_mean_20 / ewm_dd_20, 
                0
            )
        
        # Feature 3: Sortino Ratio (halflife=60)
        ewm_mean_60 = excess_returns.ewm(halflife=60, min_periods=120, adjust=False).mean() * 252
        ewm_dd_var_60 = negative_squared.ewm(halflife=60, min_periods=120, adjust=False).mean()
        ewm_dd_60 = np.sqrt(ewm_dd_var_60) * np.sqrt(252)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            features_dict['sortino_ratio_60'] = np.where(
                ewm_dd_60 > 0,
                ewm_mean_60 / ewm_dd_60,
                0
            )
        
        # DataFrame 생성
        features_df = pd.DataFrame(features_dict, index=excess_returns.index)
        
        # 데이터 정리
        features_df = self._clean_features_optimized(features_df)
        
        return features_df
    
    def _calculate_enhanced_features_optimized(self, excess_returns: pd.Series, 
                                             returns: pd.Series) -> pd.DataFrame:
        """확장 특징 계산 (최적화)"""
        # 기본 논문 특징
        features_df = self._calculate_paper_features_optimized(excess_returns)
        
        if features_df.empty:
            return features_df
        
        # 추가 특징들 (벡터화)
        # 변동성
        variance = (excess_returns ** 2).ewm(halflife=20, min_periods=20).mean()
        features_df['realized_vol'] = np.sqrt(variance) * np.sqrt(252)
        
        # 평균 초과수익률
        features_df['mean_excess_return_20'] = excess_returns.ewm(
            halflife=20, min_periods=20
        ).mean() * 252
        
        # 왜도 (롤링)
        features_df['skewness'] = returns.rolling(window=20, min_periods=10).skew()
        
        # 상승일 비율
        up_days = (excess_returns > 0).astype(float)
        features_df['up_days_ratio'] = up_days.ewm(halflife=20, min_periods=10).mean()
        
        # 최종 정리
        features_df = self._clean_features_optimized(features_df)
        
        return features_df
    
    def _clean_features_optimized(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """특징 데이터 정리 (최적화)"""
        if features_df.empty:
            return features_df
        
        # 무한대 및 NaN 처리
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill → Backward fill → 기본값
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # 컬럼별 기본값
        default_values = {
            'downside_deviation_10': 0.1,
            'sortino_ratio_20': 1.0,
            'sortino_ratio_60': 1.0,
            'realized_vol': 0.15,
            'mean_excess_return_20': 0.05,
            'skewness': 0.0,
            'up_days_ratio': 0.5
        }
        
        for col, default_val in default_values.items():
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(default_val)
        
        # 이상값 제한
        for col in features_df.columns:
            if col.startswith('downside_deviation') or col == 'realized_vol':
                features_df[col] = features_df[col].clip(lower=0, upper=1.0)
            elif col.startswith('sortino_ratio'):
                features_df[col] = features_df[col].clip(lower=-10, upper=10)
            elif col == 'skewness':
                features_df[col] = features_df[col].clip(lower=-3, upper=3)
            elif col == 'up_days_ratio':
                features_df[col] = features_df[col].clip(lower=0, upper=1)
        
        # 초기 불안정한 값 제거
        stable_start = max(120, len(features_df) // 4)
        if len(features_df) > stable_start:
            features_df = features_df.iloc[stable_start:].copy()
        
        return features_df
    
    def fit_jump_model(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Jump Model 학습 (최적화)"""
        if features_df.empty or len(features_df) < 50:
            self.logger.warning(f"Insufficient features for training: {len(features_df)}")
            return None
        
        with self.logger.timer("Model Training"):
            X = features_df.values
            
            # 데이터 정리
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 스케일링
            try:
                X_scaled = self.scaler.fit_transform(X)
            except Exception as e:
                self.logger.error(f"Scaling failed: {e}")
                return None
            
            # K-means 클러스터링
            try:
                kmeans = KMeans(
                    n_clusters=self.n_states, 
                    random_state=42, 
                    n_init=10, 
                    max_iter=300
                )
                initial_states = kmeans.fit_predict(X_scaled)
                self.cluster_centers = kmeans.cluster_centers_
            except Exception as e:
                self.logger.error(f"Clustering failed: {e}")
                return None
            
            # Jump penalty 최적화
            try:
                optimized_states = self._optimize_with_jump_penalty_vectorized(
                    X_scaled, initial_states
                )
            except Exception as e:
                self.logger.warning(f"Optimization failed: {e}")
                optimized_states = initial_states
            
            # 체제 분석
            self._analyze_regimes_optimized(features_df, optimized_states)
            
            self.is_trained = True
            self.logger.info("Model training completed")
            
            return optimized_states
    
    def _optimize_with_jump_penalty_vectorized(self, X: np.ndarray, 
                                             initial_states: np.ndarray) -> np.ndarray:
        """벡터화된 Jump penalty 최적화"""
        n_samples = len(X)
        states = initial_states.copy()
        
        # 최대 10회 반복
        for iteration in range(10):
            converged = True
            
            # 각 샘플에 대한 비용 계산 (벡터화)
            for i in range(1, n_samples - 1):
                # 각 상태에 대한 클러스터 비용
                cluster_costs = np.array([
                    np.linalg.norm(X[i] - self.cluster_centers[s]) ** 2
                    for s in range(self.n_states)
                ])
                
                # Jump 비용 계산
                jump_costs = np.zeros(self.n_states)
                for s in range(self.n_states):
                    if s != states[i-1]:
                        jump_costs[s] += self.jump_penalty
                    if i < n_samples - 1 and s != states[i+1]:
                        jump_costs[s] += self.jump_penalty
                
                # 총 비용
                total_costs = cluster_costs + jump_costs
                
                # 최적 상태 선택
                best_state = np.argmin(total_costs)
                
                if best_state != states[i]:
                    states[i] = best_state
                    converged = False
            
            if converged:
                break
        
        return states
    
    def _analyze_regimes_optimized(self, features_df: pd.DataFrame, 
                                  states: np.ndarray):
        """체제 분석 (최적화)"""
        regime_stats = {}
        
        # 각 상태별 특징 통계 (벡터화)
        for state in range(self.n_states):
            state_mask = (states == state)
            state_features = features_df[state_mask]
            
            if len(state_features) > 0:
                # 평균값 계산
                stats = {
                    'count': len(state_features),
                    'percentage': len(state_features) / len(features_df) * 100
                }
                
                # 각 특징별 평균
                for col in state_features.columns:
                    stats[f'avg_{col}'] = state_features[col].mean()
                
                regime_stats[state] = stats
            else:
                regime_stats[state] = {'count': 0, 'percentage': 0}
        
        # Bear 상태 식별 (높은 하방 변동성, 낮은 Sortino ratio)
        state_scores = {}
        for state in range(self.n_states):
            if regime_stats[state]['count'] > 0:
                bear_score = (
                    regime_stats[state].get('avg_downside_deviation_10', 0) * 3 -
                    regime_stats[state].get('avg_sortino_ratio_20', 0) * 2 -
                    regime_stats[state].get('avg_sortino_ratio_60', 0) * 2
                )
                state_scores[state] = bear_score
            else:
                state_scores[state] = 0
        
        # 상태 매핑
        bear_state = max(state_scores.keys(), key=lambda x: state_scores[x])
        
        self.state_mapping = {}
        for state in range(self.n_states):
            if state == bear_state:
                self.state_mapping[state] = 'BEAR'
            else:
                self.state_mapping[state] = 'BULL'
        
        # 로깅
        self.logger.info(f"=== {self.benchmark_name} 체제 분석 ===")
        for state, stats in regime_stats.items():
            regime_type = self.state_mapping[state]
            if stats['count'] > 0:
                self.logger.info(f"{regime_type} 체제: {stats['percentage']:.1f}%")
    
    def predict_regime(self, current_features: pd.Series) -> Tuple[str, float]:
        """현재 체제 예측"""
        if not self.is_trained or self.cluster_centers is None:
            raise ValueError("Model not trained")
        
        # 특징 준비
        if isinstance(current_features, pd.Series):
            X = current_features.values.reshape(1, -1)
        else:
            X = np.array(current_features).reshape(1, -1)
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 스케일링
        try:
            X_scaled = self.scaler.transform(X)
        except:
            # 스케일링 실패시 기본값
            return self.state_mapping.get(0, 'BULL'), 0.5
        
        # 각 클러스터까지의 거리
        distances = np.array([
            np.linalg.norm(X_scaled - center)
            for center in self.cluster_centers
        ])
        
        # 예측 상태
        predicted_state = np.argmin(distances)
        
        # 신뢰도 계산
        min_dist = distances.min()
        max_dist = distances.max()
        
        if max_dist > min_dist and max_dist > 0:
            confidence = 1 - (min_dist / max_dist)
            confidence = np.clip(confidence, 0, 1)
        else:
            confidence = 0.5
        
        regime = self.state_mapping.get(predicted_state, 'BULL')
        
        return regime, float(confidence)
    
    @data_cache.disk_cache(expiry_hours=1, cache_subdir='regime_cache')
    def get_current_regime(self, as_of_date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """현재 시장 체제 확인 (캐시 적용)"""
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # 모델 학습 확인
        if not self.is_trained:
            success = self.train_model()
            if not success:
                return None
        
        # 데이터 다운로드
        start_date = as_of_date - timedelta(days=500)
        
        data, _ = self.downloader.download_single_ticker(
            self.benchmark_ticker, start_date, as_of_date
        )
        
        if data is None or data.empty:
            self.logger.error("Failed to download data for regime analysis")
            return None
        
        # 특징 계산
        features_df = self.calculate_features(data)
        
        if features_df.empty:
            self.logger.error("Failed to calculate features")
            return None
        
        # 최신 특징
        latest_features = features_df.iloc[-1]
        latest_date = features_df.index[-1]
        
        # 체제 예측
        current_regime, confidence = self.predict_regime(latest_features)
        
        # Out-of-sample 여부
        is_oos = latest_date > self.training_cutoff_date
        
        return {
            'regime': current_regime,
            'confidence': confidence,
            'date': latest_date,
            'is_out_of_sample': is_oos,
            'training_cutoff': self.training_cutoff_date.strftime('%Y-%m-%d'),
            'features': {k: safe_float(v) for k, v in latest_features.to_dict().items()}
        }
    
    def train_model(self, start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> bool:
        """모델 학습"""
        if end_date is None:
            end_date = self.training_cutoff_date
        
        if start_date is None:
            start_date = end_date - timedelta(days=365*20)  # 20년
        
        self.logger.info(f"Training model: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 데이터 다운로드
        extended_start = start_date - timedelta(days=200)
        
        data, _ = self.downloader.download_single_ticker(
            self.benchmark_ticker, extended_start, end_date
        )
        
        if data is None or data.empty:
            self.logger.error("Failed to download training data")
            return False
        
        # 특징 계산
        features_df = self.calculate_features(data, start_date, end_date)
        
        if features_df.empty:
            self.logger.error("Failed to calculate training features")
            return False
        
        # 모델 학습
        result = self.fit_jump_model(features_df)
        
        return result is not None
    
    def get_regime_history(self, start_date: datetime, 
                          end_date: datetime) -> Optional[pd.DataFrame]:
        """기간별 체제 이력"""
        # 모델 학습 확인
        if not self.is_trained:
            success = self.train_model()
            if not success:
                return None
        
        # 데이터 다운로드
        extended_start = start_date - timedelta(days=200)
        
        data, _ = self.downloader.download_single_ticker(
            self.benchmark_ticker, extended_start, end_date
        )
        
        if data is None or data.empty:
            return None
        
        # 특징 계산
        features_df = self.calculate_features(data)
        
        if features_df.empty:
            return None
        
        # 기간 필터링
        features_df = features_df[start_date:end_date]
        
        if features_df.empty:
            return None
        
        # 각 날짜별 체제 예측
        regimes = []
        confidences = []
        
        for idx, features in features_df.iterrows():
            regime, confidence = self.predict_regime(features)
            regimes.append(regime)
            confidences.append(confidence)
        
        # DataFrame 생성
        regime_df = pd.DataFrame({
            'regime': regimes,
            'confidence': confidences,
            'state': [0 if r == 'BULL' else 1 for r in regimes]
        }, index=features_df.index)
        
        return regime_df

# 전역 Jump Model 관리자
class JumpModelManager:
    """Jump Model 관리자 (싱글톤)"""
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, ticker: str, name: Optional[str] = None, **kwargs) -> OptimizedJumpModel:
        """모델 가져오기 또는 생성"""
        if ticker not in self._models:
            if name is None:
                name = ticker
            self._models[ticker] = OptimizedJumpModel(ticker, name, **kwargs)
        return self._models[ticker]
    
    def clear_cache(self):
        """모든 모델 캐시 클리어"""
        for model in self._models.values():
            model._feature_cache.clear()
            model._regime_cache.clear()

# 전역 매니저
jump_model_manager = JumpModelManager()

# 편의 함수
def get_current_regime(ticker: str, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """현재 체제 확인 편의 함수"""
    model = jump_model_manager.get_model(ticker, name)
    return model.get_current_regime()

def analyze_multiple_regimes(tickers: List[str]) -> pd.DataFrame:
    """여러 시장의 체제 분석"""
    results = []
    
    for ticker in tickers:
        try:
            regime_info = get_current_regime(ticker)
            if regime_info:
                regime_info['ticker'] = ticker
                results.append(regime_info)
        except Exception as e:
            results.append({
                'ticker': ticker,
                'regime': 'ERROR',
                'confidence': 0,
                'error': str(e)
            })
    
    return pd.DataFrame(results)
