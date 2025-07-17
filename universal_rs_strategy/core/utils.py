"""
공통 유틸리티 함수들 - 간소화 버전
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def safe_float(value, default=0.0):
    """안전한 float 변환"""
    try:
        if pd.isna(value):
            return default
        if isinstance(value, pd.Series):
            return float(value.iloc[-1]) if len(value) > 0 else default
        return float(value)
    except:
        return default

def safe_extract_close(data):
    """Close 데이터 안전 추출"""
    try:
        if data is None or (hasattr(data, 'empty') and data.empty):
            return None
        
        if isinstance(data, pd.Series):
            return data
        
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                close_data = data['Close']
                return close_data.iloc[:, 0] if isinstance(close_data, pd.DataFrame) else close_data
            elif len(data.columns) > 0:
                return data.iloc[:, 0]
        
        return None
    except:
        return None

def validate_data(data, min_length=50):
    """데이터 유효성 검사"""
    try:
        if data is None:
            return False
        if hasattr(data, '__len__'):
            return len(data) >= min_length
        return False
    except:
        return False

def calculate_wma(data, period):
    """가중이동평균 계산"""
    try:
        if not validate_data(data, period):
            return pd.Series(dtype=float)
        
        weights = np.arange(1, period + 1, dtype=float)
        weight_sum = weights.sum()
        
        result = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(data)):
            if i < period - 1:
                result.iloc[i] = np.nan
            else:
                window = data.iloc[i-period+1:i+1].values
                if not np.any(np.isnan(window)):
                    result.iloc[i] = np.dot(window, weights) / weight_sum
        
        return result
    except:
        return pd.Series(dtype=float)

def clean_dataframe(df, default_values=None):
    """데이터프레임 정리"""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        # 무한대/NaN 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 기본값 적용
        if default_values:
            for col, default_val in default_values.items():
                if col in df.columns:
                    df[col] = df[col].fillna(default_val)
        
        return df
    except:
        return pd.DataFrame()

def calculate_basic_metrics(portfolio_df):
    """기본 성과 지표 계산"""
    try:
        if portfolio_df.empty:
            return {}
        
        total_return = (portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0] - 1) * 100
        
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = (np.power(1 + total_return/100, 1/years) - 1) * 100 if years > 0 else 0
        
        returns = portfolio_df['value'].pct_change().dropna()
        annual_volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
        
        sharpe = (annual_return - 2.0) / annual_volatility if annual_volatility > 0 else 0  # 2% 고정 RF
        
        return {
            '총 수익률': f"{total_return:.2f}%",
            '연율화 수익률': f"{annual_return:.2f}%",
            '연율화 변동성': f"{annual_volatility:.2f}%",
            '샤프 비율': f"{sharpe:.2f}"
        }
    except:
        return {}

def print_status(message, level="INFO"):
    """상태 출력"""
    prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "ℹ️")
    print(f"{prefix} {message}")
