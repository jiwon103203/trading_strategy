"""
데이터 유틸리티
데이터 검증, 변환, 정리를 위한 유틸리티 함수
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Tuple, Any
import logging
from datetime import datetime

class DataValidator:
    """데이터 검증 및 변환 유틸리티"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def safe_extract_series(data: Union[pd.DataFrame, pd.Series, None], 
                          column: str = 'Close') -> Optional[pd.Series]:
        """안전한 Series 추출"""
        if data is None:
            return None
        
        # 이미 Series인 경우
        if isinstance(data, pd.Series):
            return data if len(data) > 0 else None
        
        # DataFrame인 경우
        if isinstance(data, pd.DataFrame):
            if len(data) == 0:
                return None
            
            # MultiIndex 컬럼 처리
            if isinstance(data.columns, pd.MultiIndex):
                if column in data.columns.get_level_values(0):
                    # 첫 번째 티커의 데이터 반환
                    return data[column].iloc[:, 0] if data[column].shape[1] > 0 else None
            
            # 일반 컬럼
            if column in data.columns:
                return data[column]
            
            # Close가 없으면 첫 번째 컬럼 반환
            if len(data.columns) > 0:
                return data.iloc[:, 0]
        
        return None
    
    @staticmethod
    def safe_float_conversion(value: Any, default: float = 0.0) -> float:
        """안전한 float 변환"""
        try:
            # None이나 NaN 처리
            if value is None or pd.isna(value):
                return default
            
            # Series나 DataFrame 처리
            if isinstance(value, (pd.Series, pd.DataFrame)):
                if len(value) > 0:
                    # 마지막 값 추출
                    val = value.iloc[-1] if hasattr(value, 'iloc') else value
                    # 재귀적으로 변환
                    return DataValidator.safe_float_conversion(val, default)
                return default
            
            # 리스트나 배열 처리
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 0:
                    return DataValidator.safe_float_conversion(value[-1], default)
                return default
            
            # 직접 변환
            return float(value)
            
        except (ValueError, TypeError, AttributeError):
            return default
    
    @staticmethod
    def validate_data_length(data: Union[pd.Series, pd.DataFrame, None], 
                           min_length: int) -> bool:
        """데이터 길이 검증"""
        if data is None:
            return False
        
        try:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                return len(data) >= min_length
            return False
        except:
            return False
    
    @staticmethod
    def align_data(*data_objects: Union[pd.Series, pd.DataFrame]) -> List[Union[pd.Series, pd.DataFrame]]:
        """여러 데이터 객체의 인덱스 정렬"""
        if not data_objects:
            return []
        
        # 모든 데이터의 공통 인덱스 찾기
        common_index = None
        for data in data_objects:
            if data is not None and len(data) > 0:
                if common_index is None:
                    common_index = data.index
                else:
                    common_index = common_index.intersection(data.index)
        
        if common_index is None or len(common_index) == 0:
            return [None] * len(data_objects)
        
        # 공통 인덱스로 정렬
        aligned = []
        for data in data_objects:
            if data is not None:
                aligned.append(data.loc[common_index])
            else:
                aligned.append(None)
        
        return aligned
    
    @staticmethod
    def clean_price_data(data: pd.DataFrame, forward_fill: bool = True) -> pd.DataFrame:
        """가격 데이터 정리"""
        if data is None or data.empty:
            return pd.DataFrame()
        
        # 복사본 생성
        cleaned = data.copy()
        
        # 0이나 음수 값 제거
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in cleaned.columns:
                cleaned.loc[cleaned[col] <= 0, col] = np.nan
        
        # NaN 처리
        if forward_fill:
            cleaned = cleaned.fillna(method='ffill').fillna(method='bfill')
        else:
            cleaned = cleaned.dropna()
        
        # 중복 인덱스 제거
        if cleaned.index.duplicated().any():
            cleaned = cleaned[~cleaned.index.duplicated(keep='first')]
        
        # 인덱스 정렬
        cleaned = cleaned.sort_index()
        
        return cleaned
    
    @staticmethod
    def resample_data(data: Union[pd.Series, pd.DataFrame], 
                     freq: str = 'D') -> Union[pd.Series, pd.DataFrame]:
        """데이터 리샘플링"""
        if data is None or len(data) == 0:
            return data
        
        # 리샘플링 규칙
        if isinstance(data, pd.Series):
            return data.resample(freq).last()
        
        elif isinstance(data, pd.DataFrame):
            # OHLCV 데이터 처리
            agg_rules = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            
            # 존재하는 컬럼만 처리
            rules = {col: rule for col, rule in agg_rules.items() if col in data.columns}
            
            if rules:
                return data.resample(freq).agg(rules)
            else:
                return data.resample(freq).last()
        
        return data

class DataConverter:
    """데이터 변환 유틸리티"""
    
    @staticmethod
    def series_to_dataframe(series_dict: dict, column_name: str = 'value') -> pd.DataFrame:
        """여러 Series를 DataFrame으로 변환"""
        if not series_dict:
            return pd.DataFrame()
        
        # 모든 시리즈의 인덱스 수집
        all_dates = set()
        for series in series_dict.values():
            if series is not None and len(series) > 0:
                all_dates.update(series.index)
        
        if not all_dates:
            return pd.DataFrame()
        
        # 정렬된 인덱스
        date_index = sorted(list(all_dates))
        
        # DataFrame 생성
        df = pd.DataFrame(index=date_index)
        for name, series in series_dict.items():
            if series is not None and len(series) > 0:
                df[name] = series
        
        return df
    
    @staticmethod
    def calculate_returns(prices: Union[pd.Series, pd.DataFrame], 
                         method: str = 'simple') -> Union[pd.Series, pd.DataFrame]:
        """수익률 계산"""
        if prices is None or len(prices) == 0:
            return prices
        
        if method == 'simple':
            return prices.pct_change()
        elif method == 'log':
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown return calculation method: {method}")
    
    @staticmethod
    def normalize_data(data: Union[pd.Series, pd.DataFrame], 
                      method: str = 'minmax') -> Union[pd.Series, pd.DataFrame]:
        """데이터 정규화"""
        if data is None or len(data) == 0:
            return data
        
        if method == 'minmax':
            min_val = data.min()
            max_val = data.max()
            range_val = max_val - min_val
            
            # 0으로 나누기 방지
            if isinstance(range_val, pd.Series):
                range_val[range_val == 0] = 1
            elif range_val == 0:
                range_val = 1
            
            return (data - min_val) / range_val
        
        elif method == 'zscore':
            mean = data.mean()
            std = data.std()
            
            # 0으로 나누기 방지
            if isinstance(std, pd.Series):
                std[std == 0] = 1
            elif std == 0:
                std = 1
            
            return (data - mean) / std
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")

class DataQualityChecker:
    """데이터 품질 체크"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_data_quality(self, data: pd.DataFrame) -> dict:
        """데이터 품질 종합 체크"""
        if data is None or data.empty:
            return {'status': 'empty', 'issues': ['No data']}
        
        issues = []
        warnings = []
        
        # 1. NaN 체크
        nan_counts = data.isna().sum()
        if nan_counts.any():
            for col, count in nan_counts[nan_counts > 0].items():
                pct = count / len(data) * 100
                if pct > 50:
                    issues.append(f"{col}: {pct:.1f}% NaN values")
                elif pct > 10:
                    warnings.append(f"{col}: {pct:.1f}% NaN values")
        
        # 2. 중복 인덱스 체크
        if data.index.duplicated().any():
            dup_count = data.index.duplicated().sum()
            issues.append(f"{dup_count} duplicate indices")
        
        # 3. 날짜 갭 체크 (시계열 데이터인 경우)
        if isinstance(data.index, pd.DatetimeIndex):
            date_diff = data.index.to_series().diff()
            median_diff = date_diff.median()
            large_gaps = date_diff[date_diff > median_diff * 5]
            
            if len(large_gaps) > 0:
                warnings.append(f"{len(large_gaps)} large date gaps detected")
        
        # 4. 이상치 체크 (가격 데이터인 경우)
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                # 일일 변동률 체크
                returns = data[col].pct_change()
                extreme_returns = returns[returns.abs() > 0.5]  # 50% 이상 변동
                
                if len(extreme_returns) > 0:
                    warnings.append(f"{col}: {len(extreme_returns)} extreme price movements")
        
        # 5. 데이터 길이 체크
        if len(data) < 20:
            issues.append(f"Insufficient data length: {len(data)} rows")
        
        # 결과 정리
        if issues:
            status = 'critical'
        elif warnings:
            status = 'warning'
        else:
            status = 'good'
        
        return {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'row_count': len(data),
            'column_count': len(data.columns),
            'date_range': f"{data.index[0]} to {data.index[-1]}" if len(data) > 0 else "N/A"
        }
    
    def generate_quality_report(self, data_dict: dict) -> pd.DataFrame:
        """여러 데이터의 품질 리포트 생성"""
        reports = []
        
        for name, data in data_dict.items():
            report = self.check_data_quality(data)
            report['name'] = name
            reports.append(report)
        
        return pd.DataFrame(reports)

# Singleton 인스턴스
data_validator = DataValidator()
data_converter = DataConverter()
quality_checker = DataQualityChecker()

# 편의 함수
def validate_and_clean(data: pd.DataFrame, min_length: int = 20) -> Optional[pd.DataFrame]:
    """데이터 검증 및 정리 편의 함수"""
    if not data_validator.validate_data_length(data, min_length):
        return None
    
    return data_validator.clean_price_data(data)

def safe_float(value: Any, default: float = 0.0) -> float:
    """안전한 float 변환 편의 함수"""
    return data_validator.safe_float_conversion(value, default)
