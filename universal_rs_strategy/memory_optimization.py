"""
메모리 최적화 유틸리티
대용량 데이터 처리를 위한 메모리 효율화
"""

import pandas as pd
import numpy as np
import gc
import psutil
import os
from typing import Union, Iterator, Optional, Dict, Any
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

class MemoryOptimizer:
    """메모리 최적화 유틸리티"""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """DataFrame 메모리 사용량 최적화"""
        if df.empty:
            return df
        
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # 복사본 생성 (원본 보존)
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            # 숫자형 최적화
            if col_type != 'object':
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                
                # Integer 최적화
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        optimized_df[col] = optimized_df[col].astype(np.int64)
                
                # Float 최적화
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        optimized_df[col] = optimized_df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized_df[col] = optimized_df[col].astype(np.float32)
                    else:
                        optimized_df[col] = optimized_df[col].astype(np.float64)
            
            # Object 타입 최적화 (카테고리 변환)
            else:
                num_unique_values = len(optimized_df[col].unique())
                num_total_values = len(optimized_df[col])
                
                # 유니크 값이 50% 미만인 경우 카테고리로 변환
                if num_unique_values / num_total_values < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')
        
        final_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
        
        if verbose:
            print(f"Memory usage optimization:")
            print(f"  Initial: {initial_memory:.2f} MB")
            print(f"  Final: {final_memory:.2f} MB")
            print(f"  Reduction: {(1 - final_memory/initial_memory) * 100:.1f}%")
        
        return optimized_df
    
    @staticmethod
    def optimize_series(series: pd.Series) -> pd.Series:
        """Series 메모리 최적화"""
        if series.dtype == 'object':
            # 카테고리 변환 검토
            if len(series.unique()) / len(series) < 0.5:
                return series.astype('category')
            return series
        
        # 숫자형 최적화
        s_min = series.min()
        s_max = series.max()
        
        if pd.api.types.is_integer_dtype(series):
            if s_min > np.iinfo(np.int8).min and s_max < np.iinfo(np.int8).max:
                return series.astype(np.int8)
            elif s_min > np.iinfo(np.int16).min and s_max < np.iinfo(np.int16).max:
                return series.astype(np.int16)
            elif s_min > np.iinfo(np.int32).min and s_max < np.iinfo(np.int32).max:
                return series.astype(np.int32)
        else:
            if s_min > np.finfo(np.float32).min and s_max < np.finfo(np.float32).max:
                return series.astype(np.float32)
        
        return series
    
    @staticmethod
    def chunk_processing(df: pd.DataFrame, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """청크 단위 처리를 위한 이터레이터"""
        num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            
            chunk = df.iloc[start_idx:end_idx]
            yield chunk
            
            # 명시적 가비지 컬렉션
            if i % 10 == 0:
                gc.collect()
    
    @staticmethod
    def process_large_file(filename: str, chunk_size: int = 10000, 
                          dtype_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """대용량 파일 처리"""
        chunks = []
        
        # 청크 단위로 읽기
        for chunk in pd.read_csv(filename, chunksize=chunk_size, dtype=dtype_dict):
            # 각 청크 최적화
            optimized_chunk = MemoryOptimizer.optimize_dataframe(chunk)
            chunks.append(optimized_chunk)
            
            # 주기적 가비지 컬렉션
            if len(chunks) % 10 == 0:
                gc.collect()
        
        # 전체 데이터프레임 생성
        df = pd.concat(chunks, ignore_index=True)
        
        # 최종 가비지 컬렉션
        gc.collect()
        
        return df
    
    @staticmethod
    @contextmanager
    def memory_tracker(operation_name: str = "Operation"):
        """메모리 사용량 추적 컨텍스트 매니저"""
        # 시작 메모리
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024**2  # MB
        
        print(f"\n[Memory Tracker] Starting {operation_name}")
        print(f"Initial memory: {start_memory:.2f} MB")
        
        try:
            yield
        finally:
            # 종료 메모리
            gc.collect()
            end_memory = process.memory_info().rss / 1024**2  # MB
            memory_change = end_memory - start_memory
            
            print(f"[Memory Tracker] Completed {operation_name}")
            print(f"Final memory: {end_memory:.2f} MB")
            print(f"Memory change: {memory_change:+.2f} MB")
            print(f"Peak memory: {process.memory_info().peak_wset / 1024**2:.2f} MB\n")

class DataFrameOptimizer:
    """DataFrame 특화 최적화"""
    
    @staticmethod
    def reduce_precision(df: pd.DataFrame, columns: list = None, 
                        precision: int = 4) -> pd.DataFrame:
        """부동소수점 정밀도 감소"""
        if columns is None:
            columns = df.select_dtypes(include=['float']).columns
        
        for col in columns:
            if col in df.columns:
                df[col] = df[col].round(precision)
        
        return df
    
    @staticmethod
    def sparse_dataframe(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
        """희소 DataFrame 생성"""
        sparse_df = df.copy()
        
        for col in sparse_df.columns:
            if sparse_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                sparse_df[col] = pd.arrays.SparseArray(sparse_df[col], fill_value=fill_value)
        
        return sparse_df
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: list = None, 
                         keep: str = 'first') -> pd.DataFrame:
        """중복 제거 및 메모리 정리"""
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        removed_rows = initial_rows - len(df_cleaned)
        
        if removed_rows > 0:
            print(f"Removed {removed_rows} duplicate rows ({removed_rows/initial_rows*100:.1f}%)")
            gc.collect()
        
        return df_cleaned
    
    @staticmethod
    def optimize_index(df: pd.DataFrame) -> pd.DataFrame:
        """인덱스 최적화"""
        # DatetimeIndex 최적화
        if isinstance(df.index, pd.DatetimeIndex):
            # 불필요한 시간 정보 제거 (날짜만 필요한 경우)
            if all(df.index.time == pd.Timestamp('00:00:00').time()):
                df.index = df.index.date
        
        # 범위 인덱스로 변환 가능한 경우
        elif df.index.equals(pd.RangeIndex(len(df))):
            df = df.reset_index(drop=True)
        
        return df

class MemoryMonitor:
    """메모리 모니터링"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.history = []
    
    def get_memory_info(self) -> dict:
        """현재 메모리 정보"""
        memory_info = self.process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024**2,
            'vms_mb': memory_info.vms / 1024**2,
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024**2
        }
    
    def check_memory_usage(self, threshold_percent: float = 80.0) -> bool:
        """메모리 사용량 체크"""
        current_percent = psutil.virtual_memory().percent
        
        if current_percent > threshold_percent:
            warnings.warn(
                f"High memory usage: {current_percent:.1f}% "
                f"(threshold: {threshold_percent:.1f}%)"
            )
            return False
        
        return True
    
    def log_memory_usage(self, label: str = ""):
        """메모리 사용량 로깅"""
        info = self.get_memory_info()
        info['label'] = label
        info['timestamp'] = pd.Timestamp.now()
        
        self.history.append(info)
        
        print(f"Memory [{label}]: {info['rss_mb']:.1f} MB "
              f"({info['percent']:.1f}% of system)")
    
    def get_memory_report(self) -> pd.DataFrame:
        """메모리 사용 리포트"""
        if not self.history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history)
        df['memory_change_mb'] = df['rss_mb'].diff()
        
        return df
    
    def suggest_gc(self) -> bool:
        """가비지 컬렉션 제안"""
        gc_stats = gc.get_stats()
        
        # Generation 2 컬렉션이 오래되지 않은 경우
        if gc_stats[2]['collections'] > 0:
            last_gc2_time = gc_stats[2]['collected']
            if last_gc2_time < 1000:  # 최근에 수행됨
                return False
        
        # 메모리 사용량이 높은 경우
        if not self.check_memory_usage(70.0):
            gc.collect()
            return True
        
        return False

# 전역 메모리 모니터
memory_monitor = MemoryMonitor()

# 편의 함수
def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 메모리 최적화 편의 함수"""
    return MemoryOptimizer.optimize_dataframe(df, verbose=True)

def track_memory(operation_name: str = "Operation"):
    """메모리 추적 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with MemoryOptimizer.memory_tracker(f"{operation_name} - {func.__name__}"):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def clean_memory():
    """메모리 정리"""
    collected = gc.collect()
    print(f"Garbage collector: collected {collected} objects")
    
    # 메모리 정보 출력
    memory_info = memory_monitor.get_memory_info()
    print(f"Current memory usage: {memory_info['rss_mb']:.1f} MB")
    print(f"Available memory: {memory_info['available_mb']:.1f} MB")
