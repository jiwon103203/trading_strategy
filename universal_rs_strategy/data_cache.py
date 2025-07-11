"""
데이터 캐싱 시스템
효율적인 데이터 재사용을 위한 디스크 기반 캐시
"""

from functools import lru_cache, wraps
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging
from typing import Optional, Any, Callable

class DataCache:
    """데이터 캐싱 시스템"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # 메모리 캐시 (LRU)
        self._memory_cache = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'expired': 0
        }
    
    def cache_key(self, *args, **kwargs) -> str:
        """캐시 키 생성"""
        # 모든 인자를 문자열로 변환하여 해시
        key_parts = []
        for arg in args:
            if isinstance(arg, (datetime, pd.Timestamp)):
                key_parts.append(arg.strftime('%Y%m%d'))
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (datetime, pd.Timestamp)):
                key_parts.append(f"{k}={v.strftime('%Y%m%d')}")
            else:
                key_parts.append(f"{k}={v}")
        
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def disk_cache(self, expiry_hours: int = 24, cache_subdir: str = None):
        """디스크 기반 캐싱 데코레이터"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 캐시 키 생성
                cache_key = self.cache_key(*args, **kwargs)
                
                # 캐시 디렉토리 설정
                cache_dir = self.cache_dir
                if cache_subdir:
                    cache_dir = cache_dir / cache_subdir
                    cache_dir.mkdir(exist_ok=True)
                
                cache_file = cache_dir / f"{cache_key}.pkl"
                
                # 캐시 확인
                if cache_file.exists():
                    age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if age.total_seconds() < expiry_hours * 3600:
                        try:
                            with open(cache_file, 'rb') as f:
                                result = pickle.load(f)
                                self._cache_stats['hits'] += 1
                                self.logger.debug(f"Cache hit: {func.__name__} - {cache_key}")
                                return result
                        except Exception as e:
                            self.logger.warning(f"Cache read error: {e}")
                    else:
                        self._cache_stats['expired'] += 1
                        cache_file.unlink()  # 만료된 캐시 삭제
                
                # 캐시 미스 - 함수 실행
                self._cache_stats['misses'] += 1
                self.logger.debug(f"Cache miss: {func.__name__} - {cache_key}")
                result = func(*args, **kwargs)
                
                # 결과가 None이 아니고 비어있지 않은 경우만 캐시
                if result is not None:
                    if isinstance(result, pd.DataFrame) and result.empty:
                        return result
                    
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(result, f)
                        self.logger.debug(f"Cached: {func.__name__} - {cache_key}")
                    except Exception as e:
                        self.logger.warning(f"Cache write error: {e}")
                
                return result
            
            # 캐시 관리 메서드 추가
            wrapper.clear_cache = lambda: self.clear_func_cache(func.__name__, cache_subdir)
            wrapper.cache_stats = lambda: self.get_cache_stats()
            
            return wrapper
        return decorator
    
    def memory_cache(self, maxsize: int = 128):
        """메모리 기반 캐싱 (LRU)"""
        def decorator(func: Callable) -> Callable:
            cached_func = lru_cache(maxsize=maxsize)(func)
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return cached_func(*args, **kwargs)
            
            wrapper.cache_info = cached_func.cache_info
            wrapper.cache_clear = cached_func.cache_clear
            
            return wrapper
        return decorator
    
    def clear_func_cache(self, func_name: str, cache_subdir: str = None):
        """특정 함수의 캐시 클리어"""
        cache_dir = self.cache_dir
        if cache_subdir:
            cache_dir = cache_dir / cache_subdir
        
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.logger.info(f"Cleared cache for {func_name}")
    
    def clear_all_cache(self):
        """모든 캐시 클리어"""
        for cache_file in self.cache_dir.rglob("*.pkl"):
            cache_file.unlink()
        self.logger.info("Cleared all cache")
        
        # 통계 초기화
        self._cache_stats = {'hits': 0, 'misses': 0, 'expired': 0}
    
    def get_cache_stats(self) -> dict:
        """캐시 통계 반환"""
        total = sum(self._cache_stats.values())
        if total > 0:
            hit_rate = self._cache_stats['hits'] / total * 100
        else:
            hit_rate = 0
        
        return {
            **self._cache_stats,
            'hit_rate': f"{hit_rate:.1f}%",
            'total_requests': total
        }
    
    def get_cache_size(self) -> dict:
        """캐시 크기 정보"""
        total_size = 0
        file_count = 0
        
        for cache_file in self.cache_dir.rglob("*.pkl"):
            total_size += cache_file.stat().st_size
            file_count += 1
        
        return {
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'cache_dir': str(self.cache_dir)
        }
    
    def cleanup_expired_cache(self, expiry_hours: int = 24):
        """만료된 캐시 파일 정리"""
        now = datetime.now()
        cleaned = 0
        
        for cache_file in self.cache_dir.rglob("*.pkl"):
            age = now - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age.total_seconds() > expiry_hours * 3600:
                cache_file.unlink()
                cleaned += 1
        
        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} expired cache files")
        
        return cleaned

# 전역 캐시 인스턴스
data_cache = DataCache()

# 편의 함수
def clear_cache():
    """모든 캐시 클리어"""
    data_cache.clear_all_cache()

def cache_stats():
    """캐시 통계 출력"""
    stats = data_cache.get_cache_stats()
    size_info = data_cache.get_cache_size()
    
    print("\n=== Cache Statistics ===")
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Expired: {stats['expired']}")
    print(f"Hit Rate: {stats['hit_rate']}")
    print(f"Total Size: {size_info['total_size_mb']:.2f} MB")
    print(f"File Count: {size_info['file_count']}")
    
    return stats, size_info
