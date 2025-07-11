"""
병렬 데이터 다운로더
효율적인 멀티스레드 데이터 다운로드
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import time
from data_cache import data_cache

class ParallelDataDownloader:
    """병렬 데이터 다운로더"""
    
    def __init__(self, max_workers: int = 10, retry_count: int = 3, 
                 retry_delay: float = 1.0):
        self.max_workers = max_workers
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        # 다운로드 통계
        self.stats = {
            'success': 0,
            'failed': 0,
            'cached': 0,
            'total_time': 0
        }
    
    @data_cache.disk_cache(expiry_hours=6, cache_subdir='price_data')
    def download_single_ticker(self, ticker: str, start_date: datetime, 
                             end_date: datetime, auto_adjust: bool = True) -> Optional[pd.DataFrame]:
        """단일 티커 다운로드 (캐시 적용)"""
        for attempt in range(self.retry_count):
            try:
                data = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date,
                    progress=False,
                    auto_adjust=auto_adjust,
                    threads=False  # 개별 다운로드는 단일 스레드
                )
                
                if not data.empty:
                    self.stats['success'] += 1
                    return data
                else:
                    self.logger.warning(f"Empty data for {ticker}")
                    
            except Exception as e:
                if attempt < self.retry_count - 1:
                    self.logger.warning(f"Retry {attempt + 1}/{self.retry_count} for {ticker}: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.logger.error(f"Failed to download {ticker} after {self.retry_count} attempts: {e}")
        
        self.stats['failed'] += 1
        return None
    
    def download_multiple_tickers(self, tickers: List[str], start_date: datetime, 
                                end_date: datetime, show_progress: bool = True) -> Tuple[Dict[str, pd.DataFrame], List[Tuple[str, str]]]:
        """여러 티커를 병렬로 다운로드"""
        start_time = time.time()
        results = {}
        failed = []
        
        # 중복 제거
        unique_tickers = list(set(tickers))
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 다운로드 작업 제출
            future_to_ticker = {
                executor.submit(self.download_single_ticker, ticker, start_date, end_date): ticker
                for ticker in unique_tickers
            }
            
            # 진행 상황 표시
            if show_progress:
                futures = tqdm(as_completed(future_to_ticker), 
                             total=len(unique_tickers), 
                             desc="Downloading")
            else:
                futures = as_completed(future_to_ticker)
            
            for future in futures:
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[ticker] = data
                    else:
                        failed.append((ticker, "Empty data"))
                except Exception as e:
                    failed.append((ticker, str(e)))
                    self.logger.error(f"Error downloading {ticker}: {e}")
        
        # 통계 업데이트
        self.stats['total_time'] = time.time() - start_time
        
        self.logger.info(f"Downloaded {len(results)}/{len(unique_tickers)} tickers in {self.stats['total_time']:.2f}s")
        
        return results, failed
    
    def download_with_benchmark(self, benchmark_ticker: str, component_tickers: List[str],
                              start_date: datetime, end_date: datetime) -> Tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame], List[str]]:
        """벤치마크와 구성요소를 함께 다운로드"""
        all_tickers = [benchmark_ticker] + component_tickers
        results, failed = self.download_multiple_tickers(all_tickers, start_date, end_date)
        
        # 벤치마크 추출
        benchmark_data = results.pop(benchmark_ticker, None)
        
        # 실패한 티커 리스트
        failed_tickers = [ticker for ticker, _ in failed]
        
        if benchmark_data is None:
            self.logger.error(f"Failed to download benchmark {benchmark_ticker}")
            return None, {}, failed_tickers
        
        return benchmark_data, results, failed_tickers
    
    def download_batch_with_fallback(self, tickers: List[str], start_date: datetime,
                                   end_date: datetime, fallback_days: int = 30) -> Dict[str, pd.DataFrame]:
        """실패 시 기간을 확장하여 재시도"""
        results, failed = self.download_multiple_tickers(tickers, start_date, end_date)
        
        if failed:
            # 실패한 티커들에 대해 기간을 확장하여 재시도
            failed_tickers = [ticker for ticker, _ in failed]
            extended_start = start_date - timedelta(days=fallback_days)
            
            self.logger.info(f"Retrying {len(failed_tickers)} tickers with extended period")
            retry_results, _ = self.download_multiple_tickers(
                failed_tickers, extended_start, end_date, show_progress=False
            )
            
            results.update(retry_results)
        
        return results
    
    def validate_and_clean_data(self, data: pd.DataFrame, min_length: int = 20) -> Optional[pd.DataFrame]:
        """데이터 검증 및 정리"""
        if data is None or data.empty:
            return None
        
        # NaN 제거
        data = data.dropna()
        
        # 최소 길이 확인
        if len(data) < min_length:
            return None
        
        # 중복 인덱스 제거
        if data.index.duplicated().any():
            data = data[~data.index.duplicated(keep='first')]
        
        # 인덱스 정렬
        data = data.sort_index()
        
        return data
    
    def get_stats(self) -> dict:
        """다운로드 통계 반환"""
        total = self.stats['success'] + self.stats['failed']
        success_rate = self.stats['success'] / total * 100 if total > 0 else 0
        
        return {
            **self.stats,
            'success_rate': f"{success_rate:.1f}%",
            'avg_time_per_ticker': self.stats['total_time'] / total if total > 0 else 0
        }
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'success': 0,
            'failed': 0,
            'cached': 0,
            'total_time': 0
        }

class SmartDataDownloader(ParallelDataDownloader):
    """스마트 데이터 다운로더 (캐시 및 검증 강화)"""
    
    def __init__(self, max_workers: int = 10):
        super().__init__(max_workers)
        self.price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def extract_price_series(self, data: pd.DataFrame, column: str = 'Close') -> Optional[pd.Series]:
        """가격 시리즈 추출"""
        if data is None or data.empty:
            return None
        
        # MultiIndex 처리
        if isinstance(data.columns, pd.MultiIndex):
            if column in data.columns.get_level_values(0):
                return data[column].iloc[:, 0]
        
        # 일반 컬럼
        if column in data.columns:
            return data[column]
        
        # 첫 번째 컬럼 반환
        if len(data.columns) > 0:
            return data.iloc[:, 0]
        
        return None
    
    def download_and_extract(self, tickers: List[str], start_date: datetime, 
                           end_date: datetime, column: str = 'Close') -> Dict[str, pd.Series]:
        """다운로드 후 특정 컬럼 추출"""
        data_dict, _ = self.download_multiple_tickers(tickers, start_date, end_date)
        
        price_series = {}
        for ticker, data in data_dict.items():
            series = self.extract_price_series(data, column)
            if series is not None:
                # 검증 및 정리
                series = series.dropna()
                if len(series) >= 20:  # 최소 길이
                    price_series[ticker] = series
        
        return price_series
    
    def parallel_validation(self, data_dict: Dict[str, pd.DataFrame], 
                          min_length: int = 20) -> Dict[str, pd.DataFrame]:
        """병렬 데이터 검증"""
        validated = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.validate_and_clean_data, data, min_length): ticker
                for ticker, data in data_dict.items()
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    clean_data = future.result()
                    if clean_data is not None:
                        validated[ticker] = clean_data
                except Exception as e:
                    self.logger.error(f"Validation error for {ticker}: {e}")
        
        return validated

# 전역 다운로더 인스턴스
downloader = SmartDataDownloader(max_workers=10)

# 편의 함수
def download_tickers(tickers: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """티커 리스트 다운로드 편의 함수"""
    data_dict, failed = downloader.download_multiple_tickers(tickers, start_date, end_date)
    if failed:
        print(f"Failed to download: {[t for t, _ in failed]}")
    return data_dict
