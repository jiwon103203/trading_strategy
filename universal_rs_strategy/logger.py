"""
로깅 시스템
구조화된 로깅 및 성능 모니터링
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import time
from functools import wraps
from contextlib import contextmanager
import traceback

class ColoredFormatter(logging.Formatter):
    """컬러 포맷터"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

class StrategyLogger:
    """전략 전용 로거"""
    
    def __init__(self, name: str, level: int = logging.INFO, 
                 log_dir: str = "logs", enable_file: bool = True):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # 기존 핸들러 제거
        
        # 로그 디렉토리 생성
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러
        if enable_file:
            log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # 성능 메트릭스
        self.metrics = {
            'function_calls': {},
            'execution_times': {},
            'errors': []
        }
    
    def log_performance(self, metrics: Dict[str, Any], level: int = logging.INFO):
        """성과 지표 로깅"""
        self.logger.log(level, "=== Performance Metrics ===")
        for key, value in metrics.items():
            self.logger.log(level, f"{key}: {value}")
        self.logger.log(level, "=" * 30)
    
    def log_trade(self, trade_info: Dict[str, Any]):
        """거래 로깅"""
        trade_str = (
            f"TRADE: {trade_info.get('action', 'UNKNOWN')} "
            f"{trade_info.get('ticker', 'N/A')} "
            f"@ {trade_info.get('price', 0):.2f} "
            f"x {trade_info.get('shares', 0)} shares"
        )
        
        if trade_info.get('action', '').startswith('SELL'):
            trade_str += f" (P&L: {trade_info.get('pnl', 0):.2f})"
        
        self.logger.info(trade_str)
    
    def log_regime_change(self, old_regime: str, new_regime: str, confidence: float):
        """체제 변화 로깅"""
        emoji = "🟢" if new_regime == 'BULL' else "🔴"
        self.logger.warning(
            f"{emoji} REGIME CHANGE: {old_regime} → {new_regime} "
            f"(confidence: {confidence:.2%})"
        )
    
    def log_backtest_summary(self, start_date: datetime, end_date: datetime, 
                           metrics: Dict[str, Any]):
        """백테스트 요약 로깅"""
        self.logger.info("=" * 50)
        self.logger.info("BACKTEST SUMMARY")
        self.logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        self.logger.info("-" * 50)
        
        # 주요 지표만 로깅
        key_metrics = [
            '총 수익률', '연율화 수익률', '샤프 비율', '최대 낙폭',
            'BULL 기간', 'BEAR 기간', 'Out-of-Sample Days'
        ]
        
        for metric in key_metrics:
            if metric in metrics:
                self.logger.info(f"{metric}: {metrics[metric]}")
        
        self.logger.info("=" * 50)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """에러 로깅"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.metrics['errors'].append(error_info)
        
        self.logger.error(f"{error_info['error_type']}: {error_info['error_message']}")
        if context:
            self.logger.error(f"Context: {json.dumps(context, indent=2)}")
        
        # 디버그 모드에서는 전체 스택 트레이스 출력
        if self.logger.level <= logging.DEBUG:
            self.logger.debug(error_info['traceback'])
    
    def performance_tracker(self, func_name: Optional[str] = None):
        """함수 성능 추적 데코레이터"""
        def decorator(func):
            name = func_name or func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # 메트릭스 업데이트
                    if name not in self.metrics['function_calls']:
                        self.metrics['function_calls'][name] = 0
                        self.metrics['execution_times'][name] = []
                    
                    self.metrics['function_calls'][name] += 1
                    self.metrics['execution_times'][name].append(execution_time)
                    
                    # 느린 실행 경고
                    if execution_time > 1.0:
                        self.logger.warning(
                            f"Slow execution: {name} took {execution_time:.2f}s"
                        )
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.log_error(e, {
                        'function': name,
                        'args': str(args)[:100],
                        'kwargs': str(kwargs)[:100],
                        'execution_time': execution_time
                    })
                    raise
            
            return wrapper
        return decorator
    
    @contextmanager
    def timer(self, operation_name: str):
        """컨텍스트 매니저 형태의 타이머"""
        start_time = time.time()
        self.logger.debug(f"Starting: {operation_name}")
        
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            self.logger.debug(f"Completed: {operation_name} in {execution_time:.2f}s")
            
            # 메트릭스 업데이트
            if operation_name not in self.metrics['execution_times']:
                self.metrics['execution_times'][operation_name] = []
            self.metrics['execution_times'][operation_name].append(execution_time)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = {}
        
        # 함수별 통계
        for func_name, times in self.metrics['execution_times'].items():
            if times:
                stats[func_name] = {
                    'calls': self.metrics['function_calls'].get(func_name, len(times)),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        # 에러 통계
        stats['error_count'] = len(self.metrics['errors'])
        stats['error_types'] = {}
        for error in self.metrics['errors']:
            error_type = error['error_type']
            stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1
        
        return stats
    
    def export_metrics(self, filename: Optional[str] = None):
        """메트릭스 내보내기"""
        if filename is None:
            filename = self.log_dir / f"{self.name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'logger_name': self.name,
            'export_time': datetime.now().isoformat(),
            'performance_stats': self.get_performance_stats(),
            'errors': self.metrics['errors']
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to: {filename}")

class LoggerManager:
    """로거 관리자 (싱글톤)"""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_logger(self, name: str, level: int = logging.INFO, 
                   **kwargs) -> StrategyLogger:
        """로거 가져오기 또는 생성"""
        if name not in self._loggers:
            self._loggers[name] = StrategyLogger(name, level, **kwargs)
        return self._loggers[name]
    
    def set_global_level(self, level: int):
        """모든 로거의 레벨 설정"""
        for logger in self._loggers.values():
            logger.logger.setLevel(level)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """모든 로거의 메트릭스 수집"""
        return {
            name: logger.get_performance_stats()
            for name, logger in self._loggers.items()
        }
    
    def export_all_metrics(self, directory: str = "logs/metrics"):
        """모든 메트릭스 내보내기"""
        export_dir = Path(directory)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, logger in self._loggers.items():
            filename = export_dir / f"{name}_metrics_{timestamp}.json"
            logger.export_metrics(filename)

# 전역 로거 관리자
logger_manager = LoggerManager()

# 편의 함수
def get_logger(name: str = "default", level: int = logging.INFO) -> StrategyLogger:
    """로거 가져오기"""
    return logger_manager.get_logger(name, level)

def log_performance(metrics: Dict[str, Any], logger_name: str = "default"):
    """성과 로깅 편의 함수"""
    logger = get_logger(logger_name)
    logger.log_performance(metrics)

def log_trade(trade_info: Dict[str, Any], logger_name: str = "default"):
    """거래 로깅 편의 함수"""
    logger = get_logger(logger_name)
    logger.log_trade(trade_info)

# 글로벌 성능 추적 데코레이터
def track_performance(func_name: Optional[str] = None, logger_name: str = "default"):
    """성능 추적 데코레이터"""
    logger = get_logger(logger_name)
    return logger.performance_tracker(func_name)

# 설정 가능한 로그 레벨
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def set_log_level(level: str, logger_name: Optional[str] = None):
    """로그 레벨 설정"""
    level_value = LOG_LEVELS.get(level.upper(), logging.INFO)
    
    if logger_name:
        logger = get_logger(logger_name)
        logger.logger.setLevel(level_value)
    else:
        logger_manager.set_global_level(level_value)
