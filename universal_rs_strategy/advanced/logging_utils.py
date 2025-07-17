"""
고급 로깅 및 예외 처리 시스템
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable
from functools import wraps
import json

class UniversalRSLogger:
    """범용 RS 전략 로거"""
    
    def __init__(self, name: str = "universal_rs", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 로거 설정
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 기존 핸들러 제거 (중복 방지)
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 포매터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 파일 핸들러 (전체 로그)
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 에러 파일 핸들러
        error_file = self.log_dir / f"{name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"로거 초기화 완료: {name}")
    
    def debug(self, message: str, **kwargs) -> None:
        """디버그 로그"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """정보 로그"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """경고 로그"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """에러 로그"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """심각한 에러 로그"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs) -> None:
        """컨텍스트와 함께 로그 기록"""
        if kwargs:
            context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            full_message = f"{message} | {context}"
        else:
            full_message = message
        
        self.logger.log(level, full_message)
    
    def log_exception(self, message: str = "Exception occurred", **kwargs) -> None:
        """예외 정보와 함께 로그 기록"""
        exc_info = traceback.format_exc()
        self._log_with_context(logging.ERROR, f"{message}\n{exc_info}", **kwargs)
    
    def log_performance(self, func_name: str, execution_time: float, **kwargs) -> None:
        """성능 로그"""
        self.info(f"Performance: {func_name} took {execution_time:.4f}s", **kwargs)
    
    def log_data_info(self, data_name: str, data_shape: tuple, **kwargs) -> None:
        """데이터 정보 로그"""
        self.debug(f"Data: {data_name} shape={data_shape}", **kwargs)
    
    def log_regime_change(self, old_regime: str, new_regime: str, confidence: float, **kwargs) -> None:
        """체제 변경 로그"""
        self.info(f"Regime Change: {old_regime} -> {new_regime} (confidence: {confidence:.2%})", **kwargs)
    
    def log_backtest_start(self, strategy_name: str, start_date: str, end_date: str, **kwargs) -> None:
        """백테스트 시작 로그"""
        self.info(f"Backtest Start: {strategy_name} from {start_date} to {end_date}", **kwargs)
    
    def log_backtest_end(self, strategy_name: str, total_return: float, sharpe_ratio: float, **kwargs) -> None:
        """백테스트 종료 로그"""
        self.info(f"Backtest End: {strategy_name} | Return: {total_return:.2%} | Sharpe: {sharpe_ratio:.3f}", **kwargs)


class ExceptionHandler:
    """예외 처리 클래스"""
    
    def __init__(self, logger: Optional[UniversalRSLogger] = None):
        self.logger = logger or get_universal_logger()
    
    def handle_data_error(self, func_name: str, ticker: str = "", error: Exception = None) -> dict:
        """데이터 관련 에러 처리"""
        error_msg = f"Data error in {func_name}"
        if ticker:
            error_msg += f" for {ticker}"
        if error:
            error_msg += f": {str(error)}"
        
        self.logger.error(error_msg, func=func_name, ticker=ticker)
        
        return {
            'status': 'error',
            'error_type': 'data_error',
            'message': error_msg,
            'ticker': ticker,
            'function': func_name
        }
    
    def handle_calculation_error(self, func_name: str, calculation_type: str = "", error: Exception = None) -> dict:
        """계산 관련 에러 처리"""
        error_msg = f"Calculation error in {func_name}"
        if calculation_type:
            error_msg += f" ({calculation_type})"
        if error:
            error_msg += f": {str(error)}"
        
        self.logger.error(error_msg, func=func_name, calc_type=calculation_type)
        
        return {
            'status': 'error',
            'error_type': 'calculation_error',
            'message': error_msg,
            'calculation_type': calculation_type,
            'function': func_name
        }
    
    def handle_model_error(self, model_name: str, error: Exception = None) -> dict:
        """모델 관련 에러 처리"""
        error_msg = f"Model error in {model_name}"
        if error:
            error_msg += f": {str(error)}"
        
        self.logger.error(error_msg, model=model_name)
        
        return {
            'status': 'error',
            'error_type': 'model_error',
            'message': error_msg,
            'model': model_name
        }


def performance_monitor(logger: Optional[UniversalRSLogger] = None):
    """성능 모니터링 데코레이터"""
    if logger is None:
        logger = get_universal_logger()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.log_performance(func.__name__, execution_time)
                return result
            
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Function {func.__name__} failed after {execution_time:.4f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


def safe_execute(logger: Optional[UniversalRSLogger] = None, 
                default_return: Any = None,
                error_handler: Optional[ExceptionHandler] = None):
    """안전한 실행 데코레이터"""
    if logger is None:
        logger = get_universal_logger()
    if error_handler is None:
        error_handler = ExceptionHandler(logger)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                logger.log_exception(f"Error in {func.__name__}", func=func.__name__, error=str(e))
                
                # 에러 타입별 처리
                if "data" in str(e).lower() or "download" in str(e).lower():
                    return error_handler.handle_data_error(func.__name__, error=e)
                elif "calculation" in str(e).lower() or "compute" in str(e).lower():
                    return error_handler.handle_calculation_error(func.__name__, error=e)
                elif "model" in str(e).lower():
                    return error_handler.handle_model_error(func.__name__, error=e)
                else:
                    return default_return
        
        return wrapper
    return decorator


def log_function_call(logger: Optional[UniversalRSLogger] = None):
    """함수 호출 로깅 데코레이터"""
    if logger is None:
        logger = get_universal_logger()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 인자 로깅 (민감한 정보 제외)
            safe_args = []
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    safe_args.append(str(arg))
                else:
                    safe_args.append(f"<{type(arg).__name__}>")
            
            safe_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_kwargs[k] = v
                else:
                    safe_kwargs[k] = f"<{type(v).__name__}>"
            
            logger.debug(f"Calling {func.__name__}", args=safe_args, kwargs=safe_kwargs)
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed {func.__name__}")
                return result
            
            except Exception as e:
                logger.error(f"Failed {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    return decorator


class PerformanceTracker:
    """성능 추적기"""
    
    def __init__(self, logger: Optional[UniversalRSLogger] = None):
        self.logger = logger or get_universal_logger()
        self.metrics = {}
    
    def start_timer(self, name: str) -> None:
        """타이머 시작"""
        self.metrics[name] = {'start': datetime.now()}
    
    def end_timer(self, name: str) -> float:
        """타이머 종료 및 기록"""
        if name not in self.metrics:
            self.logger.warning(f"Timer {name} was not started")
            return 0.0
        
        end_time = datetime.now()
        start_time = self.metrics[name]['start']
        duration = (end_time - start_time).total_seconds()
        
        self.metrics[name]['end'] = end_time
        self.metrics[name]['duration'] = duration
        
        self.logger.log_performance(name, duration)
        return duration
    
    def get_summary(self) -> dict:
        """성능 요약 반환"""
        summary = {}
        for name, data in self.metrics.items():
            if 'duration' in data:
                summary[name] = data['duration']
        return summary
    
    def log_summary(self) -> None:
        """성능 요약 로그"""
        summary = self.get_summary()
        if summary:
            self.logger.info("Performance Summary:", **summary)


# 전역 로거 인스턴스
_universal_logger = None

def get_universal_logger(name: str = "universal_rs") -> UniversalRSLogger:
    """전역 로거 반환 (싱글톤 패턴)"""
    global _universal_logger
    if _universal_logger is None:
        _universal_logger = UniversalRSLogger(name)
    return _universal_logger


# 편의 함수들
def log_info(message: str, **kwargs) -> None:
    """정보 로그 편의 함수"""
    get_universal_logger().info(message, **kwargs)

def log_error(message: str, **kwargs) -> None:
    """에러 로그 편의 함수"""
    get_universal_logger().error(message, **kwargs)

def log_warning(message: str, **kwargs) -> None:
    """경고 로그 편의 함수"""
    get_universal_logger().warning(message, **kwargs)

def log_debug(message: str, **kwargs) -> None:
    """디버그 로그 편의 함수"""
    get_universal_logger().debug(message, **kwargs)


# 사용 예시
if __name__ == "__main__":
    # 로거 테스트
    logger = UniversalRSLogger("test")
    
    logger.info("테스트 시작")
    logger.log_backtest_start("S&P 500 Strategy", "2023-01-01", "2024-01-01")
    
    # 성능 모니터링 테스트
    @performance_monitor(logger)
    def test_function():
        import time
        time.sleep(1)
        return "완료"
    
    result = test_function()
    
    # 안전 실행 테스트
    @safe_execute(logger, default_return="기본값")
    def failing_function():
        raise ValueError("테스트 에러")
    
    safe_result = failing_function()
    print(f"안전 실행 결과: {safe_result}")
    
    # 성능 추적기 테스트
    tracker = PerformanceTracker(logger)
    tracker.start_timer("test_operation")
    import time
    time.sleep(0.5)
    duration = tracker.end_timer("test_operation")
    tracker.log_summary()
    
    logger.info("테스트 완료")