"""
설정 관리 클래스 - 전체 시스템 설정 중앙화
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from ..core.utils import print_status

@dataclass
class JumpModelConfig:
    """Jump Model 설정"""
    use_paper_features_only: bool = True
    jump_penalty: float = 50.0
    training_cutoff_date: str = "2024-12-31"
    min_data_length: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RiskFreeRateConfig:
    """Risk-Free Rate 설정"""
    ticker: str = "^IRX"
    default_rate: float = 0.02
    use_dynamic: bool = True
    cache_duration_minutes: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RSStrategyConfig:
    """RS 전략 설정"""
    length: int = 20
    timeframe: str = "daily"
    recent_cross_days: Optional[int] = None
    min_components: int = 1
    max_components: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BacktestConfig:
    """백테스트 설정"""
    initial_capital: float = 10000000.0
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DashboardConfig:
    """대시보드 설정"""
    cache_duration_minutes: int = 30
    max_tickers_analysis: int = 50
    debug_mode: bool = False
    auto_refresh_seconds: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ConfigManager:
    """설정 관리자 - 모든 설정을 중앙화하여 관리"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        
        # 기본 설정 초기화
        self.jump_model = JumpModelConfig()
        self.risk_free_rate = RiskFreeRateConfig()
        self.rs_strategy = RSStrategyConfig()
        self.backtest = BacktestConfig()
        self.dashboard = DashboardConfig()
        
        # 설정 파일 로드
        self.load_config()
        
        print_status(f"설정 관리자 초기화: {config_file}")
    
    def load_config(self) -> bool:
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 각 섹션별 설정 로드
                if 'jump_model' in config_data:
                    jm_config = config_data['jump_model']
                    self.jump_model = JumpModelConfig(**jm_config)
                
                if 'risk_free_rate' in config_data:
                    rf_config = config_data['risk_free_rate']
                    self.risk_free_rate = RiskFreeRateConfig(**rf_config)
                
                if 'rs_strategy' in config_data:
                    rs_config = config_data['rs_strategy']
                    self.rs_strategy = RSStrategyConfig(**rs_config)
                
                if 'backtest' in config_data:
                    bt_config = config_data['backtest']
                    self.backtest = BacktestConfig(**bt_config)
                
                if 'dashboard' in config_data:
                    db_config = config_data['dashboard']
                    self.dashboard = DashboardConfig(**db_config)
                
                print_status(f"설정 파일 로드 완료: {self.config_file}", "SUCCESS")
                return True
            else:
                print_status(f"설정 파일이 없어 기본값 사용: {self.config_file}", "WARNING")
                self.save_config()  # 기본 설정 파일 생성
                return False
                
        except Exception as e:
            print_status(f"설정 파일 로드 실패: {e}", "ERROR")
            print_status("기본 설정값 사용", "WARNING")
            return False
    
    def save_config(self) -> bool:
        """설정 파일 저장"""
        try:
            config_data = {
                'jump_model': self.jump_model.to_dict(),
                'risk_free_rate': self.risk_free_rate.to_dict(),
                'rs_strategy': self.rs_strategy.to_dict(),
                'backtest': self.backtest.to_dict(),
                'dashboard': self.dashboard.to_dict(),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print_status(f"설정 파일 저장 완료: {self.config_file}", "SUCCESS")
            return True
            
        except Exception as e:
            print_status(f"설정 파일 저장 실패: {e}", "ERROR")
            return False
    
    def update_jump_model_config(self, **kwargs) -> None:
        """Jump Model 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.jump_model, key):
                setattr(self.jump_model, key, value)
                print_status(f"Jump Model 설정 업데이트: {key} = {value}")
            else:
                print_status(f"알 수 없는 Jump Model 설정: {key}", "WARNING")
    
    def update_risk_free_rate_config(self, **kwargs) -> None:
        """Risk-Free Rate 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.risk_free_rate, key):
                setattr(self.risk_free_rate, key, value)
                print_status(f"Risk-Free Rate 설정 업데이트: {key} = {value}")
            else:
                print_status(f"알 수 없는 Risk-Free Rate 설정: {key}", "WARNING")
    
    def update_rs_strategy_config(self, **kwargs) -> None:
        """RS 전략 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.rs_strategy, key):
                setattr(self.rs_strategy, key, value)
                print_status(f"RS 전략 설정 업데이트: {key} = {value}")
            else:
                print_status(f"알 수 없는 RS 전략 설정: {key}", "WARNING")
    
    def update_backtest_config(self, **kwargs) -> None:
        """백테스트 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.backtest, key):
                setattr(self.backtest, key, value)
                print_status(f"백테스트 설정 업데이트: {key} = {value}")
            else:
                print_status(f"알 수 없는 백테스트 설정: {key}", "WARNING")
    
    def update_dashboard_config(self, **kwargs) -> None:
        """대시보드 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.dashboard, key):
                setattr(self.dashboard, key, value)
                print_status(f"대시보드 설정 업데이트: {key} = {value}")
            else:
                print_status(f"알 수 없는 대시보드 설정: {key}", "WARNING")
    
    def get_training_cutoff_date(self) -> datetime:
        """Training cutoff 날짜 반환"""
        try:
            return datetime.strptime(self.jump_model.training_cutoff_date, "%Y-%m-%d")
        except:
            return datetime(2024, 12, 31)
    
    def print_current_config(self) -> None:
        """현재 설정 출력"""
        print_status("현재 설정 상태:")
        
        print(f"\n🔧 Jump Model 설정:")
        print(f"  - Feature Type: {'논문 정확한 3특징' if self.jump_model.use_paper_features_only else '논문 기반 + 추가'}")
        print(f"  - Jump Penalty: {self.jump_model.jump_penalty}")
        print(f"  - Training Cutoff: {self.jump_model.training_cutoff_date}")
        print(f"  - Min Data Length: {self.jump_model.min_data_length}")
        
        print(f"\n🏦 Risk-Free Rate 설정:")
        print(f"  - Ticker: {self.risk_free_rate.ticker}")
        print(f"  - Default Rate: {self.risk_free_rate.default_rate*100:.1f}%")
        print(f"  - Use Dynamic: {self.risk_free_rate.use_dynamic}")
        print(f"  - Cache Duration: {self.risk_free_rate.cache_duration_minutes}분")
        
        print(f"\n📊 RS 전략 설정:")
        print(f"  - Length: {self.rs_strategy.length}")
        print(f"  - Timeframe: {self.rs_strategy.timeframe}")
        print(f"  - Recent Cross Days: {self.rs_strategy.recent_cross_days}")
        print(f"  - Component Range: {self.rs_strategy.min_components}-{self.rs_strategy.max_components}")
        
        print(f"\n💼 백테스트 설정:")
        print(f"  - Initial Capital: {self.backtest.initial_capital:,.0f}")
        print(f"  - Rebalance Frequency: {self.backtest.rebalance_frequency}")
        print(f"  - Transaction Cost: {self.backtest.transaction_cost*100:.3f}%")
        print(f"  - Slippage: {self.backtest.slippage*100:.3f}%")
        
        print(f"\n📱 대시보드 설정:")
        print(f"  - Cache Duration: {self.dashboard.cache_duration_minutes}분")
        print(f"  - Max Tickers: {self.dashboard.max_tickers_analysis}")
        print(f"  - Debug Mode: {self.dashboard.debug_mode}")
        print(f"  - Auto Refresh: {self.dashboard.auto_refresh_seconds}초")
    
    def reset_to_defaults(self) -> None:
        """기본값으로 리셋"""
        self.jump_model = JumpModelConfig()
        self.risk_free_rate = RiskFreeRateConfig()
        self.rs_strategy = RSStrategyConfig()
        self.backtest = BacktestConfig()
        self.dashboard = DashboardConfig()
        
        print_status("모든 설정을 기본값으로 리셋했습니다.", "SUCCESS")
    
    def validate_config(self) -> bool:
        """설정 유효성 검증"""
        valid = True
        
        # Jump Model 검증
        if self.jump_model.jump_penalty < 10 or self.jump_model.jump_penalty > 100:
            print_status("Jump Penalty는 10-100 범위여야 합니다.", "ERROR")
            valid = False
        
        if self.jump_model.min_data_length < 100:
            print_status("Min Data Length는 100 이상이어야 합니다.", "ERROR")
            valid = False
        
        # Risk-Free Rate 검증
        if self.risk_free_rate.default_rate < 0 or self.risk_free_rate.default_rate > 0.2:
            print_status("Default Rate는 0-20% 범위여야 합니다.", "ERROR")
            valid = False
        
        # RS 전략 검증
        if self.rs_strategy.length < 5 or self.rs_strategy.length > 100:
            print_status("RS Length는 5-100 범위여야 합니다.", "ERROR")
            valid = False
        
        if self.rs_strategy.timeframe not in ['daily', 'weekly']:
            print_status("Timeframe은 'daily' 또는 'weekly'여야 합니다.", "ERROR")
            valid = False
        
        # 백테스트 검증
        if self.backtest.initial_capital <= 0:
            print_status("Initial Capital은 0보다 커야 합니다.", "ERROR")
            valid = False
        
        if self.backtest.transaction_cost < 0 or self.backtest.transaction_cost > 0.1:
            print_status("Transaction Cost는 0-10% 범위여야 합니다.", "ERROR")
            valid = False
        
        if valid:
            print_status("모든 설정이 유효합니다.", "SUCCESS")
        
        return valid


# 전역 설정 관리자 인스턴스
_config_manager = None

def get_config_manager(config_file: str = "config.json") -> ConfigManager:
    """전역 설정 관리자 반환 (싱글톤 패턴)"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager

def load_config(config_file: str = "config.json") -> ConfigManager:
    """설정 로드 편의 함수"""
    return get_config_manager(config_file)


# 사용 예시
if __name__ == "__main__":
    # 설정 관리자 생성
    config = ConfigManager()
    
    # 현재 설정 출력
    config.print_current_config()
    
    # 설정 변경
    config.update_jump_model_config(jump_penalty=60.0, use_paper_features_only=False)
    config.update_risk_free_rate_config(ticker="^TNX", use_dynamic=False)
    
    # 설정 검증
    if config.validate_config():
        # 설정 저장
        config.save_config()
        print_status("설정 업데이트 완료!", "SUCCESS")
    else:
        print_status("설정 검증 실패!", "ERROR")