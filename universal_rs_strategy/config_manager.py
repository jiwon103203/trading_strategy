"""
설정 관리자
중앙화된 설정 관리 시스템
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from datetime import datetime

@dataclass
class StrategyConfig:
    """전략 설정 데이터 클래스"""
    # RS 전략 설정
    rs_length: int = 20
    rs_timeframe: str = 'daily'
    rs_recent_cross_days: Optional[int] = None
    
    # Jump Model 설정
    use_jump_model: bool = True
    jump_penalty: float = 50.0
    n_states: int = 2
    regime_lookback: int = 20
    training_cutoff_date: str = '2024-12-31'
    
    # Risk-Free Rate 설정
    rf_ticker: str = '^IRX'
    default_rf_rate: float = 0.02
    use_dynamic_rf: bool = True
    
    # 성능 설정
    cache_expiry_hours: int = 6
    max_parallel_downloads: int = 10
    chunk_size: int = 10000
    
    # 백테스트 설정
    initial_capital: float = 10000000
    rebalance_frequency: str = 'MS'  # Month Start
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.001  # 0.1%
    
    # 로깅 설정
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_dir: str = 'logs'
    
    # 디렉토리 설정
    cache_dir: str = 'cache'
    data_dir: str = 'data'
    output_dir: str = 'output'
    
    def __post_init__(self):
        """초기화 후 처리"""
        # 디렉토리 생성
        for dir_attr in ['cache_dir', 'data_dir', 'output_dir', 'log_dir']:
            dir_path = Path(getattr(self, dir_attr))
            dir_path.mkdir(exist_ok=True)

@dataclass
class DashboardConfig:
    """대시보드 설정"""
    refresh_interval: int = 300  # 5분
    max_regime_cache_minutes: int = 30
    show_debug_info: bool = False
    theme: str = 'light'
    
    # 차트 설정
    chart_height: int = 400
    max_display_tickers: int = 10
    
    # 알림 설정
    enable_notifications: bool = True
    notification_methods: List[str] = field(default_factory=lambda: ['email'])

@dataclass
class PresetConfig:
    """프리셋 설정"""
    name: str
    benchmark: str
    components: Dict[str, str]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            'name': self.name,
            'benchmark': self.benchmark,
            'components': self.components,
            'description': self.description,
            'tags': self.tags,
            'created_date': self.created_date
        }

class ConfigManager:
    """중앙화된 설정 관리"""
    
    def __init__(self, config_file: str = "config.json", config_format: str = "json"):
        self.config_file = config_file
        self.config_format = config_format.lower()
        self.config = self._load_config()
        self.dashboard_config = DashboardConfig()
        self.presets = self._load_presets()
        
        # 환경 변수 오버라이드
        self._apply_env_overrides()
    
    def _load_config(self) -> StrategyConfig:
        """설정 파일 로드"""
        if os.path.exists(self.config_file):
            try:
                if self.config_format == "json":
                    with open(self.config_file, 'r') as f:
                        data = json.load(f)
                elif self.config_format == "yaml":
                    with open(self.config_file, 'r') as f:
                        data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_format}")
                
                return StrategyConfig(**data)
            except Exception as e:
                print(f"Error loading config: {e}")
                return StrategyConfig()
        
        # 기본 설정으로 새 파일 생성
        default_config = StrategyConfig()
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: Optional[StrategyConfig] = None):
        """설정 저장"""
        if config is None:
            config = self.config
        
        data = asdict(config)
        
        if self.config_format == "json":
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
        elif self.config_format == "yaml":
            with open(self.config_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
    
    def _apply_env_overrides(self):
        """환경 변수로 설정 오버라이드"""
        env_mappings = {
            'RS_LENGTH': ('rs_length', int),
            'RS_TIMEFRAME': ('rs_timeframe', str),
            'USE_JUMP_MODEL': ('use_jump_model', lambda x: x.lower() == 'true'),
            'RF_TICKER': ('rf_ticker', str),
            'DEFAULT_RF_RATE': ('default_rf_rate', float),
            'INITIAL_CAPITAL': ('initial_capital', float),
            'LOG_LEVEL': ('log_level', str),
        }
        
        for env_var, (attr, converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = converter(os.environ[env_var])
                    setattr(self.config, attr, value)
                    print(f"Override {attr} = {value} from environment")
                except Exception as e:
                    print(f"Error converting {env_var}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 가져오기"""
        return getattr(self.config, key, default)
    
    def set(self, key: str, value: Any):
        """설정 값 설정"""
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.save_config()
        else:
            raise AttributeError(f"Unknown config key: {key}")
    
    def update(self, **kwargs):
        """여러 설정 업데이트"""
        for key, value in kwargs.items():
            self.set(key, value)
    
    def _load_presets(self) -> Dict[str, PresetConfig]:
        """프리셋 로드"""
        presets = {}
        preset_dir = Path('presets')
        
        if preset_dir.exists():
            for preset_file in preset_dir.glob('*.json'):
                try:
                    with open(preset_file, 'r') as f:
                        data = json.load(f)
                        preset = PresetConfig(**data)
                        presets[preset.name] = preset
                except Exception as e:
                    print(f"Error loading preset {preset_file}: {e}")
        
        return presets
    
    def save_preset(self, preset: PresetConfig):
        """프리셋 저장"""
        preset_dir = Path('presets')
        preset_dir.mkdir(exist_ok=True)
        
        filename = preset_dir / f"{preset.name.lower().replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(preset.to_dict(), f, indent=2)
        
        self.presets[preset.name] = preset
    
    def get_preset(self, name: str) -> Optional[PresetConfig]:
        """프리셋 가져오기"""
        return self.presets.get(name)
    
    def list_presets(self) -> List[str]:
        """프리셋 목록"""
        return list(self.presets.keys())
    
    def create_profile(self, profile_name: str) -> 'ConfigProfile':
        """설정 프로파일 생성"""
        return ConfigProfile(self, profile_name)
    
    def export_config(self, filename: str):
        """설정 내보내기"""
        data = {
            'strategy': asdict(self.config),
            'dashboard': asdict(self.dashboard_config),
            'presets': {name: preset.to_dict() for name, preset in self.presets.items()},
            'export_date': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_config(self, filename: str):
        """설정 가져오기"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        if 'strategy' in data:
            self.config = StrategyConfig(**data['strategy'])
        
        if 'dashboard' in data:
            self.dashboard_config = DashboardConfig(**data['dashboard'])
        
        if 'presets' in data:
            for name, preset_data in data['presets'].items():
                preset = PresetConfig(**preset_data)
                self.save_preset(preset)
        
        self.save_config()

class ConfigProfile:
    """설정 프로파일 (컨텍스트 관리)"""
    
    def __init__(self, config_manager: ConfigManager, profile_name: str):
        self.config_manager = config_manager
        self.profile_name = profile_name
        self.original_config = None
        self.profile_file = f"profiles/{profile_name}.json"
    
    def __enter__(self):
        """프로파일 진입"""
        # 현재 설정 백업
        self.original_config = asdict(self.config_manager.config)
        
        # 프로파일 로드
        if os.path.exists(self.profile_file):
            with open(self.profile_file, 'r') as f:
                profile_data = json.load(f)
                self.config_manager.config = StrategyConfig(**profile_data)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """프로파일 종료"""
        # 원래 설정 복원
        if self.original_config:
            self.config_manager.config = StrategyConfig(**self.original_config)
    
    def save(self):
        """현재 설정을 프로파일로 저장"""
        os.makedirs('profiles', exist_ok=True)
        with open(self.profile_file, 'w') as f:
            json.dump(asdict(self.config_manager.config), f, indent=2)

# 전역 설정 관리자
config_manager = ConfigManager()

# 편의 함수
def get_config(key: str, default: Any = None) -> Any:
    """설정 값 가져오기"""
    return config_manager.get(key, default)

def set_config(key: str, value: Any):
    """설정 값 설정"""
    config_manager.set(key, value)

def update_config(**kwargs):
    """여러 설정 업데이트"""
    config_manager.update(**kwargs)

# 설정 검증
def validate_config(config: StrategyConfig) -> List[str]:
    """설정 검증"""
    errors = []
    
    # RS 설정 검증
    if config.rs_length < 5 or config.rs_length > 100:
        errors.append(f"rs_length should be between 5 and 100, got {config.rs_length}")
    
    if config.rs_timeframe not in ['daily', 'weekly']:
        errors.append(f"rs_timeframe should be 'daily' or 'weekly', got {config.rs_timeframe}")
    
    # Jump Model 설정 검증
    if config.jump_penalty < 0:
        errors.append(f"jump_penalty should be positive, got {config.jump_penalty}")
    
    # RF 설정 검증
    if config.default_rf_rate < 0 or config.default_rf_rate > 1:
        errors.append(f"default_rf_rate should be between 0 and 1, got {config.default_rf_rate}")
    
    # 성능 설정 검증
    if config.max_parallel_downloads < 1 or config.max_parallel_downloads > 50:
        errors.append(f"max_parallel_downloads should be between 1 and 50, got {config.max_parallel_downloads}")
    
    return errors
