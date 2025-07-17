"""
범용 RS 모멘텀 전략 + 통합 Jump Model + 동적 Risk-Free Rate
메인 실행 스크립트 - 통합된 특징 계산 코드 사용 (효율화 버전)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from trading_strategy.universal_rs_strategy.core.preset_manager import PresetManager
from trading_strategy.universal_rs_strategy.core.universal_rs_strategy import UniversalRSStrategy
from trading_strategy.universal_rs_strategy.core.universal_jump_model import UniversalJumpModel
from trading_strategy.universal_rs_strategy.core.universal_rs_with_jump import UniversalRSWithJumpModel
from trading_strategy.universal_rs_strategy.core.utils import safe_float, safe_extract_close, validate_data, calculate_basic_metrics, print_status
import warnings
warnings.filterwarnings('ignore')

# Risk-free rate 유틸리티 import
try:
    from trading_strategy.universal_rs_strategy.advanced.risk_free_rate_utils import RiskFreeRateManager, calculate_dynamic_sharpe_ratio, calculate_dynamic_sortino_ratio
    HAS_RF_UTILS = True
except ImportError:
    print_status("risk_free_rate_utils.py가 없습니다. 기본 risk-free rate (2%) 사용", "WARNING")
    HAS_RF_UTILS = False

# 한글 폰트 설정
import platform
from matplotlib import font_manager

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    try:
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        font_manager.fontManager.addfont(font_path)
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    except:
        pass
plt.rcParams['axes.unicode_minus'] = False


class UniversalStrategyRunner:
    """범용 전략 실행기 - 통합 Jump Model + 동적 Risk-Free Rate 지원 (효율화 버전)"""
    
    def __init__(self):
        self.current_preset = None
        self.rf_ticker = '^IRX'
        self.default_rf_rate = 0.02
        
        # 통합 모델 설정
        self.use_paper_features_only = True
        self.jump_penalty = 50.0
        
        self.presets = {
            1: PresetManager.get_sp500_sectors(),
            2: PresetManager.get_kospi_sectors(),
            3: PresetManager.get_kospi_full_sectors(),
            4: PresetManager.get_kosdaq_sectors(),
            5: PresetManager.get_korea_comprehensive(),
            6: PresetManager.get_msci_countries(),
            7: PresetManager.get_europe_sectors(),
            8: PresetManager.get_global_sectors(),
            9: PresetManager.get_emerging_markets(),
            10: PresetManager.get_commodity_sectors(),
            11: PresetManager.get_factor_etfs(),
            12: PresetManager.get_thematic_etfs()
        }
    
    def print_menu(self):
        """메뉴 출력"""
        print("\n" + "="*80)
        print("범용 RS 모멘텀 전략 + 통합 Jump Model + 동적 Risk-Free Rate 시스템 (효율화)")
        print("="*80)
        print("1. 프리셋 선택 및 현재 시장 상태 확인")
        print("2. 통합 Jump Model 체제 분석")
        print("3. 백테스트 실행 (통합 Model)")
        print("4. 전략 성과 비교")
        print("5. 실시간 신호 대시보드")
        print("6. 사용자 정의 전략 생성")
        print("7. Risk-Free Rate 설정")
        print("8. 통합 모델 설정")
        print("9. 종료")
        print("="*80)
        rf_status = "📊 동적" if HAS_RF_UTILS else "📌 고정"
        feature_type = "📊 3특징" if self.use_paper_features_only else "📈 확장특징"
        print(f"현재 설정: RF={self.rf_ticker} ({rf_status}) | Features={feature_type} | Jump Penalty={self.jump_penalty}")
    
    def configure_unified_model(self):
        """통합 모델 설정"""
        print_status("통합 Jump Model 설정")
        print(f"현재 설정: {'논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가 특징'}")
        print(f"Jump Penalty: {self.jump_penalty}")
        
        print("\nFeature Type 선택:")
        print("1. 논문 정확한 3특징 (권장)")
        print("2. 논문 기반 + 추가 특징")
        
        choice = input("선택 (1-2, 엔터=현재 유지): ")
        
        if choice == '1':
            self.use_paper_features_only = True
            print_status("논문 정확한 3특징으로 설정되었습니다.", "SUCCESS")
        elif choice == '2':
            self.use_paper_features_only = False
            print_status("논문 기반 + 추가 특징으로 설정되었습니다.", "SUCCESS")
        
        # Jump Penalty 설정
        new_penalty = input(f"Jump Penalty 설정 (현재: {self.jump_penalty}, 엔터=유지): ")
        if new_penalty:
            try:
                self.jump_penalty = float(new_penalty)
                print_status(f"Jump Penalty가 {self.jump_penalty}로 설정되었습니다.", "SUCCESS")
            except:
                print_status("잘못된 입력입니다.", "ERROR")
        
        print(f"\n현재 통합 모델 설정:")
        print(f"  - Feature Type: {'논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가 특징'}")
        print(f"  - Jump Penalty: {self.jump_penalty}")
        print(f"  - Training Cutoff: 2024-12-31 (고정)")
    
    def select_preset(self):
        """프리셋 선택"""
        print_status("프리셋 선택")
        print("1. S&P 500 섹터")
        print("2. KOSPI 200 섹터 (대형주)")
        print("3. KOSPI 전체 시장 섹터")
        print("4. KOSDAQ 섹터")
        print("5. 한국 종합 시장")
        print("6. MSCI 국가별 지수")
        print("7. 유럽 섹터")
        print("8. 글로벌 섹터")
        print("9. 신흥시장")
        print("10. 원자재 섹터")
        print("11. 팩터 ETF")
        print("12. 테마 ETF")
        print("13. 사용자 정의 로드")
        
        choice = input("\n선택 (1-13): ")
        
        if choice == '13':
            filename = input("파일명 입력: ")
            self.current_preset = PresetManager.load_custom_preset(filename)
        else:
            try:
                self.current_preset = self.presets[int(choice)]
            except:
                print_status("잘못된 선택입니다.", "ERROR")
                return None
        
        if self.current_preset:
            print_status(f"선택된 전략: {self.current_preset['name']}", "SUCCESS")
            print(f"벤치마크: {self.current_preset['benchmark']}")
            print(f"구성요소 수: {len(self.current_preset['components'])}")
        
        return self.current_preset
    
    def configure_risk_free_rate(self):
        """Risk-Free Rate 설정 (간소화)"""
        print_status("Risk-Free Rate 설정")
        print(f"현재 설정: {self.rf_ticker}")
        print(f"동적 RF 지원: {'예' if HAS_RF_UTILS else '아니오'}")
        
        if not HAS_RF_UTILS:
            print_status("동적 Risk-Free Rate를 사용하려면 risk_free_rate_utils.py가 필요합니다.", "WARNING")
            new_rate = input(f"기본 금리 설정 (현재: {self.default_rf_rate*100:.1f}%, 엔터=유지): ")
            if new_rate:
                try:
                    self.default_rf_rate = float(new_rate) / 100
                    print_status(f"기본 금리가 {self.default_rf_rate*100:.1f}%로 설정되었습니다.", "SUCCESS")
                except:
                    print_status("잘못된 입력입니다.", "ERROR")
            return
        
        print("\n사용 가능한 Risk-Free Rate 티커:")
        print("1. ^IRX (미국 3개월물 국채) - 권장")
        print("2. ^TNX (미국 10년물 국채)")
        print("3. ^FVX (미국 5년물 국채)")
        print("4. 사용자 정의")
        
        choice = input("선택 (1-4, 엔터=현재 유지): ")
        
        if choice == '1':
            self.rf_ticker = '^IRX'
        elif choice == '2':
            self.rf_ticker = '^TNX'
        elif choice == '3':
            self.rf_ticker = '^FVX'
        elif choice == '4':
            custom_ticker = input("티커 입력: ")
            if custom_ticker:
                self.rf_ticker = custom_ticker
        
        # 새 설정으로 테스트
        if HAS_RF_UTILS:
            try:
                rf_manager = RiskFreeRateManager(self.rf_ticker, self.default_rf_rate)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                rf_data = rf_manager.download_risk_free_rate(start_date, end_date)
                
                if validate_data(rf_data, 1):
                    current_rate = safe_float(rf_data.iloc[-1]) * 100
                    avg_rate = safe_float(rf_data.mean()) * 100
                    print_status(f"설정 완료: {self.rf_ticker}", "SUCCESS")
                    print(f"현재 금리: {current_rate:.3f}%")
                    print(f"30일 평균: {avg_rate:.3f}%")
                else:
                    print_status(f"{self.rf_ticker} 데이터를 가져올 수 없습니다. 기본값 사용", "WARNING")
            except Exception as e:
                print_status(f"테스트 실패: {e}", "WARNING")
    
    def check_current_status(self):
        """현재 시장 상태 확인 (통합 모델 + 동적 RF 지원)"""
        if not self.current_preset:
            self.select_preset()
        
        if not self.current_preset:
            return
        
        print_status(f"{self.current_preset['name']} 현재 상태 분석 중... (통합 모델)")
        print(f"설정: Feature={'논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가'}, Jump Penalty={self.jump_penalty}")
        print(f"Risk-Free Rate: {self.rf_ticker}")
        
        # 통합 Jump Model로 체제 확인
        jump_model = UniversalJumpModel(
            benchmark_ticker=self.current_preset['benchmark'],
            benchmark_name=self.current_preset['name'],
            jump_penalty=self.jump_penalty,
            training_cutoff_date=datetime(2024, 12, 31)
        )
        
        current_regime = jump_model.get_current_regime_with_training_cutoff()
        
        if current_regime:
            oos_status = "🔮 Out-of-Sample" if current_regime['is_out_of_sample'] else "📚 In-Sample"
            
            print_status(f"시장 체제: {current_regime['regime']} (신뢰도: {current_regime['confidence']:.2%})", "SUCCESS")
            print(f"분석 상태: {oos_status}")
            print(f"기준일: {current_regime['date'].strftime('%Y-%m-%d')}")
        
        # RS 신호 확인
        if current_regime and current_regime['regime'] == 'BULL':
            strategy = UniversalRSStrategy(
                benchmark=self.current_preset['benchmark'],
                components=self.current_preset['components'],
                name=self.current_preset['name']
            )
            
            # 최근 데이터로 구성요소 선택
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            
            price_data, benchmark_data = strategy.get_price_data(start_date, end_date)
            
            if validate_data(price_data) and validate_data(benchmark_data):
                selected = strategy.select_components(price_data, benchmark_data, end_date)
                
                if selected:
                    print_status(f"투자 가능 구성요소 ({len(selected)}개):", "SUCCESS")
                    for comp in selected[:5]:  # 상위 5개만 표시
                        print(f"  - {comp['name']}: RS-Ratio={comp['rs_ratio']:.1f}, RS-Momentum={comp['rs_momentum']:.1f}")
                else:
                    print_status("투자 가능한 구성요소가 없습니다.", "WARNING")
        elif current_regime:
            print_status(f"{current_regime['regime']} 체제 - 투자 중단 권고", "WARNING")
    
    def analyze_regime(self):
        """통합 Jump Model 체제 분석"""
        if not self.current_preset:
            self.select_preset()
        
        if not self.current_preset:
            return
        
        print("\n체제 분석 기간:")
        print("1. 최근 1년")
        print("2. 최근 3년")
        print("3. 최근 5년")
        
        choice = input("선택 (1-3): ")
        years = {'1': 1, '2': 3, '3': 5}.get(choice, 3)
        
        # 통합 Jump Model 사용
        jump_model = UniversalJumpModel(
            benchmark_ticker=self.current_preset['benchmark'],
            benchmark_name=self.current_preset['name'],
            jump_penalty=self.jump_penalty,
            training_cutoff_date=datetime(2024, 12, 31)
        )
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        print_status(f"{self.current_preset['name']} 체제 분석 중... (통합 모델)")
        print(f"Feature Type: {'논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가'}")
        print(f"Jump Penalty: {self.jump_penalty}")
        print(f"Risk-Free Rate: {self.rf_ticker}")
        
        # 현재 체제만 분석 (간단한 버전)
        current_regime = jump_model.get_current_regime_with_training_cutoff()
        
        if current_regime:
            print_status(f"{self.current_preset['name']} 현재 체제 분석", "SUCCESS")
            
            # 체제 정보
            regime = current_regime['regime']
            confidence = current_regime['confidence']
            oos_status = "Out-of-Sample" if current_regime.get('is_out_of_sample', False) else "In-Sample"
            
            print(f"\n현재 체제:")
            print(f"  - 체제: {regime}")
            print(f"  - 신뢰도: {confidence:.2%}")
            print(f"  - 상태: {oos_status}")
            print(f"  - 분석일: {current_regime['date'].strftime('%Y-%m-%d')}")
            
            # 특징값 표시
            features = current_regime.get('features', {})
            if features:
                print(f"\n특징값:")
                for key, value in features.items():
                    print(f"  - {key}: {value:.6f}")
        else:
            print_status("체제 분석 실패", "ERROR")
    
    def run_backtest(self):
        """백테스트 실행 (통합 모델 + 동적 RF 지원)"""
        if not self.current_preset:
            self.select_preset()
        
        if not self.current_preset:
            return
        
        print_status("백테스트 설정")
        
        # RS 전략 설정
        timeframe = input("시간 프레임 (1: 일봉, 2: 주봉) [기본: 1]: ") or "1"
        timeframe = 'daily' if timeframe == '1' else 'weekly'
        
        use_cross = input("최근 크로스 필터링 사용? (y/n) [기본: n]: ") or "n"
        cross_days = int(input("크로스 기간 (일) [기본: 30]: ") or "30") if use_cross.lower() == 'y' else None
        
        # 통합 Jump Model 설정
        use_jump = input("통합 Jump Model 사용? (y/n) [기본: y]: ") or "y"
        use_jump = use_jump.lower() == 'y'
        
        # 백테스트 기간
        print("\n백테스트 기간:")
        print("1. 최근 1년")
        print("2. 최근 3년")
        print("3. 최근 5년")
        print("4. 사용자 지정")
        
        period = input("선택 (1-4) [기본: 2]: ") or "2"
        
        end_date = datetime.now()
        if period == '1':
            start_date = end_date - timedelta(days=365)
        elif period == '2':
            start_date = end_date - timedelta(days=365*3)
        elif period == '3':
            start_date = end_date - timedelta(days=365*5)
        else:
            start_year = int(input("시작 연도: "))
            start_month = int(input("시작 월: "))
            start_date = datetime(start_year, start_month, 1)
        
        # 초기 자본
        initial_capital = float(input("초기 자본 [기본: 10,000,000]: ") or "10000000")
        
        # 전략 생성 및 백테스트
        print_status(f"백테스트 실행 중... (통합 모델)")
        print(f"전략: {self.current_preset['name']}")
        print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print(f"설정: Feature={'논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가'}")
        print(f"Jump Penalty: {self.jump_penalty}")
        print(f"Risk-Free Rate: {self.rf_ticker}")
        
        strategy = UniversalRSWithJumpModel(
            preset_config=self.current_preset,
            rs_timeframe=timeframe,
            rs_recent_cross_days=cross_days,
            jump_penalty=self.jump_penalty,
            use_jump_model=use_jump,
            use_paper_features_only=self.use_paper_features_only,
            rf_ticker=self.rf_ticker,
            default_rf_rate=self.default_rf_rate,
            training_cutoff_date=datetime(2024, 12, 31)
        )
        
        portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date, initial_capital)
        
        if validate_data(portfolio_df, 1):
            # 성과 출력
            metrics = strategy.calculate_performance_metrics(portfolio_df)
            
            print_status("백테스트 결과 (통합 모델 + 동적 Risk-Free Rate)", "SUCCESS")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            
            # 동적 RF 추가 분석
            if HAS_RF_UTILS:
                print_status(f"동적 Risk-Free Rate 추가 분석", "SUCCESS")
                quick_sharpe = calculate_dynamic_sharpe_ratio(portfolio_df, self.rf_ticker)
                quick_sortino = calculate_dynamic_sortino_ratio(portfolio_df, self.rf_ticker)
                
                print(f"빠른 Sharpe 계산: {quick_sharpe:.3f}")
                print(f"빠른 Sortino 계산: {quick_sortino:.3f}")
            
            # 시각화
            self.visualize_results(portfolio_df, trades_df, self.current_preset['name'])
            
            # 저장
            save_choice = input("\n결과를 저장하시겠습니까? (y/n): ")
            if save_choice.lower() == 'y':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                strategy_name = self.current_preset['name'].replace(' ', '_').lower()
                portfolio_df.to_csv(f'portfolio_unified_{strategy_name}_{timestamp}.csv')
                if validate_data(trades_df, 1):
                    trades_df.to_csv(f'trades_unified_{strategy_name}_{timestamp}.csv')
                print_status("저장 완료!", "SUCCESS")
    
    def visualize_results(self, portfolio_df, trades_df, strategy_name):
        """결과 시각화 (통합 모델 정보 포함)"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{strategy_name} 백테스트 결과 (통합 모델 + 동적 RF: {self.rf_ticker})', fontsize=16)
        
        # 1. 포트폴리오 가치
        ax1 = axes[0, 0]
        ax1.plot(portfolio_df.index, portfolio_df['value'], linewidth=2)
        ax1.set_title('포트폴리오 가치')
        ax1.set_ylabel('가치')
        ax1.grid(True, alpha=0.3)
        
        # 2. 수익률
        ax2 = axes[0, 1]
        cumulative_returns = (portfolio_df['value'] / portfolio_df['value'].iloc[0] - 1) * 100
        ax2.plot(portfolio_df.index, cumulative_returns, linewidth=2)
        ax2.set_title('누적 수익률')
        ax2.set_ylabel('수익률 (%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. 보유 구성요소 수
        ax3 = axes[1, 0]
        if 'holdings' in portfolio_df.columns:
            ax3.fill_between(portfolio_df.index, portfolio_df['holdings'], alpha=0.5)
        ax3.set_title('보유 구성요소 수')
        ax3.set_ylabel('구성요소 수')
        ax3.grid(True, alpha=0.3)
        
        # 4. 체제별 포트폴리오 가치
        ax4 = axes[1, 1]
        if 'regime' in portfolio_df.columns:
            bull_mask = portfolio_df['regime'] == 'BULL'
            bear_mask = portfolio_df['regime'] == 'BEAR'
            
            if bull_mask.any():
                ax4.scatter(portfolio_df.index[bull_mask], portfolio_df['value'][bull_mask], 
                          c='green', alpha=0.3, s=1, label='BULL')
            if bear_mask.any():
                ax4.scatter(portfolio_df.index[bear_mask], portfolio_df['value'][bear_mask], 
                          c='red', alpha=0.3, s=1, label='BEAR')
            ax4.legend()
        else:
            ax4.plot(portfolio_df.index, portfolio_df['value'], linewidth=2)
        ax4.set_title('체제별 포트폴리오 가치 (통합 모델)')
        ax4.set_ylabel('가치')
        ax4.grid(True, alpha=0.3)
        
        # 통합 모델 정보 추가
        fig.text(0.02, 0.02, 
                f'통합 Jump Model: Feature={"3특징" if self.use_paper_features_only else "확장"}, '
                f'Jump Penalty={self.jump_penalty}, RF={self.rf_ticker}', 
                fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(f'{strategy_name.replace(" ", "_").lower()}_results_unified_model.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_custom_strategy(self):
        """사용자 정의 전략 생성"""
        print_status("사용자 정의 전략 생성")
        
        name = input("전략 이름: ")
        benchmark = input("벤치마크 티커 (예: ^GSPC, URTH): ")
        
        components = {}
        print("\n구성요소 추가 (빈 티커 입력시 종료)")
        while True:
            ticker = input("티커: ").strip()
            if not ticker:
                break
            component_name = input("이름: ")
            components[ticker] = component_name
        
        if components:
            PresetManager.save_custom_preset(name, benchmark, components)
            self.current_preset = {
                'name': name,
                'benchmark': benchmark,
                'components': components
            }
            print_status(f"전략 '{name}'이 생성되었습니다.", "SUCCESS")
        else:
            print_status("구성요소가 없어 전략을 생성하지 않았습니다.", "WARNING")
    
    def compare_strategies(self):
        """전략 성과 비교 (통합 모델 + 동적 RF 지원)"""
        print_status("전략 성과 비교 (통합 모델 + 동적 Risk-Free Rate)")
        
        # 첫 번째 전략
        print("\n첫 번째 전략 선택:")
        preset1 = self.select_preset()
        if not preset1:
            return
        
        # 두 번째 전략
        print("\n두 번째 전략 선택:")
        self.current_preset = None
        preset2 = self.select_preset()
        if not preset2:
            return
        
        # 백테스트 기간
        years = int(input("\n백테스트 기간 (년) [기본: 3]: ") or "3")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        print_status(f"백테스트 실행 중... (통합 모델)")
        print(f"설정: Feature={'논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가'}")
        print(f"Jump Penalty: {self.jump_penalty}")
        print(f"Risk-Free Rate: {self.rf_ticker}")
        
        results = {}
        
        # 두 전략 백테스트
        for i, preset in enumerate([preset1, preset2], 1):
            print_status(f"{i}. {preset['name']} 백테스트... (통합 모델)")
            
            strategy = UniversalRSWithJumpModel(
                preset_config=preset,
                rs_timeframe='daily',
                rs_recent_cross_days=30,
                jump_penalty=self.jump_penalty,
                use_jump_model=True,
                use_paper_features_only=self.use_paper_features_only,
                rf_ticker=self.rf_ticker,
                default_rf_rate=self.default_rf_rate,
                training_cutoff_date=datetime(2024, 12, 31)
            )
            
            portfolio_df, _, _ = strategy.backtest(start_date, end_date)
            
            if validate_data(portfolio_df, 1):
                metrics = strategy.calculate_performance_metrics(portfolio_df)
                results[preset['name']] = {
                    'portfolio': portfolio_df,
                    'metrics': metrics
                }
        
        # 결과 비교
        if len(results) == 2:
            self.display_comparison(results)
    
    def display_comparison(self, results):
        """전략 비교 결과 표시 (통합 모델 정보 포함)"""
        print_status(f"전략 비교 결과 (통합 모델 + 동적 RF: {self.rf_ticker})", "SUCCESS")
        
        # 테이블 형식으로 출력
        strategies = list(results.keys())
        print(f"\n{'지표':<30} {strategies[0]:<35} {strategies[1]:<35}")
        print("-" * 100)
        
        metrics_to_compare = [
            '총 수익률', '연율화 수익률', '연율화 변동성', 
            '샤프 비율', '통합 모델 사용', 'Feature Type', 'Jump Penalty'
        ]
        
        for metric in metrics_to_compare:
            val1 = results[strategies[0]]['metrics'].get(metric, 'N/A')
            val2 = results[strategies[1]]['metrics'].get(metric, 'N/A')
            print(f"{metric:<30} {str(val1):<35} {str(val2):<35}")
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 포트폴리오 가치
        for strategy, data in results.items():
            ax1.plot(data['portfolio'].index, data['portfolio']['value'], 
                    label=strategy, linewidth=2)
        ax1.set_ylabel('포트폴리오 가치')
        ax1.set_title(f'전략별 포트폴리오 가치 추이 (통합 모델 + 동적 RF: {self.rf_ticker})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 상대 성과
        portfolio1 = results[strategies[0]]['portfolio']
        portfolio2 = results[strategies[1]]['portfolio']
        
        # 공통 날짜 찾기
        common_dates = portfolio1.index.intersection(portfolio2.index)
        if len(common_dates) > 0:
            relative_perf = (portfolio2.loc[common_dates, 'value'] / 
                           portfolio1.loc[common_dates, 'value'] - 1) * 100
            ax2.plot(common_dates, relative_perf, 'g-', linewidth=2)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_ylabel('상대 성과 (%)')
            ax2.set_xlabel('날짜')
            ax2.set_title(f'{strategies[1]} vs {strategies[0]} 상대 성과')
            ax2.grid(True, alpha=0.3)
        
        # 통합 모델 정보 추가
        fig.text(0.02, 0.02, 
                f'통합 Jump Model: Feature={"3특징" if self.use_paper_features_only else "확장"}, '
                f'Jump Penalty={self.jump_penalty}', 
                fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig('strategy_comparison_unified_model.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dashboard(self):
        """실시간 대시보드 (통합 모델 지원)"""
        if not self.current_preset:
            self.select_preset()
        
        if not self.current_preset:
            return
        
        print_status(f"{self.current_preset['name']} 대시보드 생성 중... (통합 모델)")
        print(f"설정: Feature={'논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가'}")
        print(f"Jump Penalty: {self.jump_penalty}")
        print(f"Risk-Free Rate: {self.rf_ticker}")
        
        # 통합 Jump Model
        jump_model = UniversalJumpModel(
            benchmark_ticker=self.current_preset['benchmark'],
            benchmark_name=self.current_preset['name'],
            jump_penalty=self.jump_penalty,
            training_cutoff_date=datetime(2024, 12, 31)
        )
        
        # 현재 체제
        current_regime = jump_model.get_current_regime_with_training_cutoff()
        
        # RS 전략
        strategy = UniversalRSStrategy(
            benchmark=self.current_preset['benchmark'],
            components=self.current_preset['components'],
            name=self.current_preset['name']
        )
        
        # 최근 데이터
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        price_data, benchmark_data = strategy.get_price_data(start_date, end_date)
        
        if validate_data(price_data) and validate_data(benchmark_data):
            selected = strategy.select_components(price_data, benchmark_data, end_date)
            
            # 투자 권고 (통합 모델 고려)
            print_status("투자 권고 (통합 모델)", "SUCCESS")
            if current_regime and current_regime['regime'] == 'BULL' and selected:
                print_status(f"투자 실행 권고 (통합 모델)", "SUCCESS")
                print(f"   - 투자 가능 구성요소: {len(selected)}개")
                print(f"   - 권고사항: 선택된 구성요소에 동일 가중 분산 투자")
                print(f"   - Jump Penalty: {self.jump_penalty} (체제 안정성)")
                        
            elif current_regime and current_regime['regime'] == 'BEAR':
                print_status(f"투자 중단 권고 (통합 모델)", "WARNING")
                print(f"   - 시장 체제: BEAR (하락장)")
                print(f"   - 권고사항: 모든 포지션 청산 후 현금 보유")
                print(f"   - Jump Penalty {self.jump_penalty}로 인한 안정적 체제 판단")
            else:
                print_status(f"대기 권고 (통합 모델)", "WARNING")
                print(f"   - 투자 가능한 구성요소가 없음")
        
        print_status(f"통합 모델 대시보드 생성 완료!", "SUCCESS")


def main():
    """메인 실행 함수"""
    runner = UniversalStrategyRunner()
    
    print("\n" + "="*80)
    print("범용 RS 모멘텀 전략 + 통합 Jump Model + 동적 Risk-Free Rate 시스템 (효율화)")
    print("통합된 특징 계산 코드로 일관성 있는 분석 제공")
    print("="*80)
    
    while True:
        runner.print_menu()
        choice = input("\n선택: ")
        
        try:
            if choice == '1':
                runner.check_current_status()
            elif choice == '2':
                runner.analyze_regime()
            elif choice == '3':
                runner.run_backtest()
            elif choice == '4':
                runner.compare_strategies()
            elif choice == '5':
                runner.create_dashboard()
            elif choice == '6':
                runner.create_custom_strategy()
            elif choice == '7':
                runner.configure_risk_free_rate()
            elif choice == '8':
                runner.configure_unified_model()
            elif choice == '9':
                print_status("프로그램을 종료합니다.", "SUCCESS")
                print_status("통합 Jump Model + 동적 Risk-Free Rate 시스템을 사용해주셔서 감사합니다!", "SUCCESS")
                break
            else:
                print_status("잘못된 선택입니다. 다시 선택해주세요.", "ERROR")
                
        except Exception as e:
            print_status(f"오류가 발생했습니다: {e}", "ERROR")
            print_status("다시 시도해주세요.", "WARNING")
        
        input("\n계속하려면 엔터를 누르세요...")


if __name__ == "__main__":
    main()
