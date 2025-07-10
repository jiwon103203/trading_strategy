"""
범용 RS 모멘텀 전략 + Jump Model + 동적 Risk-Free Rate
메인 실행 스크립트
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from preset_manager import PresetManager
from universal_rs_strategy import UniversalRSStrategy
from universal_jump_model import UniversalJumpModel
from universal_rs_with_jump import UniversalRSWithJumpModel
import warnings
warnings.filterwarnings('ignore')

# Risk-free rate 유틸리티 import
try:
    from risk_free_rate_utils import RiskFreeRateManager, calculate_dynamic_sharpe_ratio, calculate_dynamic_sortino_ratio
    HAS_RF_UTILS = True
except ImportError:
    print("Warning: risk_free_rate_utils.py가 없습니다. 기본 risk-free rate (2%) 사용")
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
    """범용 전략 실행기 - 동적 Risk-Free Rate 지원"""
    
    def __init__(self):
        self.current_preset = None
        self.rf_ticker = '^IRX'  # 기본 Risk-Free Rate 티커
        self.default_rf_rate = 0.02  # 기본 2%
        
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
        print("범용 RS 모멘텀 전략 + Jump Model + 동적 Risk-Free Rate 시스템")
        print("="*80)
        print("1. 프리셋 선택 및 현재 시장 상태 확인")
        print("2. Jump Model 체제 분석 (동적 RF)")
        print("3. 백테스트 실행 (Jump Model + 동적 RF)")
        print("4. 전략 성과 비교 (동적 RF)")
        print("5. 실시간 신호 대시보드")
        print("6. 사용자 정의 전략 생성")
        print("7. Risk-Free Rate 설정")
        print("8. 동적 RF 성과 분석")
        print("9. 종료")
        print("="*80)
        rf_status = "📊 동적" if HAS_RF_UTILS else "📌 고정"
        print(f"현재 Risk-Free Rate: {self.rf_ticker} ({rf_status})")
    
    def select_preset(self):
        """프리셋 선택"""
        print("\n=== 프리셋 선택 ===")
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
                print("잘못된 선택입니다.")
                return None
        
        if self.current_preset:
            print(f"\n선택된 전략: {self.current_preset['name']}")
            print(f"벤치마크: {self.current_preset['benchmark']}")
            print(f"구성요소 수: {len(self.current_preset['components'])}")
        
        return self.current_preset
    
    def configure_risk_free_rate(self):
        """Risk-Free Rate 설정"""
        print("\n=== Risk-Free Rate 설정 ===")
        print(f"현재 설정: {self.rf_ticker}")
        print(f"동적 RF 지원: {'예' if HAS_RF_UTILS else '아니오'}")
        
        if not HAS_RF_UTILS:
            print("동적 Risk-Free Rate를 사용하려면 risk_free_rate_utils.py가 필요합니다.")
            new_rate = input(f"기본 금리 설정 (현재: {self.default_rf_rate*100:.1f}%, 엔터=유지): ")
            if new_rate:
                try:
                    self.default_rf_rate = float(new_rate) / 100
                    print(f"기본 금리가 {self.default_rf_rate*100:.1f}%로 설정되었습니다.")
                except:
                    print("잘못된 입력입니다.")
            return
        
        print("\n사용 가능한 Risk-Free Rate 티커:")
        print("1. ^IRX (미국 3개월물 국채)")
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
                
                if rf_data is not None and not rf_data.empty:
                    current_rate = rf_data.iloc[-1] * 100
                    avg_rate = rf_data.mean() * 100
                    print(f"\n✅ 설정 완료: {self.rf_ticker}")
                    print(f"현재 금리: {current_rate:.3f}%")
                    print(f"30일 평균: {avg_rate:.3f}%")
                else:
                    print(f"⚠️ {self.rf_ticker} 데이터를 가져올 수 없습니다. 기본값 사용")
            except Exception as e:
                print(f"⚠️ 테스트 실패: {e}")
    
    def check_current_status(self):
        """현재 시장 상태 확인 (동적 RF 지원)"""
        if not self.current_preset:
            self.select_preset()
        
        if not self.current_preset:
            return
        
        print(f"\n{self.current_preset['name']} 현재 상태 분석 중...")
        print(f"Risk-Free Rate: {self.rf_ticker}")
        
        # Jump Model로 체제 확인 (동적 RF 사용)
        jump_model = UniversalJumpModel(
            benchmark_ticker=self.current_preset['benchmark'],
            benchmark_name=self.current_preset['name'],
            rf_ticker=self.rf_ticker,
            default_rf_rate=self.default_rf_rate,
            training_cutoff_date=datetime(2024, 12, 31)
        )
        
        current_regime = jump_model.get_current_regime_with_training_cutoff()
        
        if current_regime:
            oos_status = "🔮 Out-of-Sample" if current_regime['is_out_of_sample'] else "📚 In-Sample"
            rf_status = "📊 동적" if current_regime['dynamic_rf_used'] else "📌 고정"
            
            print(f"\n시장 체제: {current_regime['regime']} (신뢰도: {current_regime['confidence']:.2%})")
            print(f"분석 상태: {oos_status}")
            print(f"Risk-Free Rate: {rf_status} ({current_regime['current_rf_rate']:.3f}%)")
            print(f"기준일: {current_regime['date'].strftime('%Y-%m-%d')}")
        
        # RS 신호 확인
        if current_regime and current_regime['regime'] == 'BULL':
            strategy = UniversalRSStrategy(
                benchmark=self.current_preset['benchmark'],
                components=self.current_preset['components'],
                name=self.current_preset['name'],
                rf_ticker=self.rf_ticker,
                default_rf_rate=self.default_rf_rate
            )
            
            # 최근 데이터로 구성요소 선택
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            
            price_data, benchmark_data = strategy.get_price_data(start_date, end_date)
            
            if price_data and benchmark_data:
                selected = strategy.select_components(price_data, benchmark_data, end_date)
                
                if selected:
                    print(f"\n투자 가능 구성요소 ({len(selected)}개):")
                    for comp in selected[:5]:  # 상위 5개만 표시
                        print(f"  - {comp['name']}: RS-Ratio={comp['rs_ratio']:.1f}, RS-Momentum={comp['rs_momentum']:.1f}")
                    
                    # 동적 RF 기반 권고
                    if current_regime['dynamic_rf_used']:
                        rf_level = current_regime['current_rf_rate']
                        if rf_level > 4.0:
                            print(f"\n⚠️ 높은 금리 환경 ({rf_level:.2f}%) - 신중한 투자 권고")
                        elif rf_level < 1.0:
                            print(f"\n✅ 낮은 금리 환경 ({rf_level:.2f}%) - 적극적 투자 고려")
                        else:
                            print(f"\n📊 보통 금리 환경 ({rf_level:.2f}%) - 균형적 투자")
                else:
                    print("\n투자 가능한 구성요소가 없습니다.")
        elif current_regime:
            print(f"\n{current_regime['regime']} 체제 - 투자 중단 권고")
    
    def analyze_regime(self):
        """Jump Model 체제 분석 (동적 RF 지원)"""
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
        
        jump_model = UniversalJumpModel(
            benchmark_ticker=self.current_preset['benchmark'],
            benchmark_name=self.current_preset['name'],
            rf_ticker=self.rf_ticker,
            default_rf_rate=self.default_rf_rate,
            training_cutoff_date=datetime(2024, 12, 31)
        )
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        print(f"\n{self.current_preset['name']} 체제 분석 중...")
        print(f"Risk-Free Rate: {self.rf_ticker}")
        
        stats = jump_model.get_regime_statistics(start_date, end_date)
        
        if stats:
            print(f"\n=== {self.current_preset['name']} 체제 통계 ({years}년) ===")
            
            # Risk-Free Rate 정보
            rf_info = stats.get('risk_free_rate', {})
            if rf_info.get('dynamic_used', False):
                print(f"\nRisk-Free Rate ({rf_info['ticker']}):")
                print(f"  - 평균: {rf_info['avg_rate']:.3f}%")
                print(f"  - 범위: {rf_info['min_rate']:.3f}% ~ {rf_info['max_rate']:.3f}%")
                print(f"  - 표준편차: {rf_info['std_rate']:.3f}%")
            else:
                print(f"\nRisk-Free Rate: {rf_info['avg_rate']:.1f}% (고정)")
            
            # 체제별 통계
            for regime, data in stats.items():
                if regime != 'risk_free_rate':
                    print(f"\n{regime}:")
                    print(f"  - 총 기간: {data['total_days']}일 ({data['percentage']:.1f}%)")
                    print(f"  - 평균 지속: {data['avg_duration']:.0f}일")
                    print(f"  - 최대 지속: {data['max_duration']}일")
                    print(f"  - 전환 횟수: {data['transitions']}회")
    
    def run_backtest(self):
        """백테스트 실행 (동적 RF 지원)"""
        if not self.current_preset:
            self.select_preset()
        
        if not self.current_preset:
            return
        
        print("\n백테스트 설정:")
        
        # RS 전략 설정
        timeframe = input("시간 프레임 (1: 일봉, 2: 주봉) [기본: 1]: ") or "1"
        timeframe = 'daily' if timeframe == '1' else 'weekly'
        
        use_cross = input("최근 크로스 필터링 사용? (y/n) [기본: n]: ") or "n"
        cross_days = int(input("크로스 기간 (일) [기본: 30]: ") or "30") if use_cross.lower() == 'y' else None
        
        # Jump Model 설정
        use_jump = input("Jump Model 사용? (y/n) [기본: y]: ") or "y"
        use_jump = use_jump.lower() == 'y'
        
        jump_penalty = float(input("Jump Penalty [기본: 50]: ") or "50") if use_jump else 50.0
        
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
        print(f"\n백테스트 실행 중...")
        print(f"전략: {self.current_preset['name']}")
        print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print(f"Risk-Free Rate: {self.rf_ticker}")
        
        strategy = UniversalRSWithJumpModel(
            preset_config=self.current_preset,
            rs_timeframe=timeframe,
            rs_recent_cross_days=cross_days,
            jump_penalty=jump_penalty,
            use_jump_model=use_jump,
            rf_ticker=self.rf_ticker,
            default_rf_rate=self.default_rf_rate,
            training_cutoff_date=datetime(2024, 12, 31)
        )
        
        portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date, initial_capital)
        
        if portfolio_df is not None and not portfolio_df.empty:
            # 성과 출력
            metrics = strategy.calculate_performance_metrics(portfolio_df)
            
            print("\n=== 백테스트 결과 (동적 Risk-Free Rate) ===")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            
            # 동적 RF 추가 분석
            if HAS_RF_UTILS:
                print(f"\n=== 동적 Risk-Free Rate 추가 분석 ===")
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
                portfolio_df.to_csv(f'portfolio_{strategy_name}_{timestamp}.csv')
                if not trades_df.empty:
                    trades_df.to_csv(f'trades_{strategy_name}_{timestamp}.csv')
                print("저장 완료!")
    
    def dynamic_rf_performance_analysis(self):
        """동적 Risk-Free Rate 성과 분석"""
        if not HAS_RF_UTILS:
            print("동적 Risk-Free Rate 분석을 위해서는 risk_free_rate_utils.py가 필요합니다.")
            return
        
        print("\n=== 동적 Risk-Free Rate 성과 분석 ===")
        
        # 여러 RF 티커 비교
        rf_tickers = ['^IRX', '^TNX', '^FVX']
        rf_names = ['3개월물', '10년물', '5년물']
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2년
        
        print(f"분석 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 각 RF 티커별 데이터 수집
        rf_data = {}
        for ticker, name in zip(rf_tickers, rf_names):
            try:
                rf_manager = RiskFreeRateManager(ticker, self.default_rf_rate)
                data = rf_manager.download_risk_free_rate(start_date, end_date)
                if data is not None and not data.empty:
                    rf_data[name] = data
                    print(f"{name} ({ticker}): 평균 {data.mean()*100:.3f}%")
            except Exception as e:
                print(f"{name} ({ticker}): 데이터 수집 실패 - {e}")
        
        # 현재 설정 RF 상세 분석
        if self.current_preset:
            print(f"\n현재 전략 ({self.current_preset['name']}) RF 분석:")
            
            # 시뮬레이션 포트폴리오 생성 (간단한 예시)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)
            returns = np.random.normal(0.0008, 0.015, len(dates))  # 연 20% 수익, 15% 변동성
            portfolio_values = 10000000 * (1 + returns).cumprod()
            
            portfolio_df = pd.DataFrame({
                'value': portfolio_values
            }, index=dates)
            
            # 다양한 RF로 성과 지표 계산
            for ticker, name in zip(rf_tickers, rf_names):
                try:
                    sharpe = calculate_dynamic_sharpe_ratio(portfolio_df, ticker)
                    sortino = calculate_dynamic_sortino_ratio(portfolio_df, ticker)
                    
                    print(f"  {name} 기준 - Sharpe: {sharpe:.3f}, Sortino: {sortino:.3f}")
                except:
                    print(f"  {name} 기준 - 계산 실패")
        
        # RF 변화가 성과에 미치는 영향 시뮬레이션
        print(f"\n=== Risk-Free Rate 변화 영향 분석 ===")
        
        base_return = 10.0  # 10% 연간 수익률
        volatility = 15.0   # 15% 연간 변동성
        
        rf_scenarios = [0.5, 2.0, 4.0, 6.0]  # 다양한 RF 시나리오
        
        print("RF 수준에 따른 Sharpe Ratio 변화:")
        for rf in rf_scenarios:
            sharpe = (base_return - rf) / volatility
            print(f"  RF {rf:.1f}%: Sharpe {sharpe:.3f}")
    
    def visualize_results(self, portfolio_df, trades_df, strategy_name):
        """결과 시각화 (동적 RF 정보 포함)"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{strategy_name} 백테스트 결과 (동적 RF: {self.rf_ticker})', fontsize=16)
        
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
        ax4.set_title('체제별 포트폴리오 가치')
        ax4.set_ylabel('가치')
        ax4.grid(True, alpha=0.3)
        
        # Risk-Free Rate 정보 추가
        if HAS_RF_UTILS:
            fig.text(0.02, 0.02, f'동적 Risk-Free Rate: {self.rf_ticker}', 
                    fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(f'{strategy_name.replace(" ", "_").lower()}_results_dynamic_rf.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_custom_strategy(self):
        """사용자 정의 전략 생성"""
        print("\n=== 사용자 정의 전략 생성 ===")
        
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
            print(f"\n전략 '{name}'이 생성되었습니다.")
        else:
            print("구성요소가 없어 전략을 생성하지 않았습니다.")
    
    def compare_strategies(self):
        """전략 성과 비교 (동적 RF 지원)"""
        print("\n=== 전략 성과 비교 (동적 Risk-Free Rate) ===")
        
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
        
        print(f"\n백테스트 실행 중... (Risk-Free Rate: {self.rf_ticker})")
        
        results = {}
        
        # 두 전략 백테스트
        for i, preset in enumerate([preset1, preset2], 1):
            print(f"\n{i}. {preset['name']} 백테스트...")
            
            strategy = UniversalRSWithJumpModel(
                preset_config=preset,
                rs_timeframe='daily',
                rs_recent_cross_days=30,
                jump_penalty=50.0,
                use_jump_model=True,
                rf_ticker=self.rf_ticker,
                default_rf_rate=self.default_rf_rate,
                training_cutoff_date=datetime(2024, 12, 31)
            )
            
            portfolio_df, _, _ = strategy.backtest(start_date, end_date)
            
            if portfolio_df is not None:
                metrics = strategy.calculate_performance_metrics(portfolio_df)
                results[preset['name']] = {
                    'portfolio': portfolio_df,
                    'metrics': metrics
                }
        
        # 결과 비교
        if len(results) == 2:
            self.display_comparison(results)
    
    def display_comparison(self, results):
        """전략 비교 결과 표시 (동적 RF 정보 포함)"""
        print(f"\n=== 전략 비교 결과 (동적 RF: {self.rf_ticker}) ===")
        
        # 테이블 형식으로 출력
        strategies = list(results.keys())
        print(f"\n{'지표':<25} {strategies[0]:<35} {strategies[1]:<35}")
        print("-" * 95)
        
        metrics_to_compare = [
            '총 수익률', '연율화 수익률', '연율화 변동성', 
            '샤프 비율 (동적)', '소르티노 비율 (동적)', '평균 Risk-Free Rate'
        ]
        
        for metric in metrics_to_compare:
            val1 = results[strategies[0]]['metrics'].get(metric, 'N/A')
            val2 = results[strategies[1]]['metrics'].get(metric, 'N/A')
            print(f"{metric:<25} {str(val1):<35} {str(val2):<35}")
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 포트폴리오 가치
        for strategy, data in results.items():
            ax1.plot(data['portfolio'].index, data['portfolio']['value'], 
                    label=strategy, linewidth=2)
        ax1.set_ylabel('포트폴리오 가치')
        ax1.set_title(f'전략별 포트폴리오 가치 추이 (동적 RF: {self.rf_ticker})')
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
        
        plt.tight_layout()
        plt.savefig('strategy_comparison_dynamic_rf.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dashboard(self):
        """실시간 대시보드 (동적 RF 지원)"""
        if not self.current_preset:
            self.select_preset()
        
        if not self.current_preset:
            return
        
        print(f"\n{self.current_preset['name']} 대시보드 생성 중...")
        print(f"Risk-Free Rate: {self.rf_ticker}")
        
        # Jump Model (동적 RF 사용)
        jump_model = UniversalJumpModel(
            benchmark_ticker=self.current_preset['benchmark'],
            benchmark_name=self.current_preset['name'],
            rf_ticker=self.rf_ticker,
            default_rf_rate=self.default_rf_rate,
            training_cutoff_date=datetime(2024, 12, 31)
        )
        
        # 현재 체제
        current_regime = jump_model.get_current_regime_with_training_cutoff()
        
        # RS 전략 (동적 RF 사용)
        strategy = UniversalRSStrategy(
            benchmark=self.current_preset['benchmark'],
            components=self.current_preset['components'],
            name=self.current_preset['name'],
            rf_ticker=self.rf_ticker,
            default_rf_rate=self.default_rf_rate
        )
        
        # 최근 데이터
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        price_data, benchmark_data = strategy.get_price_data(start_date, end_date)
        
        if price_data and benchmark_data:
            selected = strategy.select_components(price_data, benchmark_data, end_date)
            
            # 대시보드 생성
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
            
            # 1. 현재 체제 + RF 정보
            ax1 = fig.add_subplot(gs[0, :])
            ax1.axis('off')
            
            if current_regime:
                regime_color = 'green' if current_regime['regime'] == 'BULL' else 'red'
                oos_status = "🔮 Out-of-Sample" if current_regime['is_out_of_sample'] else "📚 In-Sample"
                rf_status = "📊 동적" if current_regime['dynamic_rf_used'] else "📌 고정"
                
                regime_text = f"""{self.current_preset['name']}
현재 체제: {current_regime['regime']} (신뢰도: {current_regime['confidence']:.2%})
분석 상태: {oos_status}
Risk-Free Rate: {rf_status} ({current_regime['current_rf_rate']:.3f}%)"""
                
                ax1.text(0.5, 0.5, regime_text, fontsize=16, fontweight='bold',
                        ha='center', va='center', color=regime_color,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                                edgecolor=regime_color, linewidth=3))
            
            # 2. 투자 가능 구성요소
            ax2 = fig.add_subplot(gs[1, :])
            if selected and current_regime and current_regime['regime'] == 'BULL':
                components = [s['name'] for s in selected[:10]]
                rs_ratios = [s['rs_ratio'] for s in selected[:10]]
                rs_momentums = [s['rs_momentum'] for s in selected[:10]]
                
                x = np.arange(len(components))
                width = 0.35
                
                ax2.bar(x - width/2, np.array(rs_ratios) - 100, width, 
                       label='RS-Ratio', alpha=0.8)
                ax2.bar(x + width/2, np.array(rs_momentums) - 100, width, 
                       label='RS-Momentum', alpha=0.8)
                
                ax2.set_xlabel('구성요소')
                ax2.set_ylabel('100 대비 초과 수준')
                ax2.set_title(f'투자 가능 구성요소 TOP 10 (총 {len(selected)}개)')
                ax2.set_xticks(x)
                ax2.set_xticklabels(components, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, '투자 가능 구성요소 없음' if current_regime and current_regime['regime'] == 'BEAR' else '데이터 없음',
                        fontsize=16, ha='center', va='center')
                ax2.set_title('투자 가능 구성요소')
            
            # 3. 최근 체제 변화
            ax3 = fig.add_subplot(gs[2, :])
            recent_regime = jump_model.get_regime_history(
                end_date - timedelta(days=90), end_date
            )
            
            if recent_regime is not None:
                dates = recent_regime.index
                regimes = recent_regime['regime'].map({'BULL': 1, 'BEAR': 0})
                
                ax3.fill_between(dates, 0, regimes, where=(regimes == 1), 
                               color='green', alpha=0.3, label='BULL')
                ax3.fill_between(dates, 0, regimes, where=(regimes == 0), 
                               color='red', alpha=0.3, label='BEAR')
                ax3.plot(dates, regimes, 'k-', linewidth=2)
                
                ax3.set_ylim(-0.1, 1.1)
                ax3.set_yticks([0, 1])
                ax3.set_yticklabels(['BEAR', 'BULL'])
                ax3.set_xlabel('날짜')
                ax3.set_title('최근 90일 체제 변화')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 4. Risk-Free Rate 추이 (동적 RF인 경우)
            ax4 = fig.add_subplot(gs[3, :])
            if HAS_RF_UTILS and current_regime and current_regime['dynamic_rf_used']:
                try:
                    rf_manager = RiskFreeRateManager(self.rf_ticker, self.default_rf_rate)
                    rf_data = rf_manager.download_risk_free_rate(
                        end_date - timedelta(days=180), end_date
                    )
                    
                    if rf_data is not None and not rf_data.empty:
                        ax4.plot(rf_data.index, rf_data * 100, linewidth=2, color='blue')
                        ax4.set_title(f'Risk-Free Rate 추이 ({self.rf_ticker})')
                        ax4.set_ylabel('금리 (%)')
                        ax4.set_xlabel('날짜')
                        ax4.grid(True, alpha=0.3)
                        
                        # 현재 금리 표시
                        current_rf = rf_data.iloc[-1] * 100
                        ax4.axhline(y=current_rf, color='red', linestyle='--', alpha=0.7)
                        ax4.text(rf_data.index[-1], current_rf + 0.1, 
                               f'현재: {current_rf:.3f}%', 
                               ha='right', va='bottom', color='red')
                    else:
                        ax4.text(0.5, 0.5, 'RF 데이터 없음', ha='center', va='center')
                        ax4.set_title(f'Risk-Free Rate 추이 ({self.rf_ticker})')
                except Exception as e:
                    ax4.text(0.5, 0.5, f'RF 데이터 오류: {e}', ha='center', va='center')
                    ax4.set_title(f'Risk-Free Rate 추이 ({self.rf_ticker})')
            else:
                ax4.text(0.5, 0.5, f'고정 RF: {self.default_rf_rate*100:.1f}%', 
                        ha='center', va='center', fontsize=14)
                ax4.set_title('Risk-Free Rate (고정)')
            
            plt.tight_layout()
            plt.savefig(f'{self.current_preset["name"].replace(" ", "_").lower()}_dashboard_dynamic_rf.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # 투자 권고 (동적 RF 고려)
            print("\n=== 투자 권고 (동적 Risk-Free Rate 고려) ===")
            if current_regime and current_regime['regime'] == 'BULL' and selected:
                print(f"✅ 투자 실행 권고")
                print(f"   - 투자 가능 구성요소: {len(selected)}개")
                print(f"   - 권고사항: 선택된 구성요소에 동일 가중 분산 투자")
                
                # RF 수준별 추가 권고
                if current_regime['dynamic_rf_used']:
                    rf_level = current_regime['current_rf_rate']
                    if rf_level > 4.0:
                        print(f"   - 🔶 높은 금리 환경 ({rf_level:.2f}%): 보수적 포지션 사이징 권고")
                    elif rf_level < 1.0:
                        print(f"   - 🔷 낮은 금리 환경 ({rf_level:.2f}%): 적극적 투자 기회")
                    else:
                        print(f"   - 🔸 보통 금리 환경 ({rf_level:.2f}%): 표준 포지션 사이징")
                        
            elif current_regime and current_regime['regime'] == 'BEAR':
                print(f"❌ 투자 중단 권고")
                print(f"   - 시장 체제: BEAR (하락장)")
                print(f"   - 권고사항: 모든 포지션 청산 후 현금 보유")
                
                if current_regime['dynamic_rf_used']:
                    rf_level = current_regime['current_rf_rate']
                    print(f"   - 현금 수익률: {rf_level:.3f}% (Risk-Free Rate)")
            else:
                print(f"⚠️ 대기 권고")
                print(f"   - 투자 가능한 구성요소가 없음")


def main():
    """메인 실행 함수"""
    runner = UniversalStrategyRunner()
    
    print("\n" + "="*80)
    print("범용 RS 모멘텀 전략 + Jump Model + 동적 Risk-Free Rate 시스템")
    print("다양한 시장과 자산군에 적용 가능한 전략")
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
                runner.dynamic_rf_performance_analysis()
            elif choice == '9':
                print("\n프로그램을 종료합니다.")
                print("동적 Risk-Free Rate 시스템을 사용해주셔서 감사합니다!")
                break
            else:
                print("\n잘못된 선택입니다. 다시 선택해주세요.")
                
        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")
        
        input("\n계속하려면 엔터를 누르세요...")


if __name__ == "__main__":
    main()
