"""
범용 RS 모멘텀 전략 + Jump Model
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
    """범용 전략 실행기"""
    
    def __init__(self):
        self.current_preset = None
        self.presets = {
            1: PresetManager.get_sp500_sectors(),
            2: PresetManager.get_kospi_sectors(),
            3: PresetManager.get_msci_countries(),
            4: PresetManager.get_europe_sectors(),
            5: PresetManager.get_global_sectors(),
            6: PresetManager.get_emerging_markets(),
            7: PresetManager.get_commodity_sectors(),
            8: PresetManager.get_crypto_assets(),
            9: PresetManager.get_factor_etfs(),
            10: PresetManager.get_thematic_etfs()
        }
    
    def print_menu(self):
        """메뉴 출력"""
        print("\n" + "="*70)
        print("범용 RS 모멘텀 전략 + Jump Model 시스템")
        print("="*70)
        print("1. 프리셋 선택 및 현재 시장 상태 확인")
        print("2. Jump Model 체제 분석")
        print("3. 백테스트 실행 (Jump Model 유/무)")
        print("4. 전략 성과 비교")
        print("5. 실시간 신호 대시보드")
        print("6. 사용자 정의 전략 생성")
        print("7. 종료")
        print("="*70)
    
    def select_preset(self):
        """프리셋 선택"""
        print("\n=== 프리셋 선택 ===")
        print("1. S&P 500 섹터")
        print("2. KOSPI 섹터")
        print("3. MSCI 국가별 지수")
        print("4. 유럽 섹터")
        print("5. 글로벌 섹터")
        print("6. 신흥시장")
        print("7. 원자재 섹터")
        print("8. 암호화폐 관련")
        print("9. 팩터 ETF")
        print("10. 테마 ETF")
        print("11. 사용자 정의 로드")
        
        choice = input("\n선택 (1-11): ")
        
        if choice == '11':
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
    
    def check_current_status(self):
        """현재 시장 상태 확인"""
        if not self.current_preset:
            self.select_preset()
        
        if not self.current_preset:
            return
        
        print(f"\n{self.current_preset['name']} 현재 상태 분석 중...")
        
        # Jump Model로 체제 확인
        jump_model = UniversalJumpModel(
            benchmark_ticker=self.current_preset['benchmark'],
            benchmark_name=self.current_preset['name']
        )
        
        current_regime = jump_model.get_current_regime()
        
        if current_regime:
            print(f"\n시장 체제: {current_regime['regime']} (신뢰도: {current_regime['confidence']:.2%})")
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
            
            if price_data and benchmark_data:
                selected = strategy.select_components(price_data, benchmark_data, end_date)
                
                if selected:
                    print(f"\n투자 가능 구성요소 ({len(selected)}개):")
                    for comp in selected[:5]:  # 상위 5개만 표시
                        print(f"  - {comp['name']}: RS-Ratio={comp['rs_ratio']:.1f}, RS-Momentum={comp['rs_momentum']:.1f}")
                else:
                    print("\n투자 가능한 구성요소가 없습니다.")
        elif current_regime:
            print(f"\n{current_regime['regime']} 체제 - 투자 중단 권고")
    
    def analyze_regime(self):
        """Jump Model 체제 분석"""
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
            benchmark_name=self.current_preset['name']
        )
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        print(f"\n{self.current_preset['name']} 체제 분석 중...")
        
        stats = jump_model.get_regime_statistics(start_date, end_date)
        
        if stats:
            print(f"\n=== {self.current_preset['name']} 체제 통계 ({years}년) ===")
            for regime, data in stats.items():
                print(f"\n{regime}:")
                print(f"  - 총 기간: {data['total_days']}일 ({data['percentage']:.1f}%)")
                print(f"  - 평균 지속: {data['avg_duration']:.0f}일")
                print(f"  - 최대 지속: {data['max_duration']}일")
                print(f"  - 전환 횟수: {data['transitions']}회")
    
    def run_backtest(self):
        """백테스트 실행"""
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
        
        strategy = UniversalRSWithJumpModel(
            preset_config=self.current_preset,
            rs_timeframe=timeframe,
            rs_recent_cross_days=cross_days,
            jump_penalty=jump_penalty,
            use_jump_model=use_jump
        )
        
        portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date, initial_capital)
        
        if portfolio_df is not None and not portfolio_df.empty:
            # 성과 출력
            metrics = strategy.calculate_performance_metrics(portfolio_df)
            
            print("\n=== 백테스트 결과 ===")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            
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
    
    def visualize_results(self, portfolio_df, trades_df, strategy_name):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{strategy_name} 백테스트 결과', fontsize=16)
        
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
        
        plt.tight_layout()
        plt.savefig(f'{strategy_name.replace(" ", "_").lower()}_results.png', dpi=300, bbox_inches='tight')
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
        """전략 성과 비교"""
        print("\n=== 전략 성과 비교 ===")
        
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
        
        print(f"\n백테스트 실행 중...")
        
        results = {}
        
        # 두 전략 백테스트
        for i, preset in enumerate([preset1, preset2], 1):
            print(f"\n{i}. {preset['name']} 백테스트...")
            
            strategy = UniversalRSWithJumpModel(
                preset_config=preset,
                rs_timeframe='daily',
                rs_recent_cross_days=30,
                jump_penalty=50.0,
                use_jump_model=True
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
        """전략 비교 결과 표시"""
        print("\n=== 전략 비교 결과 ===")
        
        # 테이블 형식으로 출력
        strategies = list(results.keys())
        print(f"\n{'지표':<20} {strategies[0]:<30} {strategies[1]:<30}")
        print("-" * 80)
        
        metrics_to_compare = ['총 수익률', '연율화 수익률', '연율화 변동성', '샤프 비율', '최대 낙폭']
        
        for metric in metrics_to_compare:
            val1 = results[strategies[0]]['metrics'].get(metric, 'N/A')
            val2 = results[strategies[1]]['metrics'].get(metric, 'N/A')
            print(f"{metric:<20} {val1:<30} {val2:<30}")
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 포트폴리오 가치
        for strategy, data in results.items():
            ax1.plot(data['portfolio'].index, data['portfolio']['value'], 
                    label=strategy, linewidth=2)
        ax1.set_ylabel('포트폴리오 가치')
        ax1.set_title('전략별 포트폴리오 가치 추이')
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
        plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dashboard(self):
        """실시간 대시보드"""
        if not self.current_preset:
            self.select_preset()
        
        if not self.current_preset:
            return
        
        print(f"\n{self.current_preset['name']} 대시보드 생성 중...")
        
        # Jump Model
        jump_model = UniversalJumpModel(
            benchmark_ticker=self.current_preset['benchmark'],
            benchmark_name=self.current_preset['name']
        )
        
        # 현재 체제
        current_regime = jump_model.get_current_regime()
        
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
        
        if price_data and benchmark_data:
            selected = strategy.select_components(price_data, benchmark_data, end_date)
            
            # 대시보드 생성
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. 현재 체제
            ax1 = fig.add_subplot(gs[0, :])
            ax1.axis('off')
            
            if current_regime:
                regime_color = 'green' if current_regime['regime'] == 'BULL' else 'red'
                regime_text = f"{self.current_preset['name']}\n현재 체제: {current_regime['regime']} (신뢰도: {current_regime['confidence']:.2%})"
                ax1.text(0.5, 0.5, regime_text, fontsize=20, fontweight='bold',
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
            
            plt.tight_layout()
            plt.savefig(f'{self.current_preset["name"].replace(" ", "_").lower()}_dashboard.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # 투자 권고
            print("\n=== 투자 권고 ===")
            if current_regime and current_regime['regime'] == 'BULL' and selected:
                print(f"✅ 투자 실행 권고")
                print(f"   - 투자 가능 구성요소: {len(selected)}개")
                print(f"   - 권고사항: 선택된 구성요소에 동일 가중 분산 투자")
            elif current_regime and current_regime['regime'] == 'BEAR':
                print(f"❌ 투자 중단 권고")
                print(f"   - 시장 체제: BEAR (하락장)")
                print(f"   - 권고사항: 모든 포지션 청산 후 현금 보유")
            else:
                print(f"⚠️ 대기 권고")
                print(f"   - 투자 가능한 구성요소가 없음")


def main():
    """메인 실행 함수"""
    runner = UniversalStrategyRunner()
    
    print("\n" + "="*70)
    print("범용 RS 모멘텀 전략 + Jump Model 시스템")
    print("다양한 시장과 자산군에 적용 가능한 전략")
    print("="*70)
    
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
                print("\n프로그램을 종료합니다.")
                break
            else:
                print("\n잘못된 선택입니다. 다시 선택해주세요.")
                
        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")
        
        input("\n계속하려면 엔터를 누르세요...")


if __name__ == "__main__":
    main()
