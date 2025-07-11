"""
최적화된 범용 RS 전략 시스템
메인 실행 파일
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 프리셋 및 전략 모듈
from preset_manager import PresetManager
from optimized_integrated_strategy import OptimizedIntegratedStrategy, create_integrated_strategy
from optimized_rs_strategy import OptimizedRSStrategy
from optimized_jump_model import OptimizedJumpModel, analyze_multiple_regimes

# 유틸리티 모듈
from config_manager import config_manager, ConfigProfile
from logger import get_logger, set_log_level
from data_cache import cache_stats, clear_cache
from memory_optimization import memory_monitor, clean_memory
from performance_reporter import PerformanceReporter

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


class OptimizedStrategyRunner:
    """최적화된 전략 실행기"""
    
    def __init__(self):
        self.logger = get_logger("StrategyRunner")
        self.current_preset = None
        self.last_results = None
        
        # 프리셋 목록
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
        print("최적화된 범용 RS 모멘텀 전략 시스템")
        print("="*80)
        print("1. 프리셋 선택 및 현재 시장 상태 확인")
        print("2. 시장 체제 분석 (Jump Model)")
        print("3. 백테스트 실행")
        print("4. 전략 성과 비교")
        print("5. 실시간 신호 모니터링")
        print("6. 설정 관리")
        print("7. 성능 통계 확인")
        print("8. 캐시 관리")
        print("9. 종료")
        print("="*80)
    
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
        
        choice = input("\n선택 (1-12): ")
        
        try:
            self.current_preset = self.presets[int(choice)]
            print(f"\n선택된 전략: {self.current_preset['name']}")
            print(f"벤치마크: {self.current_preset['benchmark']}")
            print(f"구성요소 수: {len(self.current_preset['components'])}")
            
            return self.current_preset
        except:
            print("잘못된 선택입니다.")
            return None
    
    def check_current_status(self):
        """현재 시장 상태 확인"""
        if not self.current_preset:
            self.select_preset()
        
        if not self.current_preset:
            return
        
        print(f"\n{self.current_preset['name']} 현재 상태 분석 중...")
        
        try:
            # 통합 전략 생성
            strategy = OptimizedIntegratedStrategy(
                preset_config=self.current_preset,
                use_jump_model=True
            )
            
            # 현재 상태 확인
            status = strategy.get_current_status()
            
            # 결과 출력
            print(f"\n=== 현재 시장 상태 ===")
            print(f"분석 일자: {status['date'].strftime('%Y-%m-%d')}")
            print(f"시장 체제: {status['regime']} (신뢰도: {status['regime_confidence']:.2%})")
            
            if status.get('is_out_of_sample'):
                print("상태: 🔮 Out-of-Sample (추론)")
            else:
                print("상태: 📚 In-Sample")
            
            if status['regime'] == 'BULL':
                print(f"\n투자 가능 구성요소: {status['selected_components']}개")
                
                if status['top_components']:
                    print("\n상위 5개 구성요소:")
                    for comp in status['top_components']:
                        print(f"  - {comp['name']}: RS-Ratio={comp['rs_ratio']:.1f}, RS-Momentum={comp['rs_momentum']:.1f}")
            else:
                print("\n❌ BEAR 시장 - 투자 중단 권고")
                
        except Exception as e:
            self.logger.error(f"상태 확인 실패: {e}")
    
    def analyze_regimes(self):
        """시장 체제 분석"""
        print("\n=== 시장 체제 분석 ===")
        
        # 분석할 시장 선택
        markets = [
            ('^GSPC', 'S&P 500'),
            ('069500.KS', 'KOSPI 200'),
            ('^DJI', 'Dow Jones'),
            ('URTH', 'MSCI World'),
            ('EEM', 'Emerging Markets'),
            ('GLD', 'Gold'),
            ('DXY', 'US Dollar Index')
        ]
        
        print("\n주요 시장 체제 분석 중...")
        
        # 체제 분석
        results = analyze_multiple_regimes([ticker for ticker, _ in markets])
        
        if not results.empty:
            # 결과 정리
            print(f"\n{'시장':<20} {'체제':<10} {'신뢰도':<10}")
            print("-" * 40)
            
            for ticker, name in markets:
                if ticker in results['ticker'].values:
                    row = results[results['ticker'] == ticker].iloc[0]
                    regime_emoji = "🟢" if row['regime'] == 'BULL' else "🔴"
                    print(f"{name:<20} {regime_emoji} {row['regime']:<7} {row.get('confidence', 0):.1%}")
        
        # 체제 분포 차트
        if not results.empty:
            self._plot_regime_distribution(results)
    
    def run_backtest(self):
        """백테스트 실행"""
        if not self.current_preset:
            self.select_preset()
        
        if not self.current_preset:
            return
        
        print("\n=== 백테스트 설정 ===")
        
        # Jump Model 사용 여부
        use_jump = input("Jump Model 사용? (y/n) [기본: y]: ") or "y"
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
        
        print(f"\n백테스트 실행 중...")
        print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print(f"Jump Model: {'활성화' if use_jump else '비활성화'}")
        
        try:
            # 전략 생성 및 백테스트
            strategy = OptimizedIntegratedStrategy(
                preset_config=self.current_preset,
                use_jump_model=use_jump
            )
            
            portfolio_df, trades_df, metrics = strategy.backtest(
                start_date, end_date, initial_capital
            )
            
            if portfolio_df is not None and not portfolio_df.empty:
                # 결과 저장
                self.last_results = {
                    'portfolio': portfolio_df,
                    'trades': trades_df,
                    'metrics': metrics,
                    'strategy_name': self.current_preset['name'],
                    'use_jump': use_jump
                }
                
                # 결과 출력
                print("\n=== 백테스트 결과 ===")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
                
                # 시각화
                self._visualize_results(portfolio_df, trades_df)
                
                # 리포트 생성
                save_report = input("\n리포트를 생성하시겠습니까? (y/n): ")
                if save_report.lower() == 'y':
                    self._generate_report(portfolio_df, trades_df, metrics)
                    
        except Exception as e:
            self.logger.error(f"백테스트 실패: {e}")
    
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
            
            strategy = OptimizedIntegratedStrategy(
                preset_config=preset,
                use_jump_model=True
            )
            
            portfolio_df, trades_df, metrics = strategy.backtest(start_date, end_date)
            
            if portfolio_df is not None and not portfolio_df.empty:
                results[preset['name']] = {
                    'portfolio': portfolio_df,
                    'trades': trades_df,
                    'metrics': metrics
                }
        
        # 결과 비교
        if len(results) == 2:
            self._display_comparison(results)
    
    def monitor_realtime(self):
        """실시간 신호 모니터링"""
        print("\n=== 실시간 투자 신호 ===")
        
        # 모니터링할 전략들
        strategies_to_monitor = []
        
        print("모니터링할 전략 선택 (쉼표로 구분, 예: 1,2,3):")
        for i, (num, preset) in enumerate(self.presets.items(), 1):
            print(f"{num}. {preset['name']}")
        
        choices = input("\n선택: ").split(',')
        
        for choice in choices:
            try:
                preset_num = int(choice.strip())
                if preset_num in self.presets:
                    strategies_to_monitor.append(self.presets[preset_num])
            except:
                pass
        
        if not strategies_to_monitor:
            print("선택된 전략이 없습니다.")
            return
        
        print(f"\n{len(strategies_to_monitor)}개 전략 모니터링 중...")
        
        for preset in strategies_to_monitor:
            print(f"\n=== {preset['name']} ===")
            
            try:
                strategy = OptimizedIntegratedStrategy(preset)
                status = strategy.get_current_status()
                
                print(f"시장 체제: {status['regime']} (신뢰도: {status['regime_confidence']:.1%})")
                
                if status['regime'] == 'BULL':
                    print(f"투자 가능: {status['selected_components']}개")
                    
                    if status['top_components']:
                        print("상위 구성요소:")
                        for comp in status['top_components'][:3]:
                            print(f"  - {comp['name']}: RS={comp['rs_score']:.1f}")
                else:
                    print("BEAR 체제 - 투자 중단")
                    
            except Exception as e:
                print(f"모니터링 실패: {e}")
    
    def manage_config(self):
        """설정 관리"""
        print("\n=== 설정 관리 ===")
        print("1. 현재 설정 보기")
        print("2. 설정 변경")
        print("3. 설정 내보내기")
        print("4. 설정 가져오기")
        print("5. 기본값으로 초기화")
        
        choice = input("\n선택: ")
        
        if choice == '1':
            # 현재 설정 표시
            print("\n=== 현재 설정 ===")
            for key, value in config_manager.config.__dict__.items():
                print(f"{key}: {value}")
                
        elif choice == '2':
            # 설정 변경
            print("\n변경할 설정:")
            print("1. RS 계산 기간")
            print("2. Jump Model 페널티")
            print("3. 초기 자본")
            print("4. 거래 비용")
            print("5. 로그 레벨")
            
            sub_choice = input("\n선택: ")
            
            if sub_choice == '1':
                new_value = int(input("새 RS 계산 기간 (5-100): "))
                config_manager.set('rs_length', new_value)
            elif sub_choice == '2':
                new_value = float(input("새 Jump 페널티 (0-100): "))
                config_manager.set('jump_penalty', new_value)
            elif sub_choice == '3':
                new_value = float(input("새 초기 자본: "))
                config_manager.set('initial_capital', new_value)
            elif sub_choice == '4':
                new_value = float(input("새 거래 비용 (0-0.01): "))
                config_manager.set('transaction_cost', new_value)
            elif sub_choice == '5':
                print("로그 레벨: DEBUG, INFO, WARNING, ERROR")
                new_value = input("새 로그 레벨: ").upper()
                config_manager.set('log_level', new_value)
                set_log_level(new_value)
            
            print("설정이 저장되었습니다.")
            
        elif choice == '3':
            # 설정 내보내기
            filename = input("파일명 [기본: config_export.json]: ") or "config_export.json"
            config_manager.export_config(filename)
            print(f"설정이 {filename}으로 내보내졌습니다.")
            
        elif choice == '4':
            # 설정 가져오기
            filename = input("파일명: ")
            try:
                config_manager.import_config(filename)
                print("설정을 가져왔습니다.")
            except Exception as e:
                print(f"설정 가져오기 실패: {e}")
                
        elif choice == '5':
            # 기본값으로 초기화
            confirm = input("정말로 초기화하시겠습니까? (y/n): ")
            if confirm.lower() == 'y':
                config_manager.config = config_manager.StrategyConfig()
                config_manager.save_config()
                print("설정이 초기화되었습니다.")
    
    def check_performance_stats(self):
        """성능 통계 확인"""
        print("\n=== 성능 통계 ===")
        
        # 캐시 통계
        print("\n1. 캐시 통계:")
        stats, size_info = cache_stats()
        
        # 메모리 통계
        print("\n2. 메모리 사용량:")
        memory_info = memory_monitor.get_memory_info()
        print(f"현재 메모리: {memory_info['rss_mb']:.1f} MB")
        print(f"시스템 메모리 사용률: {memory_info['percent']:.1f}%")
        print(f"사용 가능 메모리: {memory_info['available_mb']:.1f} MB")
        
        # 메모리 사용 이력
        memory_report = memory_monitor.get_memory_report()
        if not memory_report.empty:
            print("\n3. 메모리 사용 이력:")
            print(memory_report.tail())
    
    def manage_cache(self):
        """캐시 관리"""
        print("\n=== 캐시 관리 ===")
        print("1. 캐시 통계 보기")
        print("2. 캐시 클리어")
        print("3. 메모리 정리")
        
        choice = input("\n선택: ")
        
        if choice == '1':
            cache_stats()
        elif choice == '2':
            confirm = input("캐시를 클리어하시겠습니까? (y/n): ")
            if confirm.lower() == 'y':
                clear_cache()
                print("캐시가 클리어되었습니다.")
        elif choice == '3':
            clean_memory()
    
    def _visualize_results(self, portfolio_df, trades_df):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.current_preset["name"]} 백테스트 결과', fontsize=16)
        
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
        
        # 4. 체제별 포트폴리오
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
        
        # 저장 옵션
        save = input("\n차트를 저장하시겠습니까? (y/n): ")
        if save.lower() == 'y':
            filename = f"{self.current_preset['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"차트가 {filename}으로 저장되었습니다.")
        
        plt.show()
    
    def _plot_regime_distribution(self, results):
        """체제 분포 차트"""
        plt.figure(figsize=(10, 6))
        
        # 체제별 카운트
        regime_counts = results['regime'].value_counts()
        
        # 파이 차트
        colors = {'BULL': 'green', 'BEAR': 'red', 'ERROR': 'gray', 'UNKNOWN': 'orange'}
        plt.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%',
               colors=[colors.get(r, 'gray') for r in regime_counts.index])
        
        plt.title('시장 체제 분포')
        plt.show()
    
    def _display_comparison(self, results):
        """전략 비교 결과 표시"""
        print(f"\n=== 전략 비교 결과 ===")
        
        strategies = list(results.keys())
        
        # 테이블 형식으로 출력
        print(f"\n{'지표':<25} {strategies[0]:<35} {strategies[1]:<35}")
        print("-" * 95)
        
        metrics_to_compare = [
            '총 수익률', '연율화 수익률', '연율화 변동성', 
            '샤프 비율', '최대 낙폭', 'BULL 기간', 'BEAR 기간'
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
        ax1.set_title('전략별 포트폴리오 가치 추이')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 상대 성과
        portfolio1 = results[strategies[0]]['portfolio']
        portfolio2 = results[strategies[1]]['portfolio']
        
        # 공통 날짜
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
        plt.show()
    
    def _generate_report(self, portfolio_df, trades_df, metrics):
        """성과 리포트 생성"""
        reporter = PerformanceReporter(
            strategy_name=self.current_preset['name'],
            portfolio_df=portfolio_df,
            trades_df=trades_df
        )
        
        # HTML 리포트
        html_file = reporter.generate_html_report()
        print(f"HTML 리포트 생성: {html_file}")
        
        # PDF 리포트
        pdf_file = reporter.generate_pdf_report()
        print(f"PDF 리포트 생성: {pdf_file}")
        
        # CSV 저장
        csv_file = reporter.save_metrics_csv()
        print(f"성과 지표 CSV: {csv_file}")


def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("최적화된 범용 RS 모멘텀 전략 시스템")
    print("고성능 백테스팅 및 실시간 분석")
    print("="*80)
    
    # 로그 레벨 설정
    set_log_level(config_manager.get('log_level', 'INFO'))
    
    # 실행기 생성
    runner = OptimizedStrategyRunner()
    
    while True:
        runner.print_menu()
        choice = input("\n선택: ")
        
        try:
            if choice == '1':
                runner.check_current_status()
            elif choice == '2':
                runner.analyze_regimes()
            elif choice == '3':
                runner.run_backtest()
            elif choice == '4':
                runner.compare_strategies()
            elif choice == '5':
                runner.monitor_realtime()
            elif choice == '6':
                runner.manage_config()
            elif choice == '7':
                runner.check_performance_stats()
            elif choice == '8':
                runner.manage_cache()
            elif choice == '9':
                print("\n프로그램을 종료합니다.")
                print("최적화된 시스템을 사용해주셔서 감사합니다!")
                
                # 종료 전 정리
                clean_memory()
                break
            else:
                print("\n잘못된 선택입니다. 다시 선택해주세요.")
                
        except KeyboardInterrupt:
            print("\n\n프로그램이 중단되었습니다.")
            break
        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")
        
        input("\n계속하려면 엔터를 누르세요...")


if __name__ == "__main__":
    main()
