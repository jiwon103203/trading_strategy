"""
코스피 섹터 RS 모멘텀 전략 + Jump Model
메인 실행 스크립트
"""

import pandas as pd
from datetime import datetime, timedelta
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    font_path = '/content/drive/MyDrive/NanumSquare_acR.ttf'
    font_manager.fontManager.addfont(font_path)
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['axes.unicode_minus'] = False

def print_menu():
    """메뉴 출력"""
    print("\n" + "="*70)
    print("코스피 섹터 RS 모멘텀 전략 + Jump Model 시스템")
    print("="*70)
    print("1. 현재 시장 상태 및 투자 신호 확인")
    print("2. Jump Model 체제 분석")
    print("3. 백테스트 실행 (Jump Model 적용)")
    print("4. 전략 비교 (Jump Model 유/무)")
    print("5. 실시간 대시보드")
    print("6. 종료")
    print("="*70)

def check_current_status():
    """현재 시장 상태 확인"""
    from realtime_with_jump_model import RealtimeRSWithJumpModel

    analyzer = RealtimeRSWithJumpModel()

    print("\n현재 시장 상태를 분석하고 있습니다...")
    status = analyzer.get_current_market_status()

    # 투자 권고
    print("\n" + "="*60)
    print("=== 투자 권고 ===")

    if status['regime'] == 'BULL':
        if status['buy_count'] > 0:
            print(f"\n✅ 투자 실행 권고")
            print(f"   - 시장 체제: {status['regime']} (상승장)")
            print(f"   - 투자 가능 섹터: {status['buy_count']}개")
            print(f"   - 권고사항: 선택된 섹터에 동일 가중 분산 투자")

            # 포트폴리오 구성안
            if status['signals'] is not None:
                buy_signals = status['signals'][status['signals']['signal'] == 'BUY'].head(5)
                if not buy_signals.empty:
                    print(f"\n추천 포트폴리오 (상위 5개):")
                    allocation = 100 / len(buy_signals)
                    for _, signal in buy_signals.iterrows():
                        print(f"   - {signal['name']}: {allocation:.1f}%")
        else:
            print(f"\n⚠️ 대기 권고")
            print(f"   - 시장 체제: {status['regime']} (상승장)")
            print(f"   - 투자 가능 섹터: 없음")
            print(f"   - 권고사항: 조건 충족 섹터 나타날 때까지 현금 보유")
    else:
        print(f"\n❌ 투자 중단 권고")
        print(f"   - 시장 체제: {status['regime']} (하락장)")
        print(f"   - 권고사항: 모든 포지션 청산 후 현금 보유")
        print(f"   - 이유: 시장 변동성 증가 및 하방 리스크 확대")

def analyze_regime():
    """Jump Model 체제 분석"""
    from kospijumpmodel import KospiJumpModel

    print("\n체제 분석 기간을 선택하세요:")
    print("1. 최근 1년")
    print("2. 최근 3년")
    print("3. 최근 5년")

    choice = input("선택 (1-3): ")

    if choice == '1':
        years = 1
    elif choice == '2':
        years = 3
    else:
        years = 5

    jump_model = KospiJumpModel(jump_penalty=50.0)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)

    print(f"\n{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} 체제 분석 중...")

    regime_history = jump_model.get_regime_history(start_date, end_date)

    if regime_history is not None:
        # 체제 전환 분석
        regime_changes = regime_history[regime_history['regime'] != regime_history['regime'].shift()]

        print(f"\n=== 체제 전환 이력 ===")
        for date, row in regime_changes.tail(10).iterrows():
            print(f"{date.strftime('%Y-%m-%d')}: {row['regime']} 체제 시작")

        # 통계
        bull_days = (regime_history['regime'] == 'BULL').sum()
        bear_days = (regime_history['regime'] == 'BEAR').sum()

        print(f"\n=== 체제별 통계 ===")
        print(f"BULL 체제: {bull_days}일 ({bull_days/len(regime_history)*100:.1f}%)")
        print(f"BEAR 체제: {bear_days}일 ({bear_days/len(regime_history)*100:.1f}%)")

        # 평균 지속 기간
        regime_durations = []
        current_regime = regime_changes.iloc[0]['regime']
        duration = 0

        for date, regime in regime_history['regime'].items():
            if regime == current_regime:
                duration += 1
            else:
                regime_durations.append((current_regime, duration))
                current_regime = regime
                duration = 1

        bull_durations = [d for r, d in regime_durations if r == 'BULL']
        bear_durations = [d for r, d in regime_durations if r == 'BEAR']

        if bull_durations:
            print(f"\nBULL 체제 평균 지속 기간: {np.mean(bull_durations):.0f}일")
        if bear_durations:
            print(f"BEAR 체제 평균 지속 기간: {np.mean(bear_durations):.0f}일")

def run_backtest_with_jump():
    """Jump Model을 적용한 백테스트"""
    from rs_kospijump import RSStrategyWithJumpModel
    from rs_momentum_visualization import visualize_strategy_results

    print("\n백테스트 설정:")

    # RS 전략 설정
    print("\n1. RS 전략 설정")
    timeframe = input("   시간 프레임 (1: 일봉, 2: 주봉): ")
    timeframe = 'daily' if timeframe == '1' else 'weekly'

    use_cross = input("   최근 크로스 필터링 사용? (y/n): ")
    if use_cross.lower() == 'y':
        cross_days = int(input("   크로스 기간 (일): "))
    else:
        cross_days = None

    # Jump Model 설정
    print("\n2. Jump Model 설정")
    jump_penalty = float(input("   Jump Penalty (기본: 50): ") or "50")

    # 백테스트 기간
    print("\n3. 백테스트 기간")
    print("   1. 최근 1년")
    print("   2. 최근 3년")
    print("   3. 최근 5년")
    print("   4. 사용자 지정")

    period = input("   선택 (1-4): ")

    end_date = datetime.now()
    if period == '1':
        start_date = end_date - timedelta(days=365)
    elif period == '2':
        start_date = end_date - timedelta(days=365*3)
    elif period == '3':
        start_date = end_date - timedelta(days=365*5)
    else:
        start_year = int(input("   시작 연도: "))
        start_month = int(input("   시작 월: "))
        start_date = datetime(start_year, start_month, 1)

    # 전략 생성 및 백테스트
    print(f"\n백테스트 실행 중...")
    print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

    strategy = RSStrategyWithJumpModel(
        rs_timeframe=timeframe,
        rs_recent_cross_days=cross_days,
        jump_penalty=jump_penalty
    )

    portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)

    if portfolio_df is not None and not portfolio_df.empty:
        # 성과 출력
        metrics = strategy.calculate_performance_metrics(portfolio_df)

        print("\n=== 백테스트 결과 ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # 시각화
        viz_choice = input("\n결과를 시각화하시겠습니까? (y/n): ")
        if viz_choice.lower() == 'y':
            visualize_strategy_results(portfolio_df, trades_df)

        # 저장
        save_choice = input("\n결과를 저장하시겠습니까? (y/n): ")
        if save_choice.lower() == 'y':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            portfolio_df.to_csv(f'portfolio_jump_{timestamp}.csv')
            if not trades_df.empty:
                trades_df.to_csv(f'trades_jump_{timestamp}.csv')
            print("저장 완료!")

def compare_strategies():
    """Jump Model 유/무 전략 비교"""
    from rs_kospijump import RSStrategyWithJumpModel
    from enhancedkospi import EnhancedKospiSectorRSStrategy
    import matplotlib.pyplot as plt

    print("\n전략 비교 백테스트 설정:")

    # 기간 설정
    years = int(input("백테스트 기간 (년): "))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)

    print(f"\n백테스트 실행 중...")
    print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

    results = {}

    # 1. 기본 RS 전략
    print("\n1. 기본 RS 전략 백테스트...")
    basic_strategy = EnhancedKospiSectorRSStrategy(
        timeframe='daily',
        recent_cross_days=30
    )

    basic_portfolio, basic_trades = basic_strategy.backtest(start_date, end_date)

    if basic_portfolio is not None:
        basic_metrics = basic_strategy.calculate_performance_metrics(basic_portfolio)
        results['기본 RS 전략'] = {
            'portfolio': basic_portfolio,
            'metrics': basic_metrics
        }

    # 2. Jump Model 적용 전략
    print("\n2. Jump Model 적용 전략 백테스트...")
    jump_strategy = RSStrategyWithJumpModel(
        rs_timeframe='daily',
        rs_recent_cross_days=30,
        jump_penalty=50.0
    )

    jump_portfolio, jump_trades, _ = jump_strategy.backtest(start_date, end_date)

    if jump_portfolio is not None:
        jump_metrics = jump_strategy.calculate_performance_metrics(jump_portfolio)
        results['Jump Model 전략'] = {
            'portfolio': jump_portfolio,
            'metrics': jump_metrics
        }

    # 결과 비교
    if len(results) == 2:
        print("\n=== 전략 비교 결과 ===")
        print(f"{'지표':<20} {'기본 RS':<15} {'Jump Model':<15}")
        print("-" * 50)

        # 메트릭 비교
        metrics_to_compare = ['총 수익률', '연율화 수익률', '연율화 변동성', '샤프 비율', '최대 낙폭']

        for metric in metrics_to_compare:
            basic_val = results['기본 RS 전략']['metrics'].get(metric, 'N/A')
            jump_val = results['Jump Model 전략']['metrics'].get(metric, 'N/A')
            print(f"{metric:<20} {basic_val:<15} {jump_val:<15}")

        # 시각화
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 포트폴리오 가치
        ax1.plot(results['기본 RS 전략']['portfolio'].index,
                results['기본 RS 전략']['portfolio']['value'],
                label='기본 RS 전략', linewidth=2)
        ax1.plot(results['Jump Model 전략']['portfolio'].index,
                results['Jump Model 전략']['portfolio']['value'],
                label='Jump Model 전략', linewidth=2)
        ax1.set_ylabel('포트폴리오 가치')
        ax1.set_title('전략별 포트폴리오 가치 추이')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 상대 성과
        ax2.plot(results['기본 RS 전략']['portfolio'].index,
                (results['Jump Model 전략']['portfolio']['value'] /
                 results['기본 RS 전략']['portfolio']['value'] - 1) * 100,
                'g-', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('상대 성과 (%)')
        ax2.set_xlabel('날짜')
        ax2.set_title('Jump Model 전략의 상대 성과')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def show_dashboard():
    """실시간 대시보드"""
    from rs_kospijump_visualize import RealtimeRSWithJumpModel

    analyzer = RealtimeRSWithJumpModel()

    print("\n대시보드를 생성하고 있습니다...")

    # 1. 현재 상태
    analyzer.get_current_market_status()

    # 2. 체제 분석 차트
    print("\n체제 분석 차트 생성 중...")
    analyzer.visualize_regime_and_performance(lookback_days=365)

    # 3. 종합 대시보드
    print("\n종합 대시보드 생성 중...")
    analyzer.create_dashboard()

    print("\n대시보드 생성이 완료되었습니다.")

def main():
    """메인 실행 함수"""
    import numpy as np

    print("\n" + "="*70)
    print("코스피 섹터 RS 모멘텀 전략 + Jump Model 시스템")
    print("Pine Script RS 지표 + 시장 체제 감지 통합 전략")
    print("="*70)

    while True:
        print_menu()
        choice = input("\n선택: ")

        try:
            if choice == '1':
                check_current_status()
            elif choice == '2':
                analyze_regime()
            elif choice == '3':
                run_backtest_with_jump()
            elif choice == '4':
                compare_strategies()
            elif choice == '5':
                show_dashboard()
            elif choice == '6':
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
