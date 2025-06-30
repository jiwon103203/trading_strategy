"""
범용 RS 전략 사용 예시 및 빠른 시작 가이드
"""

from datetime import datetime, timedelta
from preset_manager import PresetManager
from universal_rs_strategy import UniversalRSStrategy
from universal_jump_model import UniversalJumpModel
from universal_rs_with_jump import UniversalRSWithJumpModel

# ===== 1. 기본 사용 예시 =====

def example_sp500_sectors():
    """S&P 500 섹터 전략 예시"""
    print("=== S&P 500 섹터 RS 전략 예시 ===\n")
    
    # 프리셋 가져오기
    sp500_preset = PresetManager.get_sp500_sectors()
    
    # 전략 생성
    strategy = UniversalRSStrategy(
        benchmark=sp500_preset['benchmark'],
        components=sp500_preset['components'],
        name=sp500_preset['name'],
        length=20,
        timeframe='daily',
        recent_cross_days=30
    )
    
    # 백테스트
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3년
    
    print(f"백테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    
    portfolio_df, trades_df = strategy.backtest(start_date, end_date)
    
    if portfolio_df is not None:
        metrics = strategy.calculate_performance_metrics(portfolio_df)
        print("\n백테스트 결과:")
        for key, value in metrics.items():
            print(f"{key}: {value}")


def example_msci_countries():
    """MSCI 국가별 지수 전략 예시"""
    print("\n=== MSCI 국가별 RS 전략 예시 ===\n")
    
    # 프리셋 가져오기
    msci_preset = PresetManager.get_msci_countries()
    
    # Jump Model과 함께 전략 생성
    strategy = UniversalRSWithJumpModel(
        preset_config=msci_preset,
        rs_length=20,
        rs_timeframe='weekly',  # 주봉 사용
        jump_penalty=50.0,
        use_jump_model=True
    )
    
    # 백테스트
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2년
    
    print(f"백테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print("Jump Model 활성화")
    
    portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)
    
    if portfolio_df is not None:
        metrics = strategy.calculate_performance_metrics(portfolio_df)
        print("\n백테스트 결과:")
        for key, value in metrics.items():
            print(f"{key}: {value}")


def example_custom_strategy():
    """사용자 정의 전략 예시"""
    print("\n=== 사용자 정의 전략 예시 ===\n")
    
    # 미국 대형 기술주 ETF 전략
    custom_config = {
        'name': 'US Tech Giants Strategy',
        'benchmark': 'QQQ',  # Nasdaq 100
        'components': {
            'XLK': 'Technology Select Sector',
            'VGT': 'Vanguard Information Technology',
            'IGV': 'iShares Expanded Tech-Software',
            'SOXX': 'iShares Semiconductor',
            'SKYY': 'First Trust Cloud Computing',
            'HACK': 'ETFMG Prime Cyber Security',
            'ROBO': 'ROBO Global Robotics & Automation',
            'ARKK': 'ARK Innovation ETF'
        }
    }
    
    # 전략 생성
    strategy = UniversalRSStrategy(
        benchmark=custom_config['benchmark'],
        components=custom_config['components'],
        name=custom_config['name'],
        length=15,  # 더 짧은 기간
        timeframe='daily'
    )
    
    # 현재 투자 가능한 ETF 확인
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    price_data, benchmark_data = strategy.get_price_data(start_date, end_date)
    
    if price_data and benchmark_data:
        selected = strategy.select_components(price_data, benchmark_data, end_date)
        
        print("현재 투자 가능한 ETF:")
        if selected:
            for comp in selected:
                print(f"  - {comp['name']}: RS-Ratio={comp['rs_ratio']:.1f}, RS-Momentum={comp['rs_momentum']:.1f}")
        else:
            print("  투자 가능한 ETF가 없습니다.")


def example_regime_detection():
    """시장 체제 감지 예시"""
    print("\n=== 시장 체제 감지 예시 ===\n")
    
    # 여러 시장의 현재 체제 확인
    markets = [
        ('^GSPC', 'S&P 500'),
        ('069500.KS', 'KOSPI 200'),
        ('URTH', 'MSCI World'),
        ('EEM', 'Emerging Markets'),
        ('^DJI', 'Dow Jones')
    ]
    
    for ticker, name in markets:
        try:
            jump_model = UniversalJumpModel(
                benchmark_ticker=ticker,
                benchmark_name=name,
                jump_penalty=50.0
            )
            
            current = jump_model.get_current_regime()
            
            if current:
                print(f"{name}: {current['regime']} (신뢰도: {current['confidence']:.2%})")
            else:
                print(f"{name}: 데이터 없음")
        except:
            print(f"{name}: 분석 실패")


def example_compare_strategies():
    """전략 비교 예시"""
    print("\n=== 전략 비교 예시 ===\n")
    
    # 섹터 vs 국가 전략 비교
    sp500_preset = PresetManager.get_sp500_sectors()
    msci_preset = PresetManager.get_msci_countries()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    results = {}
    
    # S&P 500 섹터 전략
    strategy1 = UniversalRSWithJumpModel(
        preset_config=sp500_preset,
        use_jump_model=True
    )
    
    portfolio1, _, _ = strategy1.backtest(start_date, end_date)
    if portfolio1 is not None:
        results['S&P 500 Sectors'] = strategy1.calculate_performance_metrics(portfolio1)
    
    # MSCI 국가 전략
    strategy2 = UniversalRSWithJumpModel(
        preset_config=msci_preset,
        use_jump_model=True
    )
    
    portfolio2, _, _ = strategy2.backtest(start_date, end_date)
    if portfolio2 is not None:
        results['MSCI Countries'] = strategy2.calculate_performance_metrics(portfolio2)
    
    # 결과 비교
    print("전략 비교 결과 (2년):")
    print(f"{'지표':<20} {'S&P 500 Sectors':<20} {'MSCI Countries':<20}")
    print("-" * 60)
    
    metrics = ['총 수익률', '연율화 수익률', '연율화 변동성', '샤프 비율', '최대 낙폭']
    for metric in metrics:
        val1 = results.get('S&P 500 Sectors', {}).get(metric, 'N/A')
        val2 = results.get('MSCI Countries', {}).get(metric, 'N/A')
        print(f"{metric:<20} {val1:<20} {val2:<20}")


# ===== 2. 실시간 모니터링 =====

def realtime_monitoring():
    """실시간 투자 신호 모니터링"""
    print("\n=== 실시간 투자 신호 ===\n")
    
    # 모니터링할 전략들
    strategies = [
        PresetManager.get_sp500_sectors(),
        PresetManager.get_global_sectors(),
        PresetManager.get_emerging_markets()
    ]
    
    for preset in strategies:
        print(f"\n{preset['name']}:")
        
        # Jump Model 체제 확인
        jump_model = UniversalJumpModel(
            benchmark_ticker=preset['benchmark'],
            benchmark_name=preset['name']
        )
        
        current_regime = jump_model.get_current_regime()
        
        if current_regime:
            print(f"  시장 체제: {current_regime['regime']}")
            
            if current_regime['regime'] == 'BULL':
                # RS 신호 확인
                strategy = UniversalRSStrategy(
                    benchmark=preset['benchmark'],
                    components=preset['components'],
                    name=preset['name']
                )
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)
                
                price_data, benchmark_data = strategy.get_price_data(start_date, end_date)
                
                if price_data and benchmark_data:
                    selected = strategy.select_components(price_data, benchmark_data, end_date)
                    
                    if selected:
                        print(f"  투자 가능: {len(selected)}개")
                        for comp in selected[:3]:  # 상위 3개만
                            print(f"    - {comp['name']}: RS-Ratio={comp['rs_ratio']:.1f}")
                    else:
                        print("  투자 가능한 구성요소 없음")
            else:
                print("  BEAR 체제 - 투자 중단")


# ===== 3. 빠른 시작 함수 =====

def quick_backtest(preset_name='sp500', years=3, use_jump=True):
    """
    빠른 백테스트 실행
    
    Parameters:
    - preset_name: 'sp500', 'kospi', 'msci', 'global', 'emerging'
    - years: 백테스트 기간 (년)
    - use_jump: Jump Model 사용 여부
    """
    presets = {
        'sp500': PresetManager.get_sp500_sectors(),
        'kospi': PresetManager.get_kospi_sectors(),
        'msci': PresetManager.get_msci_countries(),
        'global': PresetManager.get_global_sectors(),
        'emerging': PresetManager.get_emerging_markets()
    }
    
    preset = presets.get(preset_name)
    if not preset:
        print(f"알 수 없는 프리셋: {preset_name}")
        return
    
    print(f"\n{preset['name']} 백테스트 ({years}년)")
    print(f"Jump Model: {'활성화' if use_jump else '비활성화'}")
    
    strategy = UniversalRSWithJumpModel(
        preset_config=preset,
        use_jump_model=use_jump
    )
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    
    portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)
    
    if portfolio_df is not None:
        metrics = strategy.calculate_performance_metrics(portfolio_df)
        print("\n결과:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        return portfolio_df, trades_df, metrics
    
    return None, None, None


# ===== 메인 실행 =====

if __name__ == "__main__":
    print("범용 RS 전략 사용 예시\n")
    
    # 1. S&P 500 섹터 전략
    example_sp500_sectors()
    
    # 2. MSCI 국가별 전략
    example_msci_countries()
    
    # 3. 사용자 정의 전략
    example_custom_strategy()
    
    # 4. 시장 체제 감지
    example_regime_detection()
    
    # 5. 전략 비교
    example_compare_strategies()
    
    # 6. 실시간 모니터링
    realtime_monitoring()
    
    # 7. 빠른 백테스트
    print("\n=== 빠른 백테스트 예시 ===")
    portfolio, trades, metrics = quick_backtest('sp500', years=1)
    
    print("\n모든 예시 실행 완료!")