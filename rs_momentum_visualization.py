import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import platform
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

def visualize_strategy_results(portfolio_df, trades_df, benchmark_data=None):
    """전략 결과 시각화"""
    # 입력 검증
    if portfolio_df is None or portfolio_df.empty:
        print("포트폴리오 데이터가 비어있습니다.")
        return
    
    # returns 컬럼 확인 및 생성
    if 'returns' not in portfolio_df.columns:
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('코스피 섹터 RS 모멘텀 전략 백테스트 결과', fontsize=16)
    
    # 1. 포트폴리오 가치 추이
    ax1 = axes[0, 0]
    ax1.plot(portfolio_df.index, portfolio_df['value'], label='포트폴리오', linewidth=2)
    if benchmark_data is not None:
        try:
            # 벤치마크 정규화 - Series/DataFrame 모두 처리
            if isinstance(benchmark_data, pd.DataFrame):
                bench_values = benchmark_data['Close'] if 'Close' in benchmark_data.columns else benchmark_data.iloc[:, 0]
            else:
                bench_values = benchmark_data
            
            # 포트폴리오와 같은 기간만 선택
            common_dates = portfolio_df.index.intersection(bench_values.index)
            if len(common_dates) > 0:
                bench_aligned = bench_values.loc[common_dates]
                initial_value = portfolio_df['value'].iloc[0]
                bench_norm = bench_aligned / bench_aligned.iloc[0] * initial_value
                ax1.plot(bench_norm.index, bench_norm.values, label='KOSPI 200', 
                        linewidth=2, alpha=0.7)
        except Exception as e:
            print(f"벤치마크 표시 중 에러: {e}")
    ax1.set_title('포트폴리오 가치 추이')
    ax1.set_ylabel('가치 (원)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 수익률 추이
    ax2 = axes[0, 1]
    cumulative_returns = (portfolio_df['value'] / portfolio_df['value'].iloc[0] - 1) * 100
    ax2.plot(portfolio_df.index, cumulative_returns, linewidth=2)
    ax2.set_title('누적 수익률')
    ax2.set_ylabel('수익률 (%)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 3. 보유 섹터 수
    ax3 = axes[1, 0]
    ax3.fill_between(portfolio_df.index, portfolio_df['holdings'], alpha=0.5)
    ax3.set_title('보유 섹터 수')
    ax3.set_ylabel('섹터 수')
    ax3.set_ylim(0, 11)
    ax3.grid(True, alpha=0.3)
    
    # 4. 낙폭 (Drawdown)
    ax4 = axes[1, 1]
    # returns 컬럼이 없으면 계산
    if 'returns' not in portfolio_df.columns:
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
    
    cumulative = (1 + portfolio_df['returns'].fillna(0)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    ax4.fill_between(drawdown.index, drawdown.values, alpha=0.5, color='red')
    ax4.set_title('낙폭 (Drawdown)')
    ax4.set_ylabel('낙폭 (%)')
    ax4.grid(True, alpha=0.3)
    
    # 5. 월별 수익률 히트맵
    ax5 = axes[2, 0]
    monthly_returns = portfolio_df['returns'].resample('M').apply(
        lambda x: (1 + x).prod() - 1) * 100
    
    # Series를 DataFrame으로 변환하여 pivot 생성
    monthly_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    
    monthly_pivot = monthly_df.pivot_table(
        index='year',
        columns='month',
        values='return',
        aggfunc='first'
    )
    
    # 히트맵 생성
    sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                center=0, ax=ax5, cbar_kws={'label': '수익률 (%)'})
    ax5.set_title('월별 수익률 히트맵')
    ax5.set_xlabel('월')
    ax5.set_ylabel('연도')
    
    # 6. 섹터별 거래 빈도
    ax6 = axes[2, 1]
    if not trades_df.empty:
        sector_counts = trades_df['name'].value_counts()
        sector_counts.plot(kind='barh', ax=ax6)
        ax6.set_title('섹터별 거래 빈도')
        ax6.set_xlabel('거래 횟수')
    
    plt.tight_layout()
    plt.savefig('strategy_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_sector_performance(strategy, start_date, end_date):
    """개별 섹터의 RS 지표 분석"""
    # 데이터 다운로드
    price_data, benchmark_data = strategy.get_price_data(
        start_date - timedelta(days=100), end_date
    )
    
    if price_data is None:
        return None
    
    # 각 섹터의 RS 지표 계산
    sector_rs_data = {}
    
    for ticker, name in strategy.sector_etfs.items():
        if ticker in price_data:
            rs_components = strategy.calculate_rs_components(
                price_data[ticker], benchmark_data
            )
            if not rs_components.empty:
                sector_rs_data[name] = rs_components
    
    # RS 지표 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('섹터별 RS 지표 추이', fontsize=16)
    
    # RS-Ratio 추이
    for name, data in sector_rs_data.items():
        ax1.plot(data.index, data['rs_ratio'], label=name, alpha=0.7)
    ax1.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('RS-Ratio')
    ax1.set_ylabel('RS-Ratio')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # RS-Momentum 추이
    for name, data in sector_rs_data.items():
        ax2.plot(data.index, data['rs_momentum'], label=name, alpha=0.7)
    ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('RS-Momentum')
    ax2.set_ylabel('RS-Momentum')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sector_rs_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return sector_rs_data

def create_performance_report(portfolio_df, trades_df, metrics):
    """성과 보고서 생성"""
    report = []
    report.append("=" * 60)
    report.append("코스피 섹터 RS 모멘텀 전략 성과 보고서")
    report.append("=" * 60)
    report.append("")
    
    # 기본 정보
    report.append(f"백테스트 기간: {portfolio_df.index[0].strftime('%Y-%m-%d')} ~ {portfolio_df.index[-1].strftime('%Y-%m-%d')}")
    report.append(f"초기 자본: {portfolio_df['value'].iloc[0]:,.0f}원")
    report.append(f"최종 자본: {portfolio_df['value'].iloc[-1]:,.0f}원")
    report.append("")
    
    # 성과 지표
    report.append("### 성과 지표 ###")
    for key, value in metrics.items():
        report.append(f"{key}: {value}")
    report.append("")
    
    # 거래 통계
    if not trades_df.empty:
        report.append("### 거래 통계 ###")
        report.append(f"총 거래 횟수: {len(trades_df)}")
        report.append(f"평균 보유 섹터 수: {portfolio_df['holdings'].mean():.1f}")
        report.append(f"최대 보유 섹터 수: {portfolio_df['holdings'].max()}")
        report.append("")
        
        # 가장 많이 거래된 섹터
        report.append("### 거래 빈도 상위 5개 섹터 ###")
        top_sectors = trades_df['name'].value_counts().head()
        for sector, count in top_sectors.items():
            report.append(f"{sector}: {count}회")
    
    # 보고서 저장
    with open('performance_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))
    return report

# 확장된 사용 예시
if __name__ == "__main__":
    from datetime import datetime, timedelta
    import yfinance as yf
    
    # 메인 전략 클래스가 있다고 가정
    try:
        from kospi_sector_rs_strategy import KospiSectorRSStrategy
        
        # 전략 인스턴스 생성
        strategy = KospiSectorRSStrategy()
        
        # 백테스트 기간 설정
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        print("코스피 섹터 RS 모멘텀 전략 백테스트 시작...")
        
        # 백테스트 실행
        portfolio_df, trades_df = strategy.backtest(start_date, end_date)
        
        if portfolio_df is not None and not portfolio_df.empty:
            # 성과 지표 계산
            metrics = strategy.calculate_performance_metrics(portfolio_df)
            
            # 벤치마크 데이터 가져오기 (시각화용)
            benchmark = yf.download(strategy.benchmark, start=start_date, 
                                  end=end_date, progress=False)
            
            # 시각화
            visualize_strategy_results(portfolio_df, trades_df, benchmark)
            
            # 섹터별 RS 분석
            print("\n섹터별 RS 지표 분석 중...")
            sector_rs_data = analyze_sector_performance(strategy, start_date, end_date)
            
            # 성과 보고서 생성
            create_performance_report(portfolio_df, trades_df, metrics)
            
            print("\n모든 분석이 완료되었습니다!")
            print("생성된 파일:")
            print("- portfolio_history.csv: 포트폴리오 가치 기록")
            print("- trade_history.csv: 거래 내역")
            print("- strategy_visualization.png: 전략 성과 시각화")
            print("- sector_rs_analysis.png: 섹터별 RS 지표 분석")
            print("- performance_report.txt: 성과 보고서")
    except ImportError:
        print("메인 전략 파일을 찾을 수 없습니다.")
        print("독립 실행 모드로 샘플 데이터를 생성합니다...")
        
        # 샘플 데이터 생성
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        portfolio_df = pd.DataFrame({
            'value': 10000000 * (1 + np.random.randn(len(dates)).cumsum() * 0.01),
            'holdings': np.random.randint(0, 6, len(dates))
        }, index=dates)
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        
        trades_df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', end='2024-12-31', freq='MS'),
            'ticker': ['139220.KS'] * 60,
            'name': ['TIGER 200 IT'] * 60,
            'action': ['BUY'] * 60,
            'shares': np.random.randint(100, 1000, 60),
            'price': np.random.uniform(20000, 40000, 60)
        })
        
        visualize_strategy_results(portfolio_df, trades_df)