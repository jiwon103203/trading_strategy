"""
성과 리포트 생성기 - 동적 Risk-Free Rate 지원
HTML 및 PDF 형식의 전문적인 백테스트 리포트 생성
미국 3개월물 금리(^IRX)를 사용한 동적 Sharpe/Sortino ratio 계산
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Risk-free rate 유틸리티 import
try:
    from risk_free_rate_utils import RiskFreeRateManager, calculate_dynamic_sharpe_ratio, calculate_dynamic_sortino_ratio
    HAS_RF_UTILS = True
except ImportError:
    print("Warning: risk_free_rate_utils.py가 없습니다. 기본 risk-free rate (2%) 사용")
    HAS_RF_UTILS = False

class PerformanceReporter:
    """전문적인 성과 리포트 생성 - 동적 Risk-Free Rate 지원"""
    
    def __init__(self, strategy_name, portfolio_df, trades_df, benchmark_df=None, 
                 rf_ticker='^IRX', default_rf_rate=0.02):
        """
        Parameters:
        - strategy_name: 전략 이름
        - portfolio_df: 포트폴리오 데이터
        - trades_df: 거래 데이터
        - benchmark_df: 벤치마크 데이터 (선택사항)
        - rf_ticker: Risk-free rate 티커 (기본: ^IRX)
        - default_rf_rate: 기본 risk-free rate (기본: 2%)
        """
        self.strategy_name = strategy_name
        self.portfolio_df = portfolio_df
        self.trades_df = trades_df
        self.benchmark_df = benchmark_df
        self.rf_ticker = rf_ticker
        self.default_rf_rate = default_rf_rate
        
        # Risk-free rate 관리자 초기화
        if HAS_RF_UTILS:
            self.rf_manager = RiskFreeRateManager(rf_ticker, default_rf_rate)
            self._download_risk_free_rate()
        else:
            self.rf_manager = None
        
        self.metrics = self._calculate_all_metrics()
        
    def _download_risk_free_rate(self):
        """Risk-free rate 데이터 다운로드"""
        if self.rf_manager and not self.portfolio_df.empty:
            start_date = self.portfolio_df.index[0]
            end_date = self.portfolio_df.index[-1]
            self.rf_manager.download_risk_free_rate(start_date, end_date)
        
    def _calculate_all_metrics(self):
        """모든 성과 지표 계산 (동적 Risk-Free Rate 지원)"""
        metrics = {}
        
        if self.portfolio_df.empty:
            return metrics
        
        # 기본 지표
        metrics['start_date'] = self.portfolio_df.index[0]
        metrics['end_date'] = self.portfolio_df.index[-1]
        metrics['trading_days'] = len(self.portfolio_df)
        metrics['initial_capital'] = self.portfolio_df['value'].iloc[0]
        metrics['final_capital'] = self.portfolio_df['value'].iloc[-1]
        
        # 수익률
        metrics['total_return'] = (metrics['final_capital'] / metrics['initial_capital'] - 1) * 100
        years = (metrics['end_date'] - metrics['start_date']).days / 365.25
        metrics['annual_return'] = (np.power(1 + metrics['total_return']/100, 1/years) - 1) * 100 if years > 0 else 0
        
        # 변동성
        returns = self.portfolio_df['value'].pct_change().dropna()
        metrics['annual_volatility'] = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
        
        # 동적 Sharpe/Sortino Ratio 계산
        if HAS_RF_UTILS and self.rf_manager:
            try:
                print("동적 Risk-Free Rate를 사용한 성과 지표 계산 중...")
                
                # Sharpe Ratio (동적)
                sharpe_ratio = self.rf_manager.calculate_sharpe_ratio(returns, self.portfolio_df.index)
                metrics['sharpe_ratio'] = sharpe_ratio
                
                # Sortino Ratio (동적)
                sortino_ratio = self.rf_manager.calculate_sortino_ratio(returns, self.portfolio_df.index)
                metrics['sortino_ratio'] = sortino_ratio
                
                # Risk-free rate 통계
                rf_stats = self.rf_manager.get_risk_free_rate_stats(
                    metrics['start_date'], metrics['end_date']
                )
                metrics['avg_risk_free_rate'] = rf_stats['mean_rate']
                metrics['start_risk_free_rate'] = rf_stats['start_rate']
                metrics['end_risk_free_rate'] = rf_stats['end_rate']
                
                print(f"평균 Risk-Free Rate: {rf_stats['mean_rate']:.3f}%")
                print(f"Sharpe Ratio (동적): {sharpe_ratio:.3f}")
                print(f"Sortino Ratio (동적): {sortino_ratio:.3f}")
                
            except Exception as e:
                print(f"동적 성과 지표 계산 실패: {e}")
                # 기본 방식으로 fallback
                metrics['sharpe_ratio'] = (metrics['annual_return'] - self.default_rf_rate * 100) / metrics['annual_volatility'] if metrics['annual_volatility'] > 0 else 0
                metrics['sortino_ratio'] = self._calculate_basic_sortino(returns)
                metrics['avg_risk_free_rate'] = self.default_rf_rate * 100
        else:
            # 기본 방식 (2% 고정)
            metrics['sharpe_ratio'] = (metrics['annual_return'] - self.default_rf_rate * 100) / metrics['annual_volatility'] if metrics['annual_volatility'] > 0 else 0
            metrics['sortino_ratio'] = self._calculate_basic_sortino(returns)
            metrics['avg_risk_free_rate'] = self.default_rf_rate * 100
        
        # 낙폭
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # 최대 낙폭 기간
        dd_start = drawdown.idxmin()
        dd_end = cumulative[dd_start:].idxmax() if dd_start < cumulative.index[-1] else dd_start
        metrics['max_dd_duration'] = (dd_end - dd_start).days
        
        # 승률
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        metrics['win_rate'] = len(positive_returns) / len(returns) * 100 if len(returns) > 0 else 0
        metrics['avg_win'] = positive_returns.mean() * 100 if len(positive_returns) > 0 else 0
        metrics['avg_loss'] = negative_returns.mean() * 100 if len(negative_returns) > 0 else 0
        
        # 월별 수익률
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        metrics['best_month'] = monthly_returns.max() * 100
        metrics['worst_month'] = monthly_returns.min() * 100
        metrics['positive_months'] = (monthly_returns > 0).sum()
        metrics['total_months'] = len(monthly_returns)
        
        # 거래 통계
        if not self.trades_df.empty:
            metrics['total_trades'] = len(self.trades_df)
            metrics['avg_holding_period'] = self._calculate_avg_holding_period()
            metrics['turnover_rate'] = metrics['total_trades'] / (years * 12)  # 월평균 회전율
        else:
            metrics['total_trades'] = 0
            metrics['avg_holding_period'] = 0
            metrics['turnover_rate'] = 0
        
        return metrics
    
    def _calculate_basic_sortino(self, returns, rf_rate=None):
        """기본 Sortino ratio 계산 (동적 risk-free rate 없을 때)"""
        try:
            if rf_rate is None:
                rf_rate = self.default_rf_rate
            
            excess_returns = returns - rf_rate/252  # 일일 risk-free rate
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                annual_excess_return = excess_returns.mean() * 252
                return annual_excess_return / downside_deviation if downside_deviation > 0 else 0
            else:
                return float('inf')  # 하방 변동성이 0인 경우
                
        except Exception:
            return 0.0
    
    def _calculate_avg_holding_period(self):
        """평균 보유 기간 계산"""
        if self.trades_df.empty:
            return 0
        
        buy_trades = self.trades_df[self.trades_df['action'] == 'BUY']
        sell_trades = self.trades_df[self.trades_df['action'].str.contains('SELL')]
        
        holding_periods = []
        for ticker in buy_trades['ticker'].unique():
            ticker_buys = buy_trades[buy_trades['ticker'] == ticker]['date'].values
            ticker_sells = sell_trades[sell_trades['ticker'] == ticker]['date'].values
            
            for i, buy_date in enumerate(ticker_buys):
                if i < len(ticker_sells):
                    period = (pd.to_datetime(ticker_sells[i]) - pd.to_datetime(buy_date)).days
                    holding_periods.append(period)
        
        return np.mean(holding_periods) if holding_periods else 0
    
    def generate_html_report(self, filename=None):
        """HTML 형식의 리포트 생성 (동적 Risk-Free Rate 정보 포함)"""
        if filename is None:
            filename = f"{self.strategy_name.replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d')}.html"
        
        # Risk-free rate 정보
        rf_info = ""
        if HAS_RF_UTILS and 'avg_risk_free_rate' in self.metrics:
            rf_info = f"""
            <div class="metric-box">
                <div class="metric-value">
                    {self.metrics['avg_risk_free_rate']:.3f}%
                </div>
                <div class="metric-label">평균 Risk-Free Rate ({self.rf_ticker})</div>
            </div>
            """
        
        # Sortino ratio 추가
        sortino_info = ""
        if 'sortino_ratio' in self.metrics:
            sortino_value = self.metrics['sortino_ratio']
            if sortino_value == float('inf'):
                sortino_display = "∞"
            else:
                sortino_display = f"{sortino_value:.2f}"
            
            sortino_info = f"""
            <div class="metric-box">
                <div class="metric-value">
                    {sortino_display}
                </div>
                <div class="metric-label">소르티노 비율</div>
            </div>
            """
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.strategy_name} Performance Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .metric-box {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .rf-info {{
            background-color: #e8f5e8;
            border-left: 4px solid #27ae60;
            padding: 10px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.strategy_name}</h1>
        <h3>Performance Report</h3>
        <p>{self.metrics['start_date'].strftime('%Y-%m-%d')} to {self.metrics['end_date'].strftime('%Y-%m-%d')}</p>
    </div>
    
    <div class="rf-info">
        <strong>🏦 Risk-Free Rate:</strong> {self.rf_ticker} (미국 3개월물 국채) 사용 | 
        평균: {self.metrics.get('avg_risk_free_rate', self.default_rf_rate*100):.3f}% | 
        동적 Sharpe/Sortino Ratio 계산
    </div>
    
    <div class="section">
        <h2>핵심 성과 지표</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-value {self._get_color_class(self.metrics['total_return'])}">
                    {self.metrics['total_return']:.2f}%
                </div>
                <div class="metric-label">총 수익률</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {self._get_color_class(self.metrics['annual_return'])}">
                    {self.metrics['annual_return']:.2f}%
                </div>
                <div class="metric-label">연율화 수익률</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">
                    {self.metrics['sharpe_ratio']:.2f}
                </div>
                <div class="metric-label">샤프 비율 (동적)</div>
            </div>
            {sortino_info}
            <div class="metric-box">
                <div class="metric-value negative">
                    {self.metrics['max_drawdown']:.2f}%
                </div>
                <div class="metric-label">최대 낙폭</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">
                    {self.metrics['annual_volatility']:.2f}%
                </div>
                <div class="metric-label">연율화 변동성</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">
                    {self.metrics['win_rate']:.1f}%
                </div>
                <div class="metric-label">승률</div>
            </div>
            {rf_info}
        </div>
    </div>
    
    <div class="section">
        <h2>상세 통계</h2>
        <table>
            <tr>
                <th>항목</th>
                <th>값</th>
            </tr>
            <tr>
                <td>초기 자본</td>
                <td>{self.metrics['initial_capital']:,.0f}</td>
            </tr>
            <tr>
                <td>최종 자본</td>
                <td>{self.metrics['final_capital']:,.0f}</td>
            </tr>
            <tr>
                <td>거래일수</td>
                <td>{self.metrics['trading_days']}</td>
            </tr>
            <tr>
                <td>총 거래 횟수</td>
                <td>{self.metrics['total_trades']}</td>
            </tr>
            <tr>
                <td>평균 보유 기간</td>
                <td>{self.metrics['avg_holding_period']:.1f}일</td>
            </tr>
            <tr>
                <td>월평균 회전율</td>
                <td>{self.metrics['turnover_rate']:.2f}</td>
            </tr>
            <tr>
                <td>샤프 비율 (동적 RF)</td>
                <td>{self.metrics['sharpe_ratio']:.3f}</td>
            </tr>
            <tr>
                <td>소르티노 비율 (동적 RF)</td>
                <td>{'∞' if self.metrics.get('sortino_ratio') == float('inf') else f"{self.metrics.get('sortino_ratio', 0):.3f}"}</td>
            </tr>
            <tr>
                <td>평균 Risk-Free Rate</td>
                <td>{self.metrics.get('avg_risk_free_rate', self.default_rf_rate*100):.3f}%</td>
            </tr>
            <tr>
                <td>최고 월간 수익률</td>
                <td class="positive">{self.metrics['best_month']:.2f}%</td>
            </tr>
            <tr>
                <td>최저 월간 수익률</td>
                <td class="negative">{self.metrics['worst_month']:.2f}%</td>
            </tr>
            <tr>
                <td>수익 발생 월</td>
                <td>{self.metrics['positive_months']} / {self.metrics['total_months']}</td>
            </tr>
            <tr>
                <td>평균 수익 (일)</td>
                <td class="positive">{self.metrics['avg_win']:.3f}%</td>
            </tr>
            <tr>
                <td>평균 손실 (일)</td>
                <td class="negative">{self.metrics['avg_loss']:.3f}%</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <p style="text-align: center; color: #7f8c8d; font-size: 12px;">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Risk-Free Rate: {self.rf_ticker} (동적 계산)
        </p>
    </div>
</body>
</html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML 리포트가 생성되었습니다: {filename}")
        return filename
    
    def _get_color_class(self, value):
        """값에 따른 색상 클래스 반환"""
        return 'positive' if value >= 0 else 'negative'
    
    def save_metrics_csv(self, filename=None):
        """성과 지표를 CSV로 저장 (Risk-Free Rate 정보 포함)"""
        if filename is None:
            filename = f"{self.strategy_name.replace(' ', '_')}_metrics_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # 메트릭스에 Risk-Free Rate 정보 추가
        extended_metrics = self.metrics.copy()
        extended_metrics['rf_ticker'] = self.rf_ticker
        extended_metrics['dynamic_rf_used'] = HAS_RF_UTILS
        
        metrics_df = pd.DataFrame([extended_metrics]).T
        metrics_df.columns = ['Value']
        metrics_df.to_csv(filename)
        
        print(f"성과 지표가 CSV로 저장되었습니다: {filename}")
        return filename
    
    # 기존 메소드들은 동일하게 유지
    def generate_pdf_report(self, filename=None):
        """PDF 형식의 리포트 생성 (기존과 동일)"""
        if filename is None:
            filename = f"{self.strategy_name.replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        with PdfPages(filename) as pdf:
            # 페이지 1: 요약
            self._create_summary_page()
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # 페이지 2: 성과 차트
            self._create_performance_charts()
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # 페이지 3: 월별 분석
            self._create_monthly_analysis()
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # 페이지 4: 거래 분석
            if not self.trades_df.empty:
                self._create_trade_analysis()
                pdf.savefig(bbox_inches='tight')
                plt.close()
        
        print(f"PDF 리포트가 생성되었습니다: {filename}")
        return filename
    
    def _create_summary_page(self):
        """요약 페이지 생성 (동적 Risk-Free Rate 정보 포함)"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f'{self.strategy_name} Performance Report', fontsize=16, fontweight='bold')
        
        # Risk-Free Rate 정보
        rf_info = f"Risk-Free Rate: {self.rf_ticker} (평균: {self.metrics.get('avg_risk_free_rate', self.default_rf_rate*100):.3f}%)"
        
        # Sortino ratio 정보
        sortino_display = "∞" if self.metrics.get('sortino_ratio') == float('inf') else f"{self.metrics.get('sortino_ratio', 0):.2f}"
        
        # 텍스트로 요약 정보 표시
        summary_text = f"""
기간: {self.metrics['start_date'].strftime('%Y-%m-%d')} ~ {self.metrics['end_date'].strftime('%Y-%m-%d')}

{rf_info}

핵심 성과 지표:
• 총 수익률: {self.metrics['total_return']:.2f}%
• 연율화 수익률: {self.metrics['annual_return']:.2f}%
• 샤프 비율 (동적): {self.metrics['sharpe_ratio']:.3f}
• 소르티노 비율 (동적): {sortino_display}
• 최대 낙폭: {self.metrics['max_drawdown']:.2f}%
• 연율화 변동성: {self.metrics['annual_volatility']:.2f}%

거래 통계:
• 총 거래 횟수: {self.metrics['total_trades']}
• 평균 보유 기간: {self.metrics['avg_holding_period']:.1f}일
• 승률: {self.metrics['win_rate']:.1f}%

자본 변화:
• 초기 자본: {self.metrics['initial_capital']:,.0f}
• 최종 자본: {self.metrics['final_capital']:,.0f}

동적 Risk-Free Rate 사용: {self.rf_ticker}
"""
        
        plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center')
        plt.axis('off')
    
    # 나머지 메소드들은 기존과 동일하므로 생략...
    def _create_performance_charts(self):
        """성과 차트 생성"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Performance Analysis', fontsize=14, fontweight='bold')
        
        # 1. 포트폴리오 가치
        ax1 = axes[0, 0]
        ax1.plot(self.portfolio_df.index, self.portfolio_df['value'], linewidth=2)
        ax1.set_title('Portfolio Value')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # 2. 누적 수익률
        ax2 = axes[0, 1]
        cumulative_returns = (self.portfolio_df['value'] / self.portfolio_df['value'].iloc[0] - 1) * 100
        ax2.plot(self.portfolio_df.index, cumulative_returns, linewidth=2)
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. 낙폭
        ax3 = axes[1, 0]
        returns = self.portfolio_df['value'].pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        ax3.fill_between(drawdown.index, drawdown.values, alpha=0.5, color='red')
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 일일 수익률 분포
        ax4 = axes[1, 1]
        daily_returns = returns * 100
        ax4.hist(daily_returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax4.set_title('Daily Returns Distribution')
        ax4.set_xlabel('Return (%)')
        ax4.set_ylabel('Frequency')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
    
    def _create_monthly_analysis(self):
        """월별 분석 차트 생성"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Monthly Analysis', fontsize=14, fontweight='bold')
        
        # 월별 수익률 계산
        returns = self.portfolio_df['value'].pct_change().dropna()
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # 1. 월별 수익률 바 차트
        ax1 = axes[0]
        colors = ['green' if x > 0 else 'red' for x in monthly_returns]
        ax1.bar(monthly_returns.index, monthly_returns.values, color=colors, alpha=0.7)
        ax1.set_title('Monthly Returns')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 2. 월별 수익률 히트맵
        ax2 = axes[1]
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        if len(monthly_df['year'].unique()) > 1:
            monthly_pivot = monthly_df.pivot_table(
                index='year', columns='month', values='return'
            )
            sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                       center=0, ax=ax2, cbar_kws={'label': 'Return (%)'})
            ax2.set_title('Monthly Returns Heatmap')
        else:
            ax2.text(0.5, 0.5, 'Not enough data for heatmap', 
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Monthly Returns Heatmap')
        
        plt.tight_layout()
    
    def _create_trade_analysis(self):
        """거래 분석 차트 생성"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Trade Analysis', fontsize=14, fontweight='bold')
        
        # 1. 구성요소별 거래 빈도
        ax1 = axes[0, 0]
        trade_counts = self.trades_df['name'].value_counts().head(10)
        ax1.barh(trade_counts.index, trade_counts.values)
        ax1.set_title('Top 10 Most Traded Components')
        ax1.set_xlabel('Number of Trades')
        
        # 2. 월별 거래 횟수
        ax2 = axes[0, 1]
        self.trades_df['month'] = pd.to_datetime(self.trades_df['date']).dt.to_period('M')
        monthly_trades = self.trades_df.groupby('month').size()
        ax2.plot(monthly_trades.index.to_timestamp(), monthly_trades.values, marker='o')
        ax2.set_title('Monthly Trade Count')
        ax2.set_ylabel('Number of Trades')
        ax2.grid(True, alpha=0.3)
        
        # 3. 액션별 거래 분포
        ax3 = axes[1, 0]
        action_counts = self.trades_df['action'].value_counts()
        ax3.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
        ax3.set_title('Trade Actions Distribution')
        
        # 4. 보유 구성요소 수 추이
        ax4 = axes[1, 1]
        if 'holdings' in self.portfolio_df.columns:
            ax4.fill_between(self.portfolio_df.index, 
                           self.portfolio_df['holdings'].values, 
                           alpha=0.5)
            ax4.set_title('Number of Holdings Over Time')
            ax4.set_ylabel('Number of Components')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()


# 사용 예시
if __name__ == "__main__":
    # 테스트 데이터 생성
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    portfolio_df = pd.DataFrame({
        'value': 10000000 * (1 + np.random.randn(len(dates)).cumsum() * 0.01),
        'holdings': np.random.randint(3, 8, len(dates))
    }, index=dates)
    
    trades_df = pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', end='2024-12-31', freq='MS'),
        'ticker': ['XLK', 'XLF', 'XLV'] * 12,
        'name': ['Technology', 'Financials', 'Healthcare'] * 12,
        'action': ['BUY', 'SELL'] * 18,
        'shares': np.random.randint(100, 1000, 36),
        'price': np.random.uniform(50, 200, 36)
    })
    
    # 리포터 생성 (동적 Risk-Free Rate 사용)
    reporter = PerformanceReporter(
        strategy_name="Test Strategy with Dynamic RF",
        portfolio_df=portfolio_df,
        trades_df=trades_df,
        rf_ticker='^IRX',  # 미국 3개월물 금리
        default_rf_rate=0.02
    )
    
    # HTML 리포트 생성
    reporter.generate_html_report()
    
    # PDF 리포트 생성
    reporter.generate_pdf_report()
    
    # 성과 지표 CSV 저장
    reporter.save_metrics_csv()
    
    print("\n=== 동적 Risk-Free Rate 성과 지표 ===")
    for key, value in reporter.metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    print("\n모든 리포트가 생성되었습니다!")
