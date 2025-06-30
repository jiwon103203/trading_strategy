"""
실시간 모니터링 대시보드
웹 기반 인터랙티브 대시보드 (Streamlit 사용)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from preset_manager import PresetManager
from universal_rs_strategy import UniversalRSStrategy
from universal_jump_model import UniversalJumpModel
from universal_rs_with_jump import UniversalRSWithJumpModel

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Universal RS Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
.stMetric {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.bull-regime {
    color: #00cc00;
    font-weight: bold;
}
.bear-regime {
    color: #ff0000;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

class RealtimeDashboard:
    """실시간 모니터링 대시보드"""
    
    def __init__(self):
        # 세션 상태 초기화
        if 'selected_preset' not in st.session_state:
            st.session_state.selected_preset = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = None
        
        # 프리셋 목록
        self.presets = {
            'S&P 500 Sectors': PresetManager.get_sp500_sectors(),
            'KOSPI Sectors': PresetManager.get_kospi_sectors(),
            'MSCI Countries': PresetManager.get_msci_countries(),
            'Global Sectors': PresetManager.get_global_sectors(),
            'Emerging Markets': PresetManager.get_emerging_markets(),
            'Commodity Sectors': PresetManager.get_commodity_sectors(),
            'Factor ETFs': PresetManager.get_factor_etfs(),
            'Thematic ETFs': PresetManager.get_thematic_etfs()
        }
    
    def run(self):
        """대시보드 실행"""
        st.title("🚀 Universal RS Strategy Dashboard")
        st.markdown("### Real-time Market Monitoring & Signal Generation")
        
        # 사이드바
        self.create_sidebar()
        
        # 메인 컨텐츠
        if st.session_state.selected_preset:
            self.display_main_content()
        else:
            st.info("👈 Please select a strategy preset from the sidebar to begin")
    
    def create_sidebar(self):
        """사이드바 생성"""
        st.sidebar.header("Configuration")
        
        # 프리셋 선택
        preset_name = st.sidebar.selectbox(
            "Select Strategy Preset",
            options=list(self.presets.keys()),
            index=0 if st.session_state.selected_preset is None else None
        )
        
        if preset_name:
            st.session_state.selected_preset = self.presets[preset_name]
            st.session_state.preset_name = preset_name
        
        # 전략 파라미터
        st.sidebar.subheader("Strategy Parameters")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            rs_length = st.number_input("RS Length", value=20, min_value=5, max_value=50)
            use_jump = st.checkbox("Use Jump Model", value=True)
        
        with col2:
            timeframe = st.selectbox("Timeframe", ["daily", "weekly"])
            use_cross = st.checkbox("Use Cross Filter", value=False)
        
        if use_cross:
            cross_days = st.sidebar.number_input("Cross Days", value=30, min_value=5, max_value=90)
        else:
            cross_days = None
        
        # 백테스트 설정
        st.sidebar.subheader("Backtest Settings")
        backtest_years = st.sidebar.slider("Backtest Period (Years)", 1, 5, 3)
        
        # 실행 버튼
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Update", type="primary"):
                self.update_data(rs_length, timeframe, cross_days, use_jump)
        
        with col2:
            if st.button("📊 Backtest"):
                self.run_backtest(rs_length, timeframe, cross_days, use_jump, backtest_years)
        
        # 자동 업데이트
        auto_refresh = st.sidebar.checkbox("Auto Refresh (5 min)")
        if auto_refresh:
            time.sleep(300)  # 5분마다
            st.rerun()
        
        # 마지막 업데이트 시간
        if st.session_state.last_update:
            st.sidebar.info(f"Last Update: {st.session_state.last_update}")
    
    def display_main_content(self):
        """메인 컨텐츠 표시"""
        preset = st.session_state.selected_preset
        
        # 헤더 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Strategy", st.session_state.preset_name)
        with col2:
            st.metric("Benchmark", preset['benchmark'])
        with col3:
            st.metric("Components", len(preset['components']))
        
        # 탭 생성
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Market Status", "🎯 Current Signals", "📊 Backtest Results", 
            "📉 Performance Analysis", "📋 Trade History"
        ])
        
        with tab1:
            self.display_market_status()
        
        with tab2:
            self.display_current_signals()
        
        with tab3:
            self.display_backtest_results()
        
        with tab4:
            self.display_performance_analysis()
        
        with tab5:
            self.display_trade_history()
    
    def display_market_status(self):
        """시장 상태 표시"""
        st.subheader("Market Regime Analysis")
        
        preset = st.session_state.selected_preset
        
        # Jump Model 분석
        jump_model = UniversalJumpModel(
            benchmark_ticker=preset['benchmark'],
            benchmark_name=preset['name']
        )
        
        # 현재 체제
        current_regime = jump_model.get_current_regime()
        
        if current_regime:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                regime_html = f"<h3 class='{'bull-regime' if current_regime['regime'] == 'BULL' else 'bear-regime'}'>{current_regime['regime']}</h3>"
                st.markdown(regime_html, unsafe_allow_html=True)
                st.caption("Current Regime")
            
            with col2:
                st.metric("Confidence", f"{current_regime['confidence']:.1%}")
            
            with col3:
                features = current_regime['features']
                st.metric("Volatility", f"{features['realized_vol']*100:.1f}%")
            
            with col4:
                st.metric("Drawdown", f"{features['max_drawdown']*100:.1f}%")
            
            # 체제 이력 차트
            st.subheader("Regime History (90 days)")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            regime_history = jump_model.get_regime_history(start_date, end_date)
            
            if regime_history is not None:
                # Plotly 차트
                fig = go.Figure()
                
                # Bull/Bear 영역
                for i in range(len(regime_history) - 1):
                    color = 'rgba(0, 255, 0, 0.3)' if regime_history.iloc[i]['regime'] == 'BULL' else 'rgba(255, 0, 0, 0.3)'
                    fig.add_vrect(
                        x0=regime_history.index[i],
                        x1=regime_history.index[i + 1],
                        fillcolor=color,
                        layer="below",
                        line_width=0
                    )
                
                # 체제 라인
                regime_line = regime_history['regime'].map({'BULL': 1, 'BEAR': 0})
                fig.add_trace(go.Scatter(
                    x=regime_history.index,
                    y=regime_line,
                    mode='lines',
                    name='Regime',
                    line=dict(color='black', width=2)
                ))
                
                fig.update_layout(
                    title="Market Regime Changes",
                    xaxis_title="Date",
                    yaxis_title="Regime",
                    yaxis=dict(
                        tickmode='array',
                        tickvals=[0, 1],
                        ticktext=['BEAR', 'BULL']
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Unable to analyze market regime")
    
    def display_current_signals(self):
        """현재 투자 신호 표시"""
        st.subheader("Current Investment Signals")
        
        preset = st.session_state.selected_preset
        
        # RS 전략 분석
        strategy = UniversalRSStrategy(
            benchmark=preset['benchmark'],
            components=preset['components'],
            name=preset['name']
        )
        
        # 데이터 가져오기
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        with st.spinner('Analyzing components...'):
            price_data, benchmark_data = strategy.get_price_data(start_date, end_date)
        
        if price_data and benchmark_data:
            selected = strategy.select_components(price_data, benchmark_data, end_date)
            
            if selected:
                # 데이터프레임 생성
                signals_df = pd.DataFrame(selected)
                signals_df['RS_Score'] = (signals_df['rs_ratio'] + signals_df['rs_momentum']) / 2
                signals_df = signals_df.sort_values('RS_Score', ascending=False)
                
                # 신호 강도별 색상
                def get_signal_color(score):
                    if score > 105:
                        return 'background-color: #90EE90'  # 연한 초록
                    elif score > 102:
                        return 'background-color: #FFFFE0'  # 연한 노랑
                    else:
                        return 'background-color: #FFE4E1'  # 연한 빨강
                
                # 테이블 표시
                st.dataframe(
                    signals_df[['name', 'rs_ratio', 'rs_momentum', 'RS_Score']].style.applymap(
                        lambda x: get_signal_color(x) if isinstance(x, (int, float)) and x > 100 else '',
                        subset=['RS_Score']
                    ),
                    use_container_width=True
                )
                
                # 차트
                col1, col2 = st.columns(2)
                
                with col1:
                    # RS Ratio 바 차트
                    fig_ratio = px.bar(
                        signals_df.head(10),
                        x='name',
                        y='rs_ratio',
                        title='Top 10 Components by RS-Ratio',
                        color='rs_ratio',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_ratio.add_hline(y=100, line_dash="dash", line_color="black")
                    st.plotly_chart(fig_ratio, use_container_width=True)
                
                with col2:
                    # RS Momentum 바 차트
                    fig_momentum = px.bar(
                        signals_df.head(10),
                        x='name',
                        y='rs_momentum',
                        title='Top 10 Components by RS-Momentum',
                        color='rs_momentum',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_momentum.add_hline(y=100, line_dash="dash", line_color="black")
                    st.plotly_chart(fig_momentum, use_container_width=True)
                
                # 투자 권고
                st.info(f"💡 **Investment Recommendation**: {len(selected)} components meet the criteria for investment")
            else:
                st.warning("No components currently meet the investment criteria")
        else:
            st.error("Unable to fetch market data")
    
    def update_data(self, rs_length, timeframe, cross_days, use_jump):
        """데이터 업데이트"""
        st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success("Data updated successfully!")
    
    def run_backtest(self, rs_length, timeframe, cross_days, use_jump, years):
        """백테스트 실행"""
        preset = st.session_state.selected_preset
        
        with st.spinner('Running backtest...'):
            strategy = UniversalRSWithJumpModel(
                preset_config=preset,
                rs_length=rs_length,
                rs_timeframe=timeframe,
                rs_recent_cross_days=cross_days,
                use_jump_model=use_jump
            )
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*years)
            
            portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)
            
            if portfolio_df is not None:
                st.session_state.portfolio_data = {
                    'portfolio': portfolio_df,
                    'trades': trades_df,
                    'regime': regime_df,
                    'metrics': strategy.calculate_performance_metrics(portfolio_df)
                }
                st.success("Backtest completed!")
    
    def display_backtest_results(self):
        """백테스트 결과 표시"""
        st.subheader("Backtest Results")
        
        if st.session_state.portfolio_data is None:
            st.info("Run a backtest to see results")
            return
        
        data = st.session_state.portfolio_data
        metrics = data['metrics']
        
        # 핵심 지표
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", metrics.get('총 수익률', 'N/A'))
        with col2:
            st.metric("Annual Return", metrics.get('연율화 수익률', 'N/A'))
        with col3:
            st.metric("Sharpe Ratio", metrics.get('샤프 비율', 'N/A'))
        with col4:
            st.metric("Max Drawdown", metrics.get('최대 낙폭', 'N/A'))
        
        # 포트폴리오 가치 차트
        portfolio_df = data['portfolio']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_performance_analysis(self):
        """성과 분석 표시"""
        st.subheader("Performance Analysis")
        
        if st.session_state.portfolio_data is None:
            st.info("Run a backtest to see analysis")
            return
        
        portfolio_df = st.session_state.portfolio_data['portfolio']
        
        # 수익률 계산
        returns = portfolio_df['value'].pct_change().dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 월별 수익률
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            
            fig_monthly = go.Figure()
            colors = ['green' if x > 0 else 'red' for x in monthly_returns]
            fig_monthly.add_trace(go.Bar(
                x=monthly_returns.index,
                y=monthly_returns.values,
                marker_color=colors
            ))
            
            fig_monthly.update_layout(
                title="Monthly Returns",
                xaxis_title="Month",
                yaxis_title="Return (%)",
                height=400
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            # 수익률 분포
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name='Daily Returns'
            ))
            
            fig_dist.update_layout(
                title="Return Distribution",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # 낙폭 차트
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=1)
        ))
        
        fig_dd.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)
    
    def display_trade_history(self):
        """거래 내역 표시"""
        st.subheader("Trade History")
        
        if st.session_state.portfolio_data is None or st.session_state.portfolio_data['trades'].empty:
            st.info("No trades to display")
            return
        
        trades_df = st.session_state.portfolio_data['trades']
        
        # 거래 요약
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", len(trades_df))
        with col2:
            st.metric("Unique Components", trades_df['ticker'].nunique())
        with col3:
            avg_trades_per_month = len(trades_df) / ((trades_df['date'].max() - trades_df['date'].min()).days / 30)
            st.metric("Avg Trades/Month", f"{avg_trades_per_month:.1f}")
        
        # 거래 테이블
        st.dataframe(
            trades_df[['date', 'ticker', 'name', 'action', 'shares', 'price']],
            use_container_width=True
        )
        
        # 구성요소별 거래 빈도
        trade_counts = trades_df['name'].value_counts().head(10)
        
        fig_trades = px.bar(
            x=trade_counts.values,
            y=trade_counts.index,
            orientation='h',
            title='Top 10 Most Traded Components',
            labels={'x': 'Number of Trades', 'y': 'Component'}
        )
        
        st.plotly_chart(fig_trades, use_container_width=True)


# Streamlit 앱 실행
def main():
    dashboard = RealtimeDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
