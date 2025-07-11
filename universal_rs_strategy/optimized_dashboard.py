"""
최적화된 실시간 대시보드
Streamlit 기반 웹 인터페이스
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

# 전략 모듈
from preset_manager import PresetManager
from optimized_integrated_strategy import OptimizedIntegratedStrategy
from optimized_jump_model import analyze_multiple_regimes
from optimized_rs_strategy import quick_rs_analysis

# 유틸리티
from config_manager import config_manager
from data_cache import cache_stats
from memory_optimization import memory_monitor

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
.metric-card {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.bull-indicator {
    color: #00cc00;
    font-weight: bold;
}
.bear-indicator {
    color: #ff0000;
    font-weight: bold;
}
.strategy-card {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    transition: all 0.3s ease;
}
.strategy-card:hover {
    border-color: #1f77b4;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


class OptimizedDashboard:
    """최적화된 실시간 대시보드"""
    
    def __init__(self):
        self._init_session_state()
        self.presets = self._load_presets()
    
    def _init_session_state(self):
        """세션 상태 초기화"""
        defaults = {
            'selected_preset': None,
            'last_update': None,
            'portfolio_data': None,
            'cache_data': {},
            'refresh_interval': 300,  # 5분
            'auto_refresh': False,
            'dark_mode': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _load_presets(self):
        """프리셋 로드"""
        return {
            'S&P 500 Sectors': PresetManager.get_sp500_sectors(),
            'KOSPI 200 Sectors': PresetManager.get_kospi_sectors(),
            'KOSPI Full Market': PresetManager.get_kospi_full_sectors(),
            'KOSDAQ Sectors': PresetManager.get_kosdaq_sectors(),
            'Korea Comprehensive': PresetManager.get_korea_comprehensive(),
            'MSCI Countries': PresetManager.get_msci_countries(),
            'Europe Sectors': PresetManager.get_europe_sectors(),
            'Global Sectors': PresetManager.get_global_sectors(),
            'Emerging Markets': PresetManager.get_emerging_markets(),
            'Commodity Sectors': PresetManager.get_commodity_sectors(),
            'Factor ETFs': PresetManager.get_factor_etfs(),
            'Thematic ETFs': PresetManager.get_thematic_etfs()
        }
    
    def run(self):
        """대시보드 실행"""
        st.title("📈 Universal RS Strategy Dashboard")
        st.markdown("### Optimized Real-time Market Monitoring & Analysis")
        
        # 사이드바
        self.create_sidebar()
        
        # 메인 컨텐츠
        if st.session_state.selected_preset:
            self.display_main_content()
        else:
            self.display_welcome()
    
    def create_sidebar(self):
        """사이드바 생성"""
        st.sidebar.header("⚙️ Configuration")
        
        # 프리셋 선택
        preset_name = st.sidebar.selectbox(
            "Select Strategy Preset",
            options=list(self.presets.keys()),
            index=0 if st.session_state.selected_preset is None else None
        )
        
        if preset_name:
            st.session_state.selected_preset = self.presets[preset_name]
            st.session_state.preset_name = preset_name
        
        # 설정
        st.sidebar.subheader("📊 Settings")
        
        # Jump Model 설정
        use_jump = st.sidebar.checkbox("Use Jump Model", value=True)
        
        # 리프레시 설정
        st.sidebar.subheader("🔄 Auto Refresh")
        auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=60,
                max_value=600,
                value=st.session_state.refresh_interval,
                step=30
            )
            st.session_state.refresh_interval = refresh_interval
        
        # 액션 버튼
        st.sidebar.subheader("🎯 Actions")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                self._refresh_data()
        
        with col2:
            if st.button("📊 Backtest", use_container_width=True):
                st.session_state.show_backtest = True
        
        # 캐시 정보
        st.sidebar.subheader("💾 Cache Status")
        cache_info = self._get_cache_info()
        st.sidebar.metric("Hit Rate", f"{cache_info['hit_rate']}")
        st.sidebar.metric("Cache Size", f"{cache_info['size_mb']:.1f} MB")
        
        # 메모리 정보
        st.sidebar.subheader("🧠 Memory Usage")
        memory_info = memory_monitor.get_memory_info()
        st.sidebar.metric("Current", f"{memory_info['rss_mb']:.1f} MB")
        st.sidebar.progress(memory_info['percent'] / 100)
    
    def display_welcome(self):
        """환영 화면"""
        st.markdown("""
        <div class="metric-card">
            <h2>Welcome to Universal RS Strategy Dashboard</h2>
            <p>Select a strategy preset from the sidebar to begin monitoring.</p>
            
            <h3>Features:</h3>
            <ul>
                <li>🎯 Real-time market regime detection</li>
                <li>📊 Component selection based on RS momentum</li>
                <li>📈 Performance backtesting</li>
                <li>🔄 Auto-refresh capability</li>
                <li>💾 Intelligent caching system</li>
                <li>🚀 Optimized performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def display_main_content(self):
        """메인 컨텐츠 표시"""
        preset = st.session_state.selected_preset
        
        # 헤더 정보
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Strategy", st.session_state.preset_name)
        
        with col2:
            st.metric("Benchmark", preset['benchmark'])
        
        with col3:
            st.metric("Components", len(preset['components']))
        
        with col4:
            if st.session_state.last_update:
                st.metric("Last Update", st.session_state.last_update)
            else:
                st.metric("Last Update", "Never")
        
        # 탭 생성
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Market Status", 
            "🎯 Current Signals", 
            "🌍 Global Regimes", 
            "📊 Backtesting",
            "📉 Performance Analytics"
        ])
        
        with tab1:
            self._display_market_status()
        
        with tab2:
            self._display_current_signals()
        
        with tab3:
            self._display_global_regimes()
        
        with tab4:
            self._display_backtesting()
        
        with tab5:
            self._display_analytics()
    
    def _display_market_status(self):
        """시장 상태 표시"""
        st.subheader("Market Regime Analysis")
        
        preset = st.session_state.selected_preset
        
        # 현재 체제 분석
        with st.spinner("Analyzing market regime..."):
            try:
                strategy = OptimizedIntegratedStrategy(
                    preset_config=preset,
                    use_jump_model=True
                )
                
                status = strategy.get_current_status()
                
                # 체제 정보 카드
                regime_color = "bull-indicator" if status['regime'] == 'BULL' else "bear-indicator"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Market Regime</h3>
                    <p class="{regime_color}" style="font-size: 24px;">
                        {status['regime']} Market
                    </p>
                    <p>Confidence: {status['regime_confidence']:.1%}</p>
                    <p>Analysis Date: {status['date'].strftime('%Y-%m-%d')}</p>
                    <p>Status: {'🔮 Out-of-Sample' if status.get('is_out_of_sample', False) else '📚 In-Sample'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 체제 이력 차트
                if 'regime_history' in st.session_state.cache_data:
                    self._plot_regime_history(st.session_state.cache_data['regime_history'])
                
            except Exception as e:
                st.error(f"Failed to analyze market regime: {str(e)}")
    
    def _display_current_signals(self):
        """현재 신호 표시"""
        st.subheader("Current Investment Signals")
        
        preset = st.session_state.selected_preset
        
        with st.spinner("Analyzing components..."):
            try:
                # RS 분석
                rankings = quick_rs_analysis(preset)
                
                if not rankings.empty:
                    # 상위 10개 표시
                    st.dataframe(
                        rankings.head(10).style.highlight_max(subset=['rs_ratio', 'rs_momentum', 'rs_score']),
                        use_container_width=True
                    )
                    
                    # 차트
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_ratio = px.bar(
                            rankings.head(10),
                            x='ticker',
                            y='rs_ratio',
                            title='Top 10 by RS-Ratio',
                            color='rs_ratio',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_ratio, use_container_width=True)
                    
                    with col2:
                        fig_momentum = px.bar(
                            rankings.head(10),
                            x='ticker',
                            y='rs_momentum',
                            title='Top 10 by RS-Momentum',
                            color='rs_momentum',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_momentum, use_container_width=True)
                    
                else:
                    st.info("No components currently meet the investment criteria.")
                    
            except Exception as e:
                st.error(f"Failed to analyze signals: {str(e)}")
    
    def _display_global_regimes(self):
        """글로벌 시장 체제 표시"""
        st.subheader("Global Market Regimes")
        
        # 분석할 시장
        markets = [
            '^GSPC', '^DJI', 'QQQ',  # US
            '069500.KS', '^KS11',     # Korea
            '^N225', '^HSI',          # Asia
            '^FTSE', '^GDAXI',       # Europe
            'EEM', 'GLD', 'TLT'      # Others
        ]
        
        with st.spinner("Analyzing global markets..."):
            try:
                # 체제 분석
                results = analyze_multiple_regimes(markets)
                
                if not results.empty:
                    # 체제별 카운트
                    bull_count = (results['regime'] == 'BULL').sum()
                    bear_count = (results['regime'] == 'BEAR').sum()
                    
                    # 메트릭
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Markets", len(results))
                    with col2:
                        st.metric("🟢 BULL", bull_count)
                    with col3:
                        st.metric("🔴 BEAR", bear_count)
                    
                    # 차트
                    fig = go.Figure(data=[go.Pie(
                        labels=['BULL', 'BEAR'],
                        values=[bull_count, bear_count],
                        marker_colors=['#00cc00', '#ff0000']
                    )])
                    fig.update_layout(title="Global Market Regime Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 상세 테이블
                    results['regime_emoji'] = results['regime'].apply(
                        lambda x: '🟢' if x == 'BULL' else '🔴'
                    )
                    st.dataframe(
                        results[['ticker', 'regime_emoji', 'regime', 'confidence']],
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Failed to analyze global regimes: {str(e)}")
    
    def _display_backtesting(self):
        """백테스팅 인터페이스"""
        st.subheader("Backtesting")
        
        # 백테스트 설정
        col1, col2, col3 = st.columns(3)
        
        with col1:
            years = st.selectbox("Period (Years)", [1, 2, 3, 5], index=1)
        
        with col2:
            use_jump = st.checkbox("Use Jump Model", value=True)
        
        with col3:
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=1000000,
                max_value=100000000,
                value=10000000,
                step=1000000
            )
        
        if st.button("Run Backtest", type="primary"):
            self._run_backtest(years, use_jump, initial_capital)
        
        # 백테스트 결과
        if st.session_state.portfolio_data:
            self._display_backtest_results()
    
    def _display_analytics(self):
        """성과 분석"""
        st.subheader("Performance Analytics")
        
        if not st.session_state.portfolio_data:
            st.info("Run a backtest to see performance analytics.")
            return
        
        data = st.session_state.portfolio_data
        portfolio_df = data['portfolio']
        metrics = data['metrics']
        
        # 주요 메트릭
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", metrics.get('총 수익률', 'N/A'))
        
        with col2:
            st.metric("Annual Return", metrics.get('연율화 수익률', 'N/A'))
        
        with col3:
            st.metric("Sharpe Ratio", metrics.get('샤프 비율', 'N/A'))
        
        with col4:
            st.metric("Max Drawdown", metrics.get('최대 낙폭', 'N/A'))
        
        # 상세 차트
        # 1. 누적 수익률
        returns = portfolio_df['value'].pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod() - 1
        
        fig_returns = go.Figure()
        fig_returns.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values * 100,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='blue', width=2)
        ))
        fig_returns.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            height=400
        )
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # 2. 월별 수익률 히트맵
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        if len(monthly_df['year'].unique()) > 1:
            monthly_pivot = monthly_df.pivot_table(
                index='month',
                columns='year',
                values='return'
            )
            
            fig_heatmap = px.imshow(
                monthly_pivot,
                labels=dict(x="Year", y="Month", color="Return (%)"),
                title="Monthly Returns Heatmap",
                color_continuous_scale='RdYlGn',
                aspect='auto'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def _refresh_data(self):
        """데이터 새로고침"""
        st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        st.success("Data refreshed successfully!")
    
    def _get_cache_info(self):
        """캐시 정보 가져오기"""
        stats, size_info = cache_stats()
        return {
            'hit_rate': stats.get('hit_rate', '0%'),
            'size_mb': size_info.get('total_size_mb', 0)
        }
    
    def _run_backtest(self, years, use_jump, initial_capital):
        """백테스트 실행"""
        preset = st.session_state.selected_preset
        
        with st.spinner("Running backtest..."):
            try:
                # 전략 생성
                strategy = OptimizedIntegratedStrategy(
                    preset_config=preset,
                    use_jump_model=use_jump
                )
                
                # 백테스트 실행
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365*years)
                
                portfolio_df, trades_df, metrics = strategy.backtest(
                    start_date, end_date, initial_capital
                )
                
                # 결과 저장
                st.session_state.portfolio_data = {
                    'portfolio': portfolio_df,
                    'trades': trades_df,
                    'metrics': metrics,
                    'use_jump': use_jump,
                    'period': years
                }
                
                st.success("Backtest completed successfully!")
                
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
    
    def _display_backtest_results(self):
        """백테스트 결과 표시"""
        data = st.session_state.portfolio_data
        portfolio_df = data['portfolio']
        trades_df = data['trades']
        
        # 포트폴리오 차트
        fig = go.Figure()
        
        # 포트폴리오 가치
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        # 체제별 색상 (Jump Model 사용시)
        if 'regime' in portfolio_df.columns and data['use_jump']:
            bull_df = portfolio_df[portfolio_df['regime'] == 'BULL']
            bear_df = portfolio_df[portfolio_df['regime'] == 'BEAR']
            
            if not bull_df.empty:
                fig.add_trace(go.Scatter(
                    x=bull_df.index,
                    y=bull_df['value'],
                    mode='markers',
                    name='BULL',
                    marker=dict(color='green', size=2),
                    showlegend=True
                ))
            
            if not bear_df.empty:
                fig.add_trace(go.Scatter(
                    x=bear_df.index,
                    y=bear_df['value'],
                    mode='markers',
                    name='BEAR',
                    marker=dict(color='red', size=2),
                    showlegend=True
                ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 거래 통계
        if not trades_df.empty:
            st.subheader("Trade Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", len(trades_df))
            
            with col2:
                buy_trades = len(trades_df[trades_df['action'] == 'BUY'])
                st.metric("Buy Trades", buy_trades)
            
            with col3:
                sell_trades = len(trades_df[trades_df['action'].str.contains('SELL')])
                st.metric("Sell Trades", sell_trades)
    
    def _plot_regime_history(self, regime_history):
        """체제 이력 차트"""
        if regime_history is None or regime_history.empty:
            return
        
        fig = go.Figure()
        
        # 체제를 숫자로 변환
        regime_numeric = regime_history['regime'].map({'BULL': 1, 'BEAR': 0})
        
        fig.add_trace(go.Scatter(
            x=regime_history.index,
            y=regime_numeric,
            mode='lines',
            name='Market Regime',
            line=dict(color='black', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.3)'
        ))
        
        fig.update_layout(
            title="Market Regime History",
            xaxis_title="Date",
            yaxis_title="Regime",
            yaxis=dict(
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['BEAR', 'BULL']
            ),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)


def main():
    """메인 함수"""
    dashboard = OptimizedDashboard()
    
    # 자동 새로고침
    if st.session_state.auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()
    
    dashboard.run()


if __name__ == "__main__":
    main()
