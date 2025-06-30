"""
실시간 모니터링 대시보드 - 전체 ETF 버전
웹 기반 인터랙티브 대시보드 (Streamlit 사용)
전체 ETF 지원 + 종합 Bull/Bear 상태 모니터링
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
import concurrent.futures
from threading import Lock

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Universal RS Strategy Dashboard - Full Edition",
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
.regime-card {
    background-color: #ffffff;
    border: 2px solid #ddd;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.bull-card {
    border-color: #28a745;
    background-color: #f8fff9;
}
.bear-card {
    border-color: #dc3545;
    background-color: #fff8f8;
}
.strategy-header {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
}
.etf-item {
    padding: 8px;
    margin: 5px 0;
    border-radius: 5px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.etf-bull {
    background-color: #d4edda;
    border-left: 4px solid #28a745;
}
.etf-bear {
    background-color: #f8d7da;
    border-left: 4px solid #dc3545;
}
.etf-unknown {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
}
</style>
""", unsafe_allow_html=True)

def safe_data_check(data):
    """완전히 안전한 데이터 검증 함수"""
    try:
        if data is None:
            return False
        
        if isinstance(data, dict):
            return len(data) > 0
        
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return len(data) > 0
        
        if isinstance(data, list):
            return len(data) > 0
            
        return data is not None
        
    except Exception:
        return False

def safe_get_value(value, default=0):
    """안전한 값 추출"""
    try:
        if pd.isna(value):
            return default
        
        if isinstance(value, (pd.Series, pd.DataFrame)):
            if len(value) > 0:
                val = value.iloc[-1] if hasattr(value, 'iloc') else value
                return float(val) if not pd.isna(val) else default
            else:
                return default
        
        return float(value) if not pd.isna(value) else default
        
    except Exception:
        return default

class EnhancedRealtimeDashboard:
    """실시간 모니터링 대시보드 - 전체 ETF 버전"""
    
    def __init__(self):
        # 세션 상태 초기화
        if 'selected_preset' not in st.session_state:
            st.session_state.selected_preset = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = None
        if 'regime_cache' not in st.session_state:
            st.session_state.regime_cache = {}
        if 'cache_timestamp' not in st.session_state:
            st.session_state.cache_timestamp = None
        
        # 전체 프리셋 목록 (제한 없음)
        self.presets = {
            'S&P 500 Sectors': PresetManager.get_sp500_sectors(),
            'KOSPI Sectors': PresetManager.get_kospi_sectors(),
            'MSCI Countries': PresetManager.get_msci_countries(),
            'Europe Sectors': PresetManager.get_europe_sectors(),
            'Global Sectors': PresetManager.get_global_sectors(),
            'Emerging Markets': PresetManager.get_emerging_markets(),
            'Commodity Sectors': PresetManager.get_commodity_sectors(),
            'Factor ETFs': PresetManager.get_factor_etfs(),
            'Thematic ETFs': PresetManager.get_thematic_etfs()
        }
        
        # 캐시 유효 시간 (30분)
        self.cache_duration = timedelta(minutes=30)
    
    def run(self):
        """대시보드 실행"""
        st.title("🚀 Universal RS Strategy Dashboard - Full Edition")
        st.markdown("### Real-time Market Monitoring & Signal Generation (All ETFs)")
        
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
            rs_length = st.number_input("RS Length", value=20, min_value=10, max_value=50)
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
        
        # 추가 기능
        st.sidebar.subheader("Advanced Features")
        if st.sidebar.button("🌍 Refresh All Regimes"):
            self.refresh_all_regimes()
        
        if st.sidebar.button("💾 Download Results"):
            self.download_results()
        
        if st.sidebar.button("🔄 Clear Cache"):
            self.clear_cache()
        
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
        
        # 탭 생성 (전체 regime 모니터링 탭 추가)
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Market Status", "🎯 Current Signals", "🌍 All Market Regimes", "📊 Backtest Results"
        ])
        
        with tab1:
            self.display_market_status()
        
        with tab2:
            self.display_current_signals()
        
        with tab3:
            self.display_all_market_regimes()
        
        with tab4:
            self.display_backtest_results()
    
    def display_market_status(self):
        """시장 상태 표시"""
        st.subheader("Market Regime Analysis")
        
        preset = st.session_state.selected_preset
        
        if st.button("🔍 Analyze Market Regime"):
            with st.spinner("Analyzing market regime..."):
                try:
                    jump_model = UniversalJumpModel(
                        benchmark_ticker=preset['benchmark'],
                        benchmark_name=preset['name']
                    )
                    
                    current_regime = jump_model.get_current_regime()
                    
                    if current_regime:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            regime_emoji = "🟢" if current_regime['regime'] == 'BULL' else "🔴"
                            st.metric("Current Regime", f"{regime_emoji} {current_regime['regime']}")
                        
                        with col2:
                            confidence = safe_get_value(current_regime['confidence'], 0.5)
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        with col3:
                            features = current_regime.get('features', {})
                            vol = safe_get_value(features.get('realized_vol', 0), 0) * 100
                            st.metric("Volatility", f"{vol:.1f}%")
                        
                        with col4:
                            dd = safe_get_value(features.get('max_drawdown', 0), 0) * 100
                            st.metric("Drawdown", f"{dd:.1f}%")
                        
                        st.success("✅ Market regime analysis completed!")
                    else:
                        st.error("❌ Unable to analyze market regime")
                        
                except Exception as e:
                    st.error(f"Market regime analysis failed: {str(e)}")
                    st.info("💡 Check your internet connection or try again later")
    
    def display_current_signals(self):
        """현재 투자 신호 표시"""
        st.subheader("Current Investment Signals")
        
        preset = st.session_state.selected_preset
        
        if st.button("🎯 Analyze Investment Signals"):
            with st.spinner('Analyzing components...'):
                try:
                    # RS 전략 분석
                    strategy = UniversalRSStrategy(
                        benchmark=preset['benchmark'],
                        components=preset['components'],
                        name=preset['name']
                    )
                    
                    # 데이터 가져오기
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=120)
                    
                    price_data, benchmark_data = strategy.get_price_data(start_date, end_date)
                    
                    # 안전한 데이터 검증
                    price_data_ok = safe_data_check(price_data)
                    benchmark_data_ok = safe_data_check(benchmark_data)
                    
                    if price_data_ok and benchmark_data_ok:
                        selected = strategy.select_components(price_data, benchmark_data, end_date)
                        
                        if safe_data_check(selected):
                            # 데이터프레임 생성
                            signals_df = pd.DataFrame(selected)
                            signals_df['RS_Score'] = (signals_df['rs_ratio'] + signals_df['rs_momentum']) / 2
                            signals_df = signals_df.sort_values('RS_Score', ascending=False)
                            
                            # 테이블 표시
                            st.dataframe(
                                signals_df[['name', 'rs_ratio', 'rs_momentum', 'RS_Score']],
                                use_container_width=True
                            )
                            
                            # 차트
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # RS Ratio 바 차트
                                fig_ratio = px.bar(
                                    signals_df.head(15),
                                    x='name',
                                    y='rs_ratio',
                                    title='Top Components by RS-Ratio',
                                    color='rs_ratio',
                                    color_continuous_scale='RdYlGn'
                                )
                                fig_ratio.add_hline(y=100, line_dash="dash", line_color="black")
                                fig_ratio.update_xaxes(tickangle=45)
                                st.plotly_chart(fig_ratio, use_container_width=True)
                            
                            with col2:
                                # RS Momentum 바 차트
                                fig_momentum = px.bar(
                                    signals_df.head(15),
                                    x='name',
                                    y='rs_momentum',
                                    title='Top Components by RS-Momentum',
                                    color='rs_momentum',
                                    color_continuous_scale='RdYlGn'
                                )
                                fig_momentum.add_hline(y=100, line_dash="dash", line_color="black")
                                fig_momentum.update_xaxes(tickangle=45)
                                st.plotly_chart(fig_momentum, use_container_width=True)
                            
                            # 투자 권고
                            st.success(f"✅ **Investment Recommendation**: {len(selected)} components meet the criteria for investment")
                        else:
                            st.warning("⚠️ No components currently meet the investment criteria")
                    else:
                        st.error("❌ Unable to fetch market data")
                        if not price_data_ok:
                            st.error("   • Price data unavailable")
                        if not benchmark_data_ok:
                            st.error("   • Benchmark data unavailable")
                            
                except Exception as e:
                    st.error(f"Signal analysis failed: {str(e)}")
                    st.info("💡 Check your internet connection or try a simpler analysis")
    
    def analyze_single_etf_regime(self, ticker, name):
        """단일 ETF의 시장 체제 분석"""
        try:
            jump_model = UniversalJumpModel(
                benchmark_ticker=ticker,
                benchmark_name=name,
                jump_penalty=50.0
            )
            
            current_regime = jump_model.get_current_regime()
            
            if current_regime:
                return {
                    'ticker': ticker,
                    'name': name,
                    'regime': current_regime['regime'],
                    'confidence': current_regime['confidence'],
                    'status': 'success'
                }
            else:
                return {
                    'ticker': ticker,
                    'name': name,
                    'regime': 'UNKNOWN',
                    'confidence': 0.0,
                    'status': 'no_data'
                }
        except Exception as e:
            return {
                'ticker': ticker,
                'name': name,
                'regime': 'ERROR',
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_all_etf_regimes(self):
        """모든 ETF의 시장 체제 병렬 분석"""
        # 모든 전략의 모든 ETF 수집
        all_etfs = {}
        for strategy_name, preset in self.presets.items():
            for ticker, name in preset['components'].items():
                if ticker not in all_etfs:
                    all_etfs[ticker] = {
                        'name': name,
                        'strategies': [strategy_name]
                    }
                else:
                    all_etfs[ticker]['strategies'].append(strategy_name)
        
        # 벤치마크도 추가
        benchmarks = {}
        for strategy_name, preset in self.presets.items():
            benchmark = preset['benchmark']
            if benchmark not in benchmarks:
                benchmarks[benchmark] = f"{strategy_name} Benchmark"
        
        # 캐시 확인
        now = datetime.now()
        if (st.session_state.cache_timestamp and 
            now - st.session_state.cache_timestamp < self.cache_duration and
            st.session_state.regime_cache):
            return st.session_state.regime_cache
        
        results = {}
        
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_items = len(all_etfs) + len(benchmarks)
        processed = 0
        
        # 벤치마크 분석
        status_text.text("Analyzing benchmarks...")
        for ticker, name in benchmarks.items():
            result = self.analyze_single_etf_regime(ticker, name)
            results[ticker] = result
            results[ticker]['type'] = 'benchmark'
            
            processed += 1
            progress_bar.progress(processed / total_items)
        
        # ETF 분석 (배치로 처리)
        status_text.text("Analyzing ETFs...")
        batch_size = 5  # 동시에 처리할 ETF 수
        etf_items = list(all_etfs.items())
        
        for i in range(0, len(etf_items), batch_size):
            batch = etf_items[i:i+batch_size]
            
            # 병렬 처리
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = []
                for ticker, info in batch:
                    future = executor.submit(self.analyze_single_etf_regime, ticker, info['name'])
                    futures.append((ticker, info, future))
                
                for ticker, info, future in futures:
                    try:
                        result = future.result(timeout=30)  # 30초 타임아웃
                        result['type'] = 'etf'
                        result['strategies'] = info['strategies']
                        results[ticker] = result
                    except Exception as e:
                        results[ticker] = {
                            'ticker': ticker,
                            'name': info['name'],
                            'regime': 'TIMEOUT',
                            'confidence': 0.0,
                            'status': 'timeout',
                            'type': 'etf',
                            'strategies': info['strategies']
                        }
                    
                    processed += 1
                    progress_bar.progress(processed / total_items)
        
        # 캐시 업데이트
        st.session_state.regime_cache = results
        st.session_state.cache_timestamp = now
        
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def display_all_market_regimes(self):
        """모든 시장 체제 현황 표시"""
        st.subheader("🌍 All Market Regimes Overview")
        st.markdown("Current Bull/Bear status for all ETFs across all strategies")
        
        if st.button("🔄 Analyze All Market Regimes", type="primary"):
            with st.spinner("Analyzing all market regimes... This may take a few minutes"):
                results = self.analyze_all_etf_regimes()
                
                if results:
                    # 통계 요약
                    bull_count = sum(1 for r in results.values() if r['regime'] == 'BULL')
                    bear_count = sum(1 for r in results.values() if r['regime'] == 'BEAR')
                    unknown_count = sum(1 for r in results.values() if r['regime'] in ['UNKNOWN', 'ERROR', 'TIMEOUT'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Assets", len(results))
                    with col2:
                        st.metric("🟢 BULL", bull_count)
                    with col3:
                        st.metric("🔴 BEAR", bear_count)
                    with col4:
                        st.metric("⚠️ Unknown", unknown_count)
                    
                    # 전략별 정리
                    st.subheader("📊 By Strategy")
                    
                    for strategy_name, preset in self.presets.items():
                        with st.expander(f"{strategy_name} ({len(preset['components'])} ETFs)"):
                            
                            # 벤치마크 상태
                            benchmark_ticker = preset['benchmark']
                            if benchmark_ticker in results:
                                benchmark_result = results[benchmark_ticker]
                                regime_class = f"{benchmark_result['regime'].lower()}-card" if benchmark_result['regime'] in ['BULL', 'BEAR'] else "unknown-card"
                                
                                st.markdown(f"""
                                <div class="regime-card {regime_class}">
                                    <div class="strategy-header">📊 Benchmark: {benchmark_result['name']}</div>
                                    <div><strong>Regime:</strong> {benchmark_result['regime']} 
                                    {'(Confidence: ' + f"{benchmark_result['confidence']:.1%}" + ')' if benchmark_result['confidence'] > 0 else ''}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # ETF 목록
                            st.markdown("**Components:**")
                            
                            bull_etfs = []
                            bear_etfs = []
                            unknown_etfs = []
                            
                            for ticker, name in preset['components'].items():
                                if ticker in results:
                                    result = results[ticker]
                                    if result['regime'] == 'BULL':
                                        bull_etfs.append(result)
                                    elif result['regime'] == 'BEAR':
                                        bear_etfs.append(result)
                                    else:
                                        unknown_etfs.append(result)
                                else:
                                    unknown_etfs.append({
                                        'ticker': ticker,
                                        'name': name,
                                        'regime': 'NOT_ANALYZED',
                                        'confidence': 0.0
                                    })
                            
                            # BULL ETFs
                            if bull_etfs:
                                st.markdown("🟢 **BULL Regime:**")
                                for etf in bull_etfs:
                                    confidence_text = f" (Confidence: {etf['confidence']:.1%})" if etf['confidence'] > 0 else ""
                                    st.markdown(f"""
                                    <div class="etf-item etf-bull">
                                        <span><strong>{etf['ticker']}</strong> - {etf['name']}</span>
                                        <span>{etf['regime']}{confidence_text}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # BEAR ETFs
                            if bear_etfs:
                                st.markdown("🔴 **BEAR Regime:**")
                                for etf in bear_etfs:
                                    confidence_text = f" (Confidence: {etf['confidence']:.1%})" if etf['confidence'] > 0 else ""
                                    st.markdown(f"""
                                    <div class="etf-item etf-bear">
                                        <span><strong>{etf['ticker']}</strong> - {etf['name']}</span>
                                        <span>{etf['regime']}{confidence_text}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Unknown ETFs
                            if unknown_etfs:
                                st.markdown("⚠️ **Unknown/Error:**")
                                for etf in unknown_etfs:
                                    st.markdown(f"""
                                    <div class="etf-item etf-unknown">
                                        <span><strong>{etf['ticker']}</strong> - {etf['name']}</span>
                                        <span>{etf['regime']}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # 종합 차트
                    st.subheader("📈 Regime Distribution")
                    
                    # 파이 차트
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['BULL', 'BEAR', 'Unknown'],
                        values=[bull_count, bear_count, unknown_count],
                        marker_colors=['#28a745', '#dc3545', '#ffc107']
                    )])
                    fig_pie.update_layout(title="Overall Market Regime Distribution")
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # 전략별 요약 차트
                    strategy_data = []
                    for strategy_name, preset in self.presets.items():
                        strategy_bull = 0
                        strategy_bear = 0
                        strategy_unknown = 0
                        
                        for ticker in preset['components'].keys():
                            if ticker in results:
                                regime = results[ticker]['regime']
                                if regime == 'BULL':
                                    strategy_bull += 1
                                elif regime == 'BEAR':
                                    strategy_bear += 1
                                else:
                                    strategy_unknown += 1
                            else:
                                strategy_unknown += 1
                        
                        strategy_data.append({
                            'Strategy': strategy_name,
                            'BULL': strategy_bull,
                            'BEAR': strategy_bear,
                            'Unknown': strategy_unknown
                        })
                    
                    strategy_df = pd.DataFrame(strategy_data)
                    
                    fig_strategy = go.Figure()
                    fig_strategy.add_trace(go.Bar(name='BULL', x=strategy_df['Strategy'], y=strategy_df['BULL'], marker_color='#28a745'))
                    fig_strategy.add_trace(go.Bar(name='BEAR', x=strategy_df['Strategy'], y=strategy_df['BEAR'], marker_color='#dc3545'))
                    fig_strategy.add_trace(go.Bar(name='Unknown', x=strategy_df['Strategy'], y=strategy_df['Unknown'], marker_color='#ffc107'))
                    
                    fig_strategy.update_layout(
                        title='Regime Distribution by Strategy',
                        barmode='stack',
                        xaxis_title='Strategy',
                        yaxis_title='Number of ETFs'
                    )
                    fig_strategy.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_strategy, use_container_width=True)
                    
                    st.success(f"✅ Analysis completed! {len(results)} assets analyzed.")
                else:
                    st.error("❌ Failed to analyze market regimes")
        else:
            # 캐시된 결과 표시
            if st.session_state.regime_cache and st.session_state.cache_timestamp:
                cache_age = datetime.now() - st.session_state.cache_timestamp
                st.info(f"📋 Cached results available (Updated {cache_age.seconds//60} minutes ago). Click 'Analyze' to refresh.")
    
    def refresh_all_regimes(self):
        """모든 체제 정보 새로고침"""
        try:
            st.session_state.regime_cache = {}
            st.session_state.cache_timestamp = None
            st.success("✅ Regime cache cleared! Click 'All Market Regimes' tab and 'Analyze' to refresh.")
        except Exception as e:
            st.error(f"Cache refresh failed: {str(e)}")
    
    def update_data(self, rs_length, timeframe, cross_days, use_jump):
        """데이터 업데이트"""
        try:
            st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("✅ Data updated successfully!")
        except Exception as e:
            st.error(f"Update failed: {str(e)}")
    
    def run_backtest(self, rs_length, timeframe, cross_days, use_jump, years):
        """백테스트 실행"""
        preset = st.session_state.selected_preset
        
        with st.spinner('Running backtest... This may take a few minutes'):
            try:
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
                
                if safe_data_check(portfolio_df):
                    st.session_state.portfolio_data = {
                        'portfolio': portfolio_df,
                        'trades': trades_df if safe_data_check(trades_df) else pd.DataFrame(),
                        'regime': regime_df if safe_data_check(regime_df) else pd.DataFrame(),
                        'metrics': strategy.calculate_performance_metrics(portfolio_df)
                    }
                    st.success("✅ Backtest completed!")
                else:
                    st.error("❌ Backtest failed - no results generated")
                    
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                st.info("💡 Try reducing the backtest period or check your internet connection")
    
    def display_backtest_results(self):
        """백테스트 결과 표시"""
        st.subheader("Backtest Results")
        
        if not safe_data_check(st.session_state.portfolio_data):
            st.info("💡 Run a backtest to see results")
            return
        
        data = st.session_state.portfolio_data
        metrics = data.get('metrics', {})
        
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
        portfolio_df = data.get('portfolio')
        
        if safe_data_check(portfolio_df):
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
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 상세 메트릭스 테이블
            if metrics:
                st.subheader("Detailed Metrics")
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                st.dataframe(metrics_df, use_container_width=True)
        else:
            st.warning("Portfolio data not available for charting")
    
    def download_results(self):
        """결과 다운로드"""
        try:
            if safe_data_check(st.session_state.portfolio_data):
                portfolio_df = st.session_state.portfolio_data['portfolio']
                if safe_data_check(portfolio_df):
                    csv = portfolio_df.to_csv()
                    st.download_button(
                        label="📥 Download Portfolio Data",
                        data=csv,
                        file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No portfolio data to download")
            else:
                st.warning("No data to download")
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            st.session_state.portfolio_data = None
            st.session_state.last_update = None
            st.session_state.regime_cache = {}
            st.session_state.cache_timestamp = None
            st.success("✅ All cache cleared!")
        except Exception as e:
            st.error(f"Cache clear failed: {str(e)}")


# Streamlit 앱 실행
def main():
    try:
        dashboard = EnhancedRealtimeDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard initialization failed: {str(e)}")
        st.info("💡 Try refreshing the page or checking your file paths")


if __name__ == "__main__":
    main()
