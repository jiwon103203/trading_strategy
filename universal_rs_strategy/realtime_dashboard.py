"""
실시간 모니터링 대시보드 - 효율화된 버전
웹 기반 인터랙티브 대시보드 (Streamlit 사용)
전체 ETF 지원 + 종합 Bull/Bear 상태 모니터링
2024년까지 학습, 2025년 추론 모델 적용
동적 Risk-Free Rate (^IRX) 기반 성과 분석
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import yfinance as yf
from preset_manager import PresetManager
from universal_rs_strategy import UniversalRSStrategy
from universal_jump_model import UniversalJumpModel
from universal_rs_with_jump import UniversalRSWithJumpModel
import concurrent.futures
from threading import Lock
import traceback

# Risk-free rate 유틸리티 import
try:
    from risk_free_rate_utils import RiskFreeRateManager, calculate_dynamic_sharpe_ratio, calculate_dynamic_sortino_ratio
    HAS_RF_UTILS = True
except ImportError:
    st.warning("⚠️ risk_free_rate_utils.py가 없습니다. 기본 risk-free rate (2%) 사용")
    HAS_RF_UTILS = False

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Universal RS Strategy Dashboard - Dynamic RF Edition",
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
.rf-info {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}
.dynamic-rf {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
    padding: 10px;
    margin: 5px 0;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

def safe_data_check(data):
    """안전한 데이터 검증"""
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

class StreamlinedRealtimeDashboard:
    """실시간 모니터링 대시보드 - 효율화된 버전"""
    
    def __init__(self):
        self._init_session_state()
        self.presets = {
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
        self.cache_duration = timedelta(minutes=30)
    
    def _init_session_state(self):
        """세션 상태 초기화"""
        defaults = {
            'selected_preset': None,
            'last_update': None,
            'portfolio_data': None,
            'regime_cache': {},
            'cache_timestamp': None,
            'selected_tickers_for_regime': [],
            'regime_analysis_mode': 'all',
            'rf_ticker': '^IRX',
            'default_rf_rate': 0.02,
            'debug_mode': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """대시보드 실행"""
        st.title("🚀 Universal RS Strategy Dashboard - Dynamic RF Edition")
        st.markdown("### Real-time Market Monitoring & Signal Generation")
        
        st.success("📊 **Enhanced Version**: 기존 수준의 오류 진단 기능 복원!")
        
        # Risk-Free Rate 상태 표시
        rf_status = "📊 동적" if HAS_RF_UTILS else "📌 고정"
        st.markdown(f"**🏦 Risk-Free Rate**: {st.session_state.rf_ticker} ({rf_status}) | **🎯 Training**: 2005-2024 | **🔮 Inference**: 2025")
        
        # 사이드바 및 메인 컨텐츠
        self.create_sidebar()
        
        if st.session_state.selected_preset:
            self.display_main_content()
        else:
            st.info("👈 Please select a strategy preset from the sidebar to begin")
    
    def create_sidebar(self):
        """사이드바 생성"""
        st.sidebar.header("Configuration")
        
        # 디버그 모드
        st.session_state.debug_mode = st.sidebar.checkbox(
            "🐛 Debug Mode", 
            value=st.session_state.debug_mode,
            help="Show detailed error information"
        )
        
        # Risk-Free Rate 설정
        self._configure_risk_free_rate()
        
        # 프리셋 선택
        self._select_preset()
        
        # 전략 파라미터
        self._strategy_parameters()
        
        # 실행 버튼
        self._control_buttons()
        
        # 고급 기능
        self._advanced_features()
        
        # 버전 정보
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Dashboard Info**")
        st.sidebar.info("Version: 3.0.1 (Error Handling Restored)")
        st.sidebar.success("✅ Enhanced error diagnostics")
        st.sidebar.success("✅ Detailed failure analysis")
    
    def _configure_risk_free_rate(self):
        """Risk-Free Rate 설정"""
        st.sidebar.subheader("🏦 Risk-Free Rate Settings")
        
        if HAS_RF_UTILS:
            rf_options = {
                '^IRX': '미국 3개월물 국채',
                '^TNX': '미국 10년물 국채',
                '^FVX': '미국 5년물 국채'
            }
            
            selected_rf = st.sidebar.selectbox(
                "Risk-Free Rate Ticker",
                options=list(rf_options.keys()),
                index=list(rf_options.keys()).index(st.session_state.rf_ticker),
                format_func=lambda x: f"{x} - {rf_options[x]}"
            )
            
            if selected_rf != st.session_state.rf_ticker:
                st.session_state.rf_ticker = selected_rf
                st.sidebar.success(f"RF 티커가 {selected_rf}로 변경되었습니다.")
            
            # RF 테스트
            if st.sidebar.button("🔍 RF 테스트"):
                self._test_risk_free_rate()
        else:
            default_rf_pct = st.sidebar.number_input(
                "Default RF Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.default_rf_rate * 100,
                step=0.1
            )
            st.session_state.default_rf_rate = default_rf_pct / 100
    
    def _test_risk_free_rate(self):
        """Risk-Free Rate 테스트"""
        with st.spinner("RF 데이터 확인 중..."):
            try:
                rf_manager = RiskFreeRateManager(st.session_state.rf_ticker, st.session_state.default_rf_rate)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                rf_data = rf_manager.download_risk_free_rate(start_date, end_date)
                
                if rf_data is not None and not rf_data.empty:
                    current_rate = rf_data.iloc[-1] * 100
                    avg_rate = rf_data.mean() * 100
                    st.sidebar.success(f"✅ 현재: {current_rate:.3f}%")
                    st.sidebar.info(f"30일 평균: {avg_rate:.3f}%")
                    st.success(f"🏦 RF 테스트 성공: {current_rate:.3f}%")
                else:
                    st.error(f"❌ {st.session_state.rf_ticker} 데이터를 가져올 수 없습니다.")
            except Exception as e:
                st.error(f"🚨 RF 테스트 실패: {str(e)}")
    
    def _select_preset(self):
        """프리셋 선택"""
        st.sidebar.subheader("Strategy Selection")
        preset_name = st.sidebar.selectbox(
            "Select Strategy Preset",
            options=list(self.presets.keys()),
            index=0 if st.session_state.selected_preset is None else None
        )
        
        if preset_name:
            st.session_state.selected_preset = self.presets[preset_name]
            st.session_state.preset_name = preset_name
    
    def _strategy_parameters(self):
        """전략 파라미터 설정"""
        st.sidebar.subheader("Strategy Parameters")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            rs_length = st.number_input("RS Length", value=20, min_value=10, max_value=50)
            use_jump = st.checkbox("Use Jump Model", value=True)
        
        with col2:
            timeframe = st.selectbox("Timeframe", ["daily", "weekly"])
            use_cross = st.checkbox("Use Cross Filter", value=False)
        
        cross_days = st.sidebar.number_input("Cross Days", value=30, min_value=5, max_value=90) if use_cross else None
        
        # 백테스트 설정
        st.sidebar.subheader("Backtest Settings")
        backtest_years = st.sidebar.slider("Backtest Period (Years)", 1, 5, 3)
        
        # 세션 상태에 저장
        st.session_state.strategy_params = {
            'rs_length': rs_length,
            'timeframe': timeframe,
            'cross_days': cross_days,
            'use_jump': use_jump,
            'backtest_years': backtest_years
        }
    
    def _control_buttons(self):
        """제어 버튼들"""
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Update", type="primary"):
                self._update_data()
        
        with col2:
            if st.button("📊 Backtest"):
                self._run_backtest()
    
    def _advanced_features(self):
        """고급 기능"""
        st.sidebar.subheader("Advanced Features")
        
        # Regime 분석 모드 선택
        regime_mode = st.sidebar.radio(
            "Regime Analysis Mode",
            ["All Markets", "Selected Tickers"],
            index=0 if st.session_state.regime_analysis_mode == 'all' else 1
        )
        st.session_state.regime_analysis_mode = 'all' if regime_mode == "All Markets" else 'selected'
        
        # 선택적 분석 모드
        if st.session_state.regime_analysis_mode == 'selected':
            all_tickers = self._get_all_tickers()
            selected_tickers = st.sidebar.multiselect(
                "Select Tickers to Analyze",
                options=list(all_tickers.keys()),
                default=st.session_state.selected_tickers_for_regime,
                format_func=lambda x: f"{x} - {all_tickers[x]}"
            )
            st.session_state.selected_tickers_for_regime = selected_tickers
        
        # 기능 버튼들
        if st.sidebar.button("🌍 Refresh Regimes"):
            self._refresh_regimes()
        
        if st.sidebar.button("💾 Download Results"):
            self._download_results()
        
        if st.sidebar.button("🔄 Clear Cache"):
            self._clear_cache()
    
    def _get_all_tickers(self):
        """모든 사용 가능한 티커 수집"""
        all_tickers = {}
        for strategy_name, preset in self.presets.items():
            benchmark = preset['benchmark']
            all_tickers[benchmark] = f"{strategy_name} Benchmark"
            
            for ticker, name in preset['components'].items():
                if ticker not in all_tickers:
                    all_tickers[ticker] = name
        
        return all_tickers
    
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
            rf_status = "📊 Dynamic" if HAS_RF_UTILS else "📌 Fixed"
            st.metric("Risk-Free Rate", f"{rf_status}")
        
        # Risk-Free Rate 상세 정보
        if HAS_RF_UTILS:
            self._display_rf_info()
        
        # 탭 생성
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Market Status", "🎯 Current Signals", "🌍 All Regimes", "📊 Backtest Results", "🏦 RF Analysis"
        ])
        
        with tab1:
            self._display_market_status()
        
        with tab2:
            self._display_current_signals()
        
        with tab3:
            self._display_all_market_regimes()
        
        with tab4:
            self._display_backtest_results()
        
        with tab5:
            self._display_rf_analysis()
    
    def _display_rf_info(self):
        """Risk-Free Rate 정보 표시"""
        try:
            rf_manager = RiskFreeRateManager(st.session_state.rf_ticker, st.session_state.default_rf_rate)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            rf_data = rf_manager.download_risk_free_rate(start_date, end_date)
            
            if rf_data is not None and not rf_data.empty:
                current_rate = rf_data.iloc[-1] * 100
                avg_rate = rf_data.mean() * 100
                min_rate = rf_data.min() * 100
                max_rate = rf_data.max() * 100
                
                st.markdown(f"""
                <div class="rf-info">
                    <strong>🏦 Dynamic Risk-Free Rate Status ({st.session_state.rf_ticker})</strong><br>
                    📊 Current: {current_rate:.3f}% | 📈 30-day Avg: {avg_rate:.3f}% | 
                    📉 Range: {min_rate:.3f}% - {max_rate:.3f}%
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class="rf-info">
                <strong>❌ Risk-Free Rate Error</strong><br>
                {str(e)[:100]}...
            </div>
            """, unsafe_allow_html=True)
    
    def _display_market_status(self):
        """시장 상태 표시"""
        st.subheader("Market Regime Analysis")
        st.markdown("**Training Period**: 2005-2024 | **Inference Period**: 2025")
        
        preset = st.session_state.selected_preset
        
        if st.button("🔍 Analyze Market Regime"):
            with st.spinner("Analyzing market regime..."):
                try:
                    jump_model = UniversalJumpModel(
                        benchmark_ticker=preset['benchmark'],
                        benchmark_name=preset['name'],
                        training_cutoff_date=datetime(2024, 12, 31),
                        rf_ticker=st.session_state.rf_ticker,
                        default_rf_rate=st.session_state.default_rf_rate
                    )
                    
                    current_regime = jump_model.get_current_regime_with_training_cutoff()
                    
                    if current_regime:
                        self._display_regime_info(current_regime)
                        st.success("✅ Market regime analysis completed!")
                    else:
                        st.error("❌ Unable to analyze market regime")
                        
                except Exception as e:
                    st.error(f"Market regime analysis failed: {str(e)}")
                    if st.session_state.debug_mode:
                        st.code(traceback.format_exc())
    
    def _display_regime_info(self, regime_info):
        """체제 정보 표시"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            regime_emoji = "🟢" if regime_info['regime'] == 'BULL' else "🔴"
            oos_indicator = "🔮" if regime_info.get('is_out_of_sample', False) else "📚"
            st.metric("Current Regime", f"{regime_emoji} {regime_info['regime']} {oos_indicator}")
        
        with col2:
            confidence = safe_get_value(regime_info['confidence'], 0.5)
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            current_rf = regime_info.get('current_rf_rate', st.session_state.default_rf_rate * 100)
            st.metric("Current RF", f"{current_rf:.3f}%")
        
        with col4:
            rf_status = "📊 Dynamic" if regime_info.get('dynamic_rf_used', False) else "📌 Fixed"
            st.metric("RF Type", rf_status)
        
        with col5:
            features = regime_info.get('features', {})
            risk_adj = safe_get_value(features.get('risk_adjusted_return', 0), 0)
            st.metric("Risk-Adj Return", f"{risk_adj:.3f}")
        
        # 추가 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📅 Analysis Date: {regime_info['date'].strftime('%Y-%m-%d')}")
        with col2:
            oos_status = "Out-of-Sample" if regime_info.get('is_out_of_sample', False) else "In-Sample"
            st.info(f"🔮 Status: {oos_status}")
        with col3:
            rf_ticker = regime_info.get('rf_ticker', st.session_state.rf_ticker)
            st.info(f"🏦 RF Ticker: {rf_ticker}")
    
    def _display_current_signals(self):
        """현재 투자 신호 표시"""
        st.subheader("Current Investment Signals")
        
        preset = st.session_state.selected_preset
        
        if st.button("🎯 Analyze Investment Signals"):
            with st.spinner('Analyzing components...'):
                try:
                    # 체제 분석
                    jump_model = UniversalJumpModel(
                        benchmark_ticker=preset['benchmark'],
                        benchmark_name=preset['name'],
                        training_cutoff_date=datetime(2024, 12, 31),
                        rf_ticker=st.session_state.rf_ticker,
                        default_rf_rate=st.session_state.default_rf_rate
                    )
                    
                    current_regime = jump_model.get_current_regime_with_training_cutoff()
                    
                    if current_regime:
                        self._display_regime_info(current_regime)
                        
                        if current_regime['regime'] == 'BEAR':
                            st.error("🔴 **BEAR Market Detected** - Investment suspension recommended")
                            return
                    
                    # RS 전략 분석
                    self._analyze_rs_strategy(preset, current_regime)
                    
                except Exception as e:
                    st.error(f"Signal analysis failed: {str(e)}")
                    if st.session_state.debug_mode:
                        st.code(traceback.format_exc())
    
    def _analyze_rs_strategy(self, preset, current_regime):
        """RS 전략 분석"""
        strategy = UniversalRSStrategy(
            benchmark=preset['benchmark'],
            components=preset['components'],
            name=preset['name'],
            rf_ticker=st.session_state.rf_ticker,
            default_rf_rate=st.session_state.default_rf_rate
        )
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)
        
        price_data, benchmark_data = strategy.get_price_data(start_date, end_date)
        
        if safe_data_check(price_data) and safe_data_check(benchmark_data):
            selected = strategy.select_components(price_data, benchmark_data, end_date)
            
            if safe_data_check(selected):
                self._display_rs_results(selected, current_regime)
            else:
                st.warning("⚠️ No components currently meet the investment criteria")
        else:
            st.error("❌ Unable to fetch market data")
    
    def _display_rs_results(self, selected, current_regime):
        """RS 결과 표시"""
        signals_df = pd.DataFrame(selected)
        signals_df['RS_Score'] = (signals_df['rs_ratio'] + signals_df['rs_momentum']) / 2
        signals_df = signals_df.sort_values('RS_Score', ascending=False)
        
        # 투자 권고 메시지
        if current_regime and current_regime['regime'] == 'BULL':
            current_rf = current_regime.get('current_rf_rate', 0)
            
            if current_rf > 4.0:
                st.warning(f"🟡 **BULL Market + High RF ({current_rf:.2f}%)** - Conservative investment recommended")
            elif current_rf < 1.0:
                st.success(f"🟢 **BULL Market + Low RF ({current_rf:.2f}%)** - Aggressive investment opportunity!")
            else:
                st.success(f"🟢 **BULL Market + Normal RF ({current_rf:.2f}%)** - Standard investment execution")
        
        st.info(f"📊 {len(selected)} Strong Components identified")
        
        # 테이블 표시
        st.dataframe(
            signals_df[['name', 'rs_ratio', 'rs_momentum', 'RS_Score']],
            use_container_width=True
        )
        
        # 차트
        col1, col2 = st.columns(2)
        
        with col1:
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
    
    def _display_all_market_regimes(self):
        """모든 시장 체제 현황 표시"""
        st.subheader("🌍 All Market Regimes Overview")
        
        # 분석 모드 표시
        if st.session_state.regime_analysis_mode == 'selected':
            if st.session_state.selected_tickers_for_regime:
                st.markdown(f"**Selected Tickers Analysis** ({len(st.session_state.selected_tickers_for_regime)} tickers)")
            else:
                st.warning("No tickers selected. Please select tickers from the sidebar.")
                return
        else:
            st.markdown("**Full Market Analysis** (All ETFs across all strategies)")
        
        button_text = "🔄 Analyze Selected Tickers" if st.session_state.regime_analysis_mode == 'selected' else "🔄 Analyze All Regimes"
        
        if st.button(button_text, type="primary"):
            with st.spinner("Analyzing market regimes..."):
                if st.session_state.regime_analysis_mode == 'selected':
                    if not st.session_state.selected_tickers_for_regime:
                        st.error("Please select at least one ticker")
                        return
                    results = self._analyze_selected_regimes()
                else:
                    results = self._analyze_all_regimes()
                
                if results:
                    self._display_regime_results(results)
                else:
                    st.error("❌ Failed to analyze market regimes")
    
    def _analyze_selected_regimes(self):
        """선택된 티커들의 체제 분석"""
        results = {}
        
        for ticker in st.session_state.selected_tickers_for_regime:
            try:
                # 티커 정보 찾기
                ticker_name = self._find_ticker_name(ticker)
                
                result = self._analyze_single_ticker(ticker, ticker_name)
                results[ticker] = result
                
                # 실시간 결과 표시
                if result['status'] == 'success':
                    st.success(f"✅ {ticker}: {result['regime']} (Confidence: {result['confidence']:.1%})")
                else:
                    st.error(f"❌ {ticker}: {result['regime']}")
                    
            except Exception as e:
                st.error(f"❌ {ticker}: Analysis failed - {str(e)}")
                results[ticker] = {
                    'ticker': ticker,
                    'name': ticker,
                    'regime': 'ANALYSIS_ERROR',
                    'confidence': 0.0,
                    'status': 'error'
                }
        
        return results
    
    def _analyze_all_regimes(self):
        """모든 ETF 체제 분석"""
        all_etfs = {}
        benchmarks = {}
        
        # 모든 ETF 수집
        for strategy_name, preset in self.presets.items():
            for ticker, name in preset['components'].items():
                if ticker not in all_etfs:
                    all_etfs[ticker] = {
                        'name': name,
                        'strategies': [strategy_name]
                    }
                else:
                    all_etfs[ticker]['strategies'].append(strategy_name)
            
            # 벤치마크 추가
            benchmark = preset['benchmark']
            if benchmark not in benchmarks:
                benchmarks[benchmark] = f"{strategy_name} Benchmark"
        
        # 캐시 확인
        now = datetime.now()
        if (st.session_state.cache_timestamp and 
            now - st.session_state.cache_timestamp < self.cache_duration and
            st.session_state.regime_cache):
            st.info("📋 Using cached results")
            return st.session_state.regime_cache
        
        results = {}
        
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_items = len(all_etfs) + len(benchmarks)
        processed = 0
        
        # 벤치마크 분석
        for ticker, name in benchmarks.items():
            status_text.text(f"Analyzing benchmark: {ticker}")
            
            result = self._analyze_single_ticker(ticker, name)
            results[ticker] = result
            results[ticker]['type'] = 'benchmark'
            
            processed += 1
            progress_bar.progress(processed / total_items)
        
        # ETF 분석
        for ticker, info in all_etfs.items():
            status_text.text(f"Analyzing ETF: {ticker}")
            
            result = self._analyze_single_ticker(ticker, info['name'])
            result['type'] = 'etf'
            result['strategies'] = info['strategies']
            results[ticker] = result
            
            processed += 1
            progress_bar.progress(processed / total_items)
        
        # 캐시 업데이트
        st.session_state.regime_cache = results
        st.session_state.cache_timestamp = now
        
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def _analyze_single_ticker(self, ticker, name):
        """단일 티커 분석 - 기존 코드 수준의 오류 처리 복원"""
        try:
            # 1단계: 데이터 사전 검증 (기존 코드 로직 복원)
            try:
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period="5y", timeout=30)
                
                if hist.empty:
                    return {
                        'ticker': ticker, 'name': name, 'regime': 'NO_HISTORICAL_DATA',
                        'confidence': 0.0, 'status': 'data_unavailable'
                    }
                
                # 데이터 품질 검사
                hist_clean = hist.dropna()
                if len(hist_clean) < 300:
                    return {
                        'ticker': ticker, 'name': name, 'regime': 'INSUFFICIENT_DATA',
                        'confidence': 0.0, 'status': 'insufficient_data'
                    }
                    
            except Exception as e:
                return {
                    'ticker': ticker, 'name': name, 'regime': 'DATA_FETCH_ERROR',
                    'confidence': 0.0, 'status': 'data_fetch_error',
                    'error': str(e)
                }
            
            # 2단계: JumpModel 초기화 (기존 파라미터 복원)
            try:
                jump_model = UniversalJumpModel(
                    benchmark_ticker=ticker,
                    benchmark_name=name,
                    jump_penalty=50.0,  # 기존 코드 파라미터 복원
                    use_paper_features_only=True,  # 기존 코드 파라미터 복원
                    training_cutoff_date=datetime(2024, 12, 31),
                    rf_ticker=st.session_state.rf_ticker,
                    default_rf_rate=st.session_state.default_rf_rate
                )
                
            except Exception as e:
                return {
                    'ticker': ticker, 'name': name, 'regime': 'MODEL_INIT_ERROR',
                    'confidence': 0.0, 'status': 'model_init_error',
                    'error': str(e)
                }
            
            # 3단계: 체제 분석 실행
            try:
                current_regime = jump_model.get_current_regime_with_training_cutoff()
                
                if current_regime is None:
                    return {
                        'ticker': ticker, 'name': name, 'regime': 'NO_REGIME_DATA',
                        'confidence': 0.0, 'status': 'no_regime_data'
                    }
                
                if not isinstance(current_regime, dict):
                    return {
                        'ticker': ticker, 'name': name, 'regime': 'INVALID_REGIME_DATA',
                        'confidence': 0.0, 'status': 'invalid_regime_data'
                    }
                
                # 필수 키 확인
                required_keys = ['regime', 'confidence', 'date']
                missing_keys = [key for key in required_keys if key not in current_regime]
                
                if missing_keys:
                    return {
                        'ticker': ticker, 'name': name, 'regime': 'INCOMPLETE_REGIME_DATA',
                        'confidence': 0.0, 'status': 'incomplete_regime_data',
                        'missing_keys': missing_keys
                    }
                
            except Exception as e:
                return {
                    'ticker': ticker, 'name': name, 'regime': 'REGIME_ANALYSIS_ERROR',
                    'confidence': 0.0, 'status': 'analysis_error',
                    'error': str(e)
                }
            
            # 4단계: 결과 처리
            try:
                # 안전한 값 추출
                regime = current_regime.get('regime', 'UNKNOWN')
                confidence = current_regime.get('confidence', 0.0)
                
                # confidence 안전 변환
                if isinstance(confidence, pd.Series):
                    if len(confidence) > 0:
                        confidence = float(confidence.iloc[-1])
                    else:
                        confidence = 0.0
                elif not isinstance(confidence, (int, float)):
                    confidence = 0.0
                
                # 신뢰도 범위 검증
                confidence = max(0.0, min(1.0, confidence))
                
                result = {
                    'ticker': ticker,
                    'name': name,
                    'regime': regime,
                    'confidence': confidence,
                    'is_out_of_sample': current_regime.get('is_out_of_sample', False),
                    'analysis_date': current_regime.get('date', datetime.now()).strftime('%Y-%m-%d') if hasattr(current_regime.get('date'), 'strftime') else str(current_regime.get('date', 'Unknown')),
                    'rf_ticker': current_regime.get('rf_ticker', st.session_state.rf_ticker),
                    'current_rf_rate': current_regime.get('current_rf_rate', st.session_state.default_rf_rate * 100),
                    'dynamic_rf_used': current_regime.get('dynamic_rf_used', False),
                    'status': 'success'
                }
                
                return result
                
            except Exception as e:
                return {
                    'ticker': ticker, 'name': name, 'regime': 'RESULTS_PROCESSING_ERROR',
                    'confidence': 0.0, 'status': 'results_processing_error',
                    'error': str(e)
                }
                
        except Exception as e:
            return {
                'ticker': ticker, 'name': name, 'regime': 'FATAL_ERROR',
                'confidence': 0.0, 'status': 'fatal_error',
                'error': str(e)
            }
    
    def _find_ticker_name(self, ticker):
        """티커 이름 찾기"""
        for strategy_name, preset in self.presets.items():
            if preset['benchmark'] == ticker:
                return f"{strategy_name} Benchmark"
            if ticker in preset['components']:
                return preset['components'][ticker]
        return ticker
    
    def _display_regime_results(self, results):
        """체제 분석 결과 표시"""
        # 통계 요약
        bull_count = sum(1 for r in results.values() if r['regime'] == 'BULL')
        bear_count = sum(1 for r in results.values() if r['regime'] == 'BEAR')
        unknown_count = len(results) - bull_count - bear_count
        
        oos_count = sum(1 for r in results.values() if r.get('is_out_of_sample', False))
        dynamic_rf_count = sum(1 for r in results.values() if r.get('dynamic_rf_used', False))
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Total Assets", len(results))
        with col2:
            st.metric("🟢 BULL", bull_count)
        with col3:
            st.metric("🔴 BEAR", bear_count)
        with col4:
            st.metric("⚠️ Issues", unknown_count)
        with col5:
            st.metric("🔮 Out-of-Sample", oos_count)
        with col6:
            st.metric("📊 Dynamic RF", dynamic_rf_count)
        
        # 차트들
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 체제 분포
            if bull_count + bear_count > 0:
                fig_regime = go.Figure(data=[go.Pie(
                    labels=['BULL', 'BEAR'],
                    values=[bull_count, bear_count],
                    marker_colors=['#28a745', '#dc3545']
                )])
                fig_regime.update_layout(title="Regime Distribution")
                st.plotly_chart(fig_regime, use_container_width=True)
        
        with col2:
            # Out-of-Sample 분포
            in_sample_count = len(results) - oos_count
            fig_oos = go.Figure(data=[go.Pie(
                labels=['Out-of-Sample', 'In-Sample'],
                values=[oos_count, in_sample_count],
                marker_colors=['#17a2b8', '#6c757d']
            )])
            fig_oos.update_layout(title="Sample Distribution")
            st.plotly_chart(fig_oos, use_container_width=True)
        
        with col3:
            # Dynamic RF 분포
            fixed_rf_count = len(results) - dynamic_rf_count
            fig_rf = go.Figure(data=[go.Pie(
                labels=['Dynamic RF', 'Fixed RF'],
                values=[dynamic_rf_count, fixed_rf_count],
                marker_colors=['#9c27b0', '#795548']
            )])
            fig_rf.update_layout(title="Risk-Free Rate Type")
            st.plotly_chart(fig_rf, use_container_width=True)
        
        # 성공률 정보
        success_rate = (bull_count + bear_count) / len(results) * 100
        st.success(f"✅ Analysis completed! Success rate: {success_rate:.1f}%")
    
    def _display_backtest_results(self):
        """백테스트 결과 표시"""
        st.subheader("Backtest Results")
        
        if not safe_data_check(st.session_state.portfolio_data):
            st.info("💡 Run a backtest to see results")
            return
        
        data = st.session_state.portfolio_data
        
        # 백테스트 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jump Model", "Enabled" if data.get('use_jump_model', False) else "Disabled")
        with col2:
            rf_status = "📊 Dynamic" if data.get('dynamic_rf_used', False) else "📌 Fixed"
            st.metric("Risk-Free Rate", rf_status)
        with col3:
            st.metric("Training Cutoff", data.get('training_cutoff', 'N/A'))
        
        # 성과 지표
        metrics = data.get('metrics', {})
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", metrics.get('총 수익률', 'N/A'))
            with col2:
                st.metric("Annual Return", metrics.get('연율화 수익률', 'N/A'))
            with col3:
                st.metric("Sharpe Ratio", metrics.get('샤프 비율 (동적)', 'N/A'))
            with col4:
                st.metric("Max Drawdown", metrics.get('최대 낙폭', 'N/A'))
        
        # 포트폴리오 차트
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
    
    def _display_rf_analysis(self):
        """Risk-Free Rate 분석"""
        st.subheader("🏦 Risk-Free Rate Analysis")
        
        if not HAS_RF_UTILS:
            st.warning("⚠️ Dynamic Risk-Free Rate analysis requires risk_free_rate_utils.py")
            return
        
        # 분석 기간 선택
        period_options = {30: "1 Month", 90: "3 Months", 180: "6 Months", 365: "1 Year"}
        selected_days = st.selectbox(
            "Analysis Period", 
            options=list(period_options.keys()),
            format_func=lambda x: period_options[x],
            index=3
        )
        
        if st.button("📊 Analyze RF Data", type="primary"):
            with st.spinner("Analyzing Risk-Free Rate data..."):
                self._analyze_rf_data(selected_days)
    
    def _analyze_rf_data(self, days):
        """Risk-Free Rate 데이터 분석"""
        try:
            rf_manager = RiskFreeRateManager(st.session_state.rf_ticker, st.session_state.default_rf_rate)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            rf_data = rf_manager.download_risk_free_rate(start_date, end_date)
            
            if rf_data is not None and not rf_data.empty:
                stats = rf_manager.get_risk_free_rate_stats(start_date, end_date)
                
                # 메트릭 표시
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Rate", f"{stats['end_rate']:.3f}%")
                with col2:
                    st.metric("Average Rate", f"{stats['mean_rate']:.3f}%")
                with col3:
                    st.metric("Volatility", f"{stats['std_rate']:.3f}%")
                with col4:
                    rate_change = stats['end_rate'] - stats['start_rate']
                    st.metric("Period Change", f"{rate_change:+.3f}%")
                
                # RF 차트
                fig_rf = go.Figure()
                fig_rf.add_trace(go.Scatter(
                    x=rf_data.index,
                    y=rf_data.values * 100,
                    mode='lines',
                    name=f'Risk-Free Rate ({st.session_state.rf_ticker})',
                    line=dict(color='blue', width=2)
                ))
                fig_rf.add_hline(y=stats['mean_rate'], line_dash="dash", 
                               line_color="red", annotation_text=f"Average: {stats['mean_rate']:.3f}%")
                fig_rf.update_layout(
                    title=f"Risk-Free Rate Trend ({st.session_state.rf_ticker})",
                    xaxis_title="Date",
                    yaxis_title="Rate (%)",
                    height=400
                )
                st.plotly_chart(fig_rf, use_container_width=True)
                
                # 투자 시사점
                self._display_investment_implications(stats['end_rate'])
                
        except Exception as e:
            st.error(f"RF analysis failed: {str(e)}")
    
    def _display_investment_implications(self, current_rate):
        """투자 시사점 표시"""
        st.subheader("💡 Investment Implications")
        
        if current_rate > 4.0:
            st.error(f"🔴 **High RF Environment** ({current_rate:.2f}%)")
            st.markdown("""
            - **Cash becomes attractive**: High opportunity cost for risky assets
            - **Strategy**: Consider conservative positioning, high-quality assets
            """)
        elif current_rate < 1.0:
            st.success(f"🟢 **Low RF Environment** ({current_rate:.2f}%)")
            st.markdown("""
            - **Risk asset friendly**: Low opportunity cost encourages risk-taking
            - **Strategy**: Consider aggressive positioning, growth assets
            """)
        else:
            st.info(f"🟡 **Normal RF Environment** ({current_rate:.2f}%)")
            st.markdown("""
            - **Balanced outlook**: Moderate opportunity cost for risky assets
            - **Strategy**: Maintain balanced portfolio allocation
            """)
    
    def _update_data(self):
        """데이터 업데이트"""
        try:
            st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("✅ Data updated successfully!")
        except Exception as e:
            st.error(f"Update failed: {str(e)}")
    
    def _run_backtest(self):
        """백테스트 실행"""
        preset = st.session_state.selected_preset
        params = st.session_state.strategy_params
        
        with st.spinner('Running backtest...'):
            try:
                strategy = UniversalRSWithJumpModel(
                    preset_config=preset,
                    rs_length=params['rs_length'],
                    rs_timeframe=params['timeframe'],
                    rs_recent_cross_days=params['cross_days'],
                    use_jump_model=params['use_jump'],
                    rf_ticker=st.session_state.rf_ticker,
                    default_rf_rate=st.session_state.default_rf_rate,
                    training_cutoff_date=datetime(2024, 12, 31)
                )
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365*params['backtest_years'])
                
                portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)
                
                if safe_data_check(portfolio_df):
                    st.session_state.portfolio_data = {
                        'portfolio': portfolio_df,
                        'trades': trades_df,
                        'regime': regime_df,
                        'metrics': strategy.calculate_performance_metrics(portfolio_df),
                        'use_jump_model': params['use_jump'],
                        'training_cutoff': '2024-12-31' if params['use_jump'] else 'N/A',
                        'rf_ticker': st.session_state.rf_ticker,
                        'dynamic_rf_used': HAS_RF_UTILS
                    }
                    st.success("✅ Backtest completed!")
                else:
                    st.error("❌ Backtest failed - no results generated")
                    
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                if st.session_state.debug_mode:
                    st.code(traceback.format_exc())
    
    def _refresh_regimes(self):
        """체제 새로고침"""
        st.session_state.regime_cache = {}
        st.session_state.cache_timestamp = None
        st.success("✅ Regime cache cleared!")
    
    def _download_results(self):
        """결과 다운로드"""
        try:
            if safe_data_check(st.session_state.portfolio_data):
                portfolio_df = st.session_state.portfolio_data['portfolio']
                if safe_data_check(portfolio_df):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv = portfolio_df.to_csv()
                    st.download_button(
                        label="📥 Download Portfolio Data",
                        data=csv,
                        file_name=f"portfolio_{timestamp}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No data to download")
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
    
    def _clear_cache(self):
        """캐시 정리"""
        keys_to_clear = ['portfolio_data', 'last_update', 'regime_cache', 'cache_timestamp']
        for key in keys_to_clear:
            if key in st.session_state:
                st.session_state[key] = None if key in ['portfolio_data', 'last_update'] else {}
        st.success("✅ Cache cleared!")

def main():
    """메인 함수"""
    try:
        dashboard = StreamlinedRealtimeDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"🚨 Dashboard initialization failed: {str(e)}")
        st.info("💡 Try refreshing the page")

if __name__ == "__main__":
    main()
