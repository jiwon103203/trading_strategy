"""
실시간 모니터링 대시보드 - 통합된 Jump Model 사용 (효율화 버전)
웹 기반 인터랙티브 대시보드 (Streamlit 사용)
전체 ETF 지원 + 종합 Bull/Bear 상태 모니터링
2024년까지 학습, 2025년 추론 모델 적용
Risk-Free Rate 분석 기능 제거 (간소화)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import yfinance as yf
from ..core.preset_manager import PresetManager
from ..core.universal_rs_strategy import UniversalRSStrategy
from ..core.universal_jump_model import UniversalJumpModel
from ..core.universal_rs_with_jump import UniversalRSWithJumpModel
from ..core.utils import safe_float, safe_extract_close, validate_data, calculate_basic_metrics, print_status
import concurrent.futures
from threading import Lock
import traceback

# Risk-free rate 유틸리티 import
try:
    from ..advanced.risk_free_rate_utils import RiskFreeRateManager, calculate_dynamic_sharpe_ratio, calculate_dynamic_sortino_ratio
    HAS_RF_UTILS = True
except ImportError:
    st.warning("⚠️ risk_free_rate_utils.py가 없습니다. 기본 risk-free rate (2%) 사용")
    HAS_RF_UTILS = False

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Universal RS Strategy Dashboard - Unified Model Edition",
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
.unified-model {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
    padding: 10px;
    margin: 5px 0;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

def safe_data_check(data):
    """안전한 데이터 검증 - utils 기반"""
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
    """안전한 값 추출 - utils 기반"""
    return safe_float(value, default)

class UnifiedRealtimeDashboard:
    """실시간 모니터링 대시보드 - 통합된 Jump Model 사용 (효율화 버전)"""
    
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
        """세션 상태 초기화 - 통합 모델 기준"""
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
            'use_dynamic_rf': True,
            'use_paper_features_only': True,
            'jump_penalty': 50.0,
            'debug_mode': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """대시보드 실행"""
        st.title("🚀 Universal RS Strategy Dashboard - Unified Model Edition")
        st.markdown("### Real-time Market Monitoring & Signal Generation with Unified Jump Model")
        
        st.success("🔧 **Unified Model**: Jump Model 특징 계산 코드가 통합되었습니다!")
        
        # Risk-Free Rate 상태 표시 (간소화)
        rf_status = "📊 Dynamic" if (HAS_RF_UTILS and st.session_state.use_dynamic_rf) else "📌 Fixed"
        feature_type = "논문 정확한 3특징" if st.session_state.use_paper_features_only else "논문 기반 + 추가"
        st.markdown(f"**🏦 Risk-Free Rate**: ^IRX ({rf_status}) | **🎯 Features**: {feature_type} | **🔮 Jump Penalty**: {st.session_state.jump_penalty}")
        
        # 사이드바 및 메인 컨텐츠
        self.create_sidebar()
        
        if st.session_state.selected_preset:
            self.display_main_content()
        else:
            st.info("👈 Please select a strategy preset from the sidebar to begin")
    
    def create_sidebar(self):
        """사이드바 생성 - 통합 모델 설정 포함"""
        st.sidebar.header("Configuration")
        
        # 디버그 모드
        st.session_state.debug_mode = st.sidebar.checkbox(
            "🐛 Debug Mode", 
            value=st.session_state.debug_mode,
            help="Show detailed error information"
        )
        
        # 통합 모델 설정
        self._configure_unified_model()
        
        # Risk-Free Rate 설정 (간소화)
        self._configure_risk_free_rate_simple()
        
        # 프리셋 선택
        self._select_preset()
        
        # 실행 버튼
        self._control_buttons()
        
        # 고급 기능
        self._advanced_features()
        
        # 버전 정보
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Dashboard Info**")
        st.sidebar.info("Version: 4.0.0 (Unified Model)")
        st.sidebar.success("✅ Jump Model 특징 계산 통합")
        st.sidebar.success("✅ RF 분석 기능 간소화")
    
    def _configure_unified_model(self):
        """통합 모델 설정"""
        st.sidebar.subheader("🔧 Unified Jump Model Settings")
        
        # Feature Type 설정
        feature_mode = st.sidebar.radio(
            "Feature Type",
            ["논문 정확한 3특징", "논문 기반 + 추가 특징"],
            index=0 if st.session_state.use_paper_features_only else 1,
            help="논문 정확한 3특징: realtime_dashboard 기준 최적화된 특징만 사용"
        )
        
        st.session_state.use_paper_features_only = (feature_mode == "논문 정확한 3특징")
        
        # Jump Penalty 설정
        st.session_state.jump_penalty = st.sidebar.number_input(
            "Jump Penalty",
            min_value=10.0,
            max_value=100.0,
            value=st.session_state.jump_penalty,
            step=10.0,
            help="체제 전환 억제 강도 (높을수록 안정적)"
        )
        
        # 모델 상태 표시
        if st.session_state.use_paper_features_only:
            st.sidebar.success("📊 논문 정확한 3특징 (최적화)")
        else:
            st.sidebar.info("📈 논문 기반 + 추가 특징")
        
        st.sidebar.info(f"🎯 Jump Penalty: {st.session_state.jump_penalty}")
    
    def _configure_risk_free_rate_simple(self):
        """Risk-Free Rate 설정 (간소화 버전)"""
        st.sidebar.subheader("🏦 Risk-Free Rate Settings")
        
        # Dynamic vs Fixed 선택
        rf_mode = st.sidebar.radio(
            "Risk-Free Rate Mode",
            ["Dynamic (^IRX)", "Fixed Rate"],
            index=0 if st.session_state.get('use_dynamic_rf', True) else 1,
            help="Dynamic: Uses real-time 3-month Treasury rate. Fixed: Uses static rate."
        )
        
        use_dynamic_rf = rf_mode == "Dynamic (^IRX)"
        st.session_state.use_dynamic_rf = use_dynamic_rf
        st.session_state.rf_ticker = '^IRX'
        
        if use_dynamic_rf:
            if HAS_RF_UTILS:
                st.sidebar.success("📊 Using Dynamic RF (^IRX)")
                
                # RF 간단 테스트
                if st.sidebar.button("🔍 Test ^IRX"):
                    self._test_risk_free_rate_simple()
            else:
                st.sidebar.error("❌ Dynamic RF not available")
                st.sidebar.info("risk_free_rate_utils.py required")
                st.session_state.use_dynamic_rf = False
        else:
            # Fixed Rate 설정
            default_rf_pct = st.sidebar.number_input(
                "Fixed RF Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.default_rf_rate * 100,
                step=0.1,
                help="Fixed risk-free rate for all calculations"
            )
            st.session_state.default_rf_rate = default_rf_pct / 100
            st.sidebar.info(f"📌 Using Fixed RF: {default_rf_pct:.1f}%")
    
    def _test_risk_free_rate_simple(self):
        """Risk-Free Rate 간단 테스트"""
        with st.spinner("Testing ^IRX data..."):
            try:
                rf_manager = RiskFreeRateManager(st.session_state.rf_ticker, st.session_state.default_rf_rate)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                rf_data = rf_manager.download_risk_free_rate(start_date, end_date)
                
                if validate_data(rf_data, 1):
                    current_rate = safe_float(rf_data.iloc[-1]) * 100
                    avg_rate = safe_float(rf_data.mean()) * 100
                    st.sidebar.success(f"✅ Current: {current_rate:.3f}%")
                    st.sidebar.info(f"30-day Avg: {avg_rate:.3f}%")
                    st.success(f"🏦 ^IRX Test Success: {current_rate:.3f}%")
                else:
                    st.error(f"❌ Cannot fetch ^IRX data")
            except Exception as e:
                st.error(f"🚨 ^IRX Test Failed: {str(e)}")
    
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
            feature_status = "📊 3특징" if st.session_state.use_paper_features_only else "📈 확장특징"
            st.metric("Features", f"{feature_status}")
        
        # 통합 모델 정보
        st.markdown(f"""
        <div class="unified-model">
            <strong>🔧 Unified Jump Model Status</strong><br>
            📊 Feature Type: {'논문 정확한 3특징' if st.session_state.use_paper_features_only else '논문 기반 + 추가 특징'}<br>
            🎯 Jump Penalty: {st.session_state.jump_penalty} | 
            🏦 RF Ticker: {st.session_state.rf_ticker} | 
            📅 Training Cutoff: 2024-12-31
        </div>
        """, unsafe_allow_html=True)
        
        # 탭 생성 (RF 분석 탭 제거)
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Market Status", "🎯 Current Signals", "🌍 All Regimes", "📊 Backtest Results"
        ])
        
        with tab1:
            self._display_market_status()
        
        with tab2:
            self._display_current_signals()
        
        with tab3:
            self._display_all_market_regimes()
        
        with tab4:
            self._display_backtest_results()
    
    def _display_market_status(self):
        """시장 상태 표시"""
        st.subheader("Market Regime Analysis")
        st.markdown("**Unified Model**: Integrated feature calculation | **Training**: 2005-2024 | **Inference**: 2025")
        
        preset = st.session_state.selected_preset
        
        if st.button("🔍 Analyze Market Regime (Unified Model)"):
            with st.spinner("Analyzing market regime with unified model..."):
                try:
                    # 통합 모델 사용
                    jump_model = UniversalJumpModel(
                        benchmark_ticker=preset['benchmark'],
                        benchmark_name=preset['name'],
                        jump_penalty=st.session_state.jump_penalty,
                        training_cutoff_date=datetime(2024, 12, 31)
                    )
                    
                    current_regime = jump_model.get_current_regime_with_training_cutoff()
                    
                    if current_regime:
                        self._display_regime_info(current_regime)
                        st.success("✅ Market regime analysis completed with unified model!")
                    else:
                        st.error("❌ Unable to analyze market regime")
                        
                except Exception as e:
                    st.error(f"Market regime analysis failed: {str(e)}")
                    if st.session_state.debug_mode:
                        st.code(traceback.format_exc())
    
    def _display_regime_info(self, regime_info):
        """체제 정보 표시 - 통합 모델 버전"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            regime_emoji = "🟢" if regime_info['regime'] == 'BULL' else "🔴"
            oos_indicator = "🔮" if regime_info.get('is_out_of_sample', False) else "📚"
            st.metric("Current Regime", f"{regime_emoji} {regime_info['regime']} {oos_indicator}")
        
        with col2:
            confidence = safe_get_value(regime_info['confidence'], 0.5)
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            feature_type = regime_info.get('feature_type', 'Unknown')
            feature_short = "3특징" if "논문 정확한 3특징" in feature_type else "확장특징"
            st.metric("Features", feature_short)
        
        with col4:
            training_cutoff = regime_info.get('training_cutoff', '2024-12-31')
            st.metric("Training Cutoff", training_cutoff)
        
        # 추가 정보
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📅 Analysis Date: {regime_info['date'].strftime('%Y-%m-%d')}")
        with col2:
            oos_status = "Out-of-Sample" if regime_info.get('is_out_of_sample', False) else "In-Sample"
            st.info(f"🔮 Status: {oos_status}")
        
        # 특징값 표시
        features = regime_info.get('features', {})
        if features:
            st.markdown("**🔧 Unified Model Features:**")
            feature_cols = st.columns(len(features))
            for i, (key, value) in enumerate(features.items()):
                with feature_cols[i]:
                    st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
    
    def _display_current_signals(self):
        """현재 투자 신호 표시"""
        st.subheader("Current Investment Signals")
        
        preset = st.session_state.selected_preset
        
        if st.button("🎯 Analyze Investment Signals (Unified Model)"):
            with st.spinner('Analyzing components with unified model...'):
                try:
                    # 통합 모델로 체제 분석
                    jump_model = UniversalJumpModel(
                        benchmark_ticker=preset['benchmark'],
                        benchmark_name=preset['name'],
                        jump_penalty=st.session_state.jump_penalty,
                        training_cutoff_date=datetime(2024, 12, 31)
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
            name=preset['name']
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
        
        # 투자 권고 메시지 (통합 모델 정보 포함)
        if current_regime and current_regime['regime'] == 'BULL':
            feature_type = current_regime.get('feature_type', 'Unknown')
            st.success(f"🟢 **BULL Market** - Standard investment execution")
            st.info(f"🔧 Unified Model: {feature_type} | Jump Penalty: {st.session_state.jump_penalty}")
        
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
                title='Top Components by RS-Ratio (Unified Model)',
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
                title='Top Components by RS-Momentum (Unified Model)',
                color='rs_momentum',
                color_continuous_scale='RdYlGn'
            )
            fig_momentum.add_hline(y=100, line_dash="dash", line_color="black")
            fig_momentum.update_xaxes(tickangle=45)
            st.plotly_chart(fig_momentum, use_container_width=True)
    
    def _display_all_market_regimes(self):
        """모든 시장 체제 현황 표시"""
        st.subheader("🌍 All Market Regimes Overview (Unified Model)")
        
        # 분석 모드 표시
        if st.session_state.regime_analysis_mode == 'selected':
            if st.session_state.selected_tickers_for_regime:
                st.markdown(f"**Selected Tickers Analysis** ({len(st.session_state.selected_tickers_for_regime)} tickers)")
            else:
                st.warning("No tickers selected. Please select tickers from the sidebar.")
                return
        else:
            st.markdown("**Full Market Analysis** (All ETFs across all strategies)")
        
        # 통합 모델 정보 표시
        st.markdown(f"""
        <div class="unified-model">
            <strong>🔧 Analysis with Unified Jump Model</strong><br>
            📊 Feature Type: {'논문 정확한 3특징' if st.session_state.use_paper_features_only else '논문 기반 + 추가 특징'}<br>
            🎯 Jump Penalty: {st.session_state.jump_penalty}
        </div>
        """, unsafe_allow_html=True)
        
        button_text = "🔄 Analyze Selected Tickers (Unified)" if st.session_state.regime_analysis_mode == 'selected' else "🔄 Analyze All Regimes (Unified)"
        
        if st.button(button_text, type="primary"):
            with st.spinner("Analyzing market regimes with unified model..."):
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
        """선택된 티커들의 체제 분석 - 통합 모델"""
        results = {}
        
        for ticker in st.session_state.selected_tickers_for_regime:
            try:
                # 티커 정보 찾기
                ticker_name = self._find_ticker_name(ticker)
                
                result = self._analyze_single_ticker_unified(ticker, ticker_name)
                results[ticker] = result
                
                # 실시간 결과 표시
                if result['status'] == 'success':
                    st.success(f"✅ {ticker}: {result['regime']} (Confidence: {result['confidence']:.1%}) [Unified]")
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
        """모든 ETF 체제 분석 - 통합 모델"""
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
            st.info("📋 Using cached results (Unified Model)")
            return st.session_state.regime_cache
        
        results = {}
        
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_items = len(all_etfs) + len(benchmarks)
        processed = 0
        
        # 벤치마크 분석
        for ticker, name in benchmarks.items():
            status_text.text(f"Analyzing benchmark with unified model: {ticker}")
            
            result = self._analyze_single_ticker_unified(ticker, name)
            results[ticker] = result
            results[ticker]['type'] = 'benchmark'
            
            processed += 1
            progress_bar.progress(processed / total_items)
        
        # ETF 분석
        for ticker, info in all_etfs.items():
            status_text.text(f"Analyzing ETF with unified model: {ticker}")
            
            result = self._analyze_single_ticker_unified(ticker, info['name'])
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
    
    def _analyze_single_ticker_unified(self, ticker, name):
        """단일 티커 분석 - 통합 모델 사용 (utils 기반 효율화)"""
        try:
            # 1단계: 데이터 사전 검증
            try:
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period="5y", timeout=30)
                
                if hist.empty:
                    return {
                        'ticker': ticker, 'name': name, 'regime': 'NO_HISTORICAL_DATA',
                        'confidence': 0.0, 'status': 'data_unavailable'
                    }
                
                # 데이터 품질 검사 (통합 모델 기준)
                hist_clean = hist.dropna()
                if not validate_data(hist_clean, 300):  # utils 함수 사용
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
            
            # 2단계: 통합 JumpModel 초기화
            try:
                jump_model = UniversalJumpModel(
                    benchmark_ticker=ticker,
                    benchmark_name=name,
                    jump_penalty=st.session_state.jump_penalty,
                    training_cutoff_date=datetime(2024, 12, 31)
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
            
            # 4단계: 결과 처리 (utils 함수 사용)
            try:
                # 안전한 값 추출
                regime = current_regime.get('regime', 'UNKNOWN')
                confidence = safe_float(current_regime.get('confidence', 0.0))
                
                # 신뢰도 범위 검증
                confidence = max(0.0, min(1.0, confidence))
                
                result = {
                    'ticker': ticker,
                    'name': name,
                    'regime': regime,
                    'confidence': confidence,
                    'is_out_of_sample': current_regime.get('is_out_of_sample', False),
                    'analysis_date': current_regime.get('date', datetime.now()).strftime('%Y-%m-%d') if hasattr(current_regime.get('date'), 'strftime') else str(current_regime.get('date', 'Unknown')),
                    'unified_model_used': True,
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
        """체제 분석 결과 표시 - 통합 모델 버전"""
        # 통계 요약
        bull_count = sum(1 for r in results.values() if r['regime'] == 'BULL')
        bear_count = sum(1 for r in results.values() if r['regime'] == 'BEAR')
        unknown_count = len(results) - bull_count - bear_count
        
        oos_count = sum(1 for r in results.values() if r.get('is_out_of_sample', False))
        unified_model_count = sum(1 for r in results.values() if r.get('unified_model_used', False))
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Assets", len(results))
        with col2:
            st.metric("🟢 BULL", bull_count)
        with col3:
            st.metric("🔴 BEAR", bear_count)
        with col4:
            st.metric("⚠️ Issues", unknown_count)
        with col5:
            st.metric("🔧 Unified Model", unified_model_count)
        
        # 추가 메트릭
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔮 Out-of-Sample", oos_count)
        with col2:
            success_rate = (bull_count + bear_count) / len(results) * 100 if len(results) > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # 차트들
        col1, col2 = st.columns(2)
        
        with col1:
            # 체제 분포
            if bull_count + bear_count > 0:
                fig_regime = go.Figure(data=[go.Pie(
                    labels=['BULL', 'BEAR'],
                    values=[bull_count, bear_count],
                    marker_colors=['#28a745', '#dc3545']
                )])
                fig_regime.update_layout(title="Regime Distribution (Unified Model)")
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
        
        # 성공률 정보
        success_rate = (bull_count + bear_count) / len(results) * 100
        st.success(f"✅ Analysis completed with Unified Model! Success rate: {success_rate:.1f}%")
        
        # 통합 모델 상태 정보
        st.markdown(f"""
        <div class="unified-model">
            <strong>🔧 Unified Model Analysis Summary</strong><br>
            📊 Assets analyzed with unified model: {unified_model_count}/{len(results)}<br>
            🎯 Jump Penalty used: {st.session_state.jump_penalty}
        </div>
        """, unsafe_allow_html=True)
    
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
            st.metric("Training Cutoff", data.get('training_cutoff', 'N/A'))
        with col3:
            unified_status = "✅ Yes" if data.get('unified_model_used', False) else "❌ No"
            st.metric("Unified Model", unified_status)
        
        # 성과 지표 (utils 기반)
        metrics = data.get('metrics', {})
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", metrics.get('총 수익률', 'N/A'))
            with col2:
                st.metric("Annual Return", metrics.get('연율화 수익률', 'N/A'))
            with col3:
                st.metric("Sharpe Ratio", metrics.get('샤프 비율', 'N/A'))
            with col4:
                st.metric("Volatility", metrics.get('연율화 변동성', 'N/A'))
        
        # 포트폴리오 차트
        portfolio_df = data.get('portfolio')
        if safe_data_check(portfolio_df):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['value'],
                mode='lines',
                name='Portfolio Value (Unified Model)',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Portfolio Value Over Time (Unified Model)",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _update_data(self):
        """데이터 업데이트"""
        try:
            st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("✅ Data updated successfully! (Unified Model)")
        except Exception as e:
            st.error(f"Update failed: {str(e)}")
    
    def _run_backtest(self):
        """백테스트 실행 - 통합 모델"""
        preset = st.session_state.selected_preset
        
        with st.spinner('Running backtest with unified model...'):
            try:
                strategy = UniversalRSWithJumpModel(
                    preset_config=preset,
                    rs_length=20,
                    rs_timeframe='daily',
                    rs_recent_cross_days=30,
                    use_jump_model=True,
                    use_paper_features_only=st.session_state.use_paper_features_only,
                    jump_penalty=st.session_state.jump_penalty,
                    rf_ticker=st.session_state.rf_ticker,
                    default_rf_rate=st.session_state.default_rf_rate,
                    training_cutoff_date=datetime(2024, 12, 31)
                )
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365*3)  # 3년
                
                portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)
                
                if safe_data_check(portfolio_df):
                    st.session_state.portfolio_data = {
                        'portfolio': portfolio_df,
                        'trades': trades_df,
                        'regime': regime_df,
                        'metrics': strategy.calculate_performance_metrics(portfolio_df),
                        'use_jump_model': True,
                        'training_cutoff': '2024-12-31',
                        'unified_model_used': True,
                        'feature_type': '논문 정확한 3특징' if st.session_state.use_paper_features_only else '논문 기반 + 추가',
                        'jump_penalty': st.session_state.jump_penalty
                    }
                    st.success("✅ Backtest completed with unified model!")
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
        st.success("✅ Regime cache cleared! (Unified Model)")
    
    def _download_results(self):
        """결과 다운로드"""
        try:
            if safe_data_check(st.session_state.portfolio_data):
                portfolio_df = st.session_state.portfolio_data['portfolio']
                if safe_data_check(portfolio_df):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv = portfolio_df.to_csv()
                    st.download_button(
                        label="📥 Download Portfolio Data (Unified Model)",
                        data=csv,
                        file_name=f"portfolio_unified_{timestamp}.csv",
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
        st.success("✅ Cache cleared! (Unified Model)")

def main():
    """메인 함수"""
    try:
        dashboard = UnifiedRealtimeDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"🚨 Dashboard initialization failed: {str(e)}")
        st.info("💡 Try refreshing the page")

if __name__ == "__main__":
    main()
