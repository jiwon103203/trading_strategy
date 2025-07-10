"""
실시간 모니터링 대시보드 - 전체 ETF 버전 + 동적 Risk-Free Rate 지원
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
from preset_manager import PresetManager
from universal_rs_strategy import UniversalRSStrategy
from universal_jump_model import UniversalJumpModel
from universal_rs_with_jump import UniversalRSWithJumpModel
import concurrent.futures
from threading import Lock

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
    """실시간 모니터링 대시보드 - 전체 ETF 버전 + 동적 Risk-Free Rate"""
    
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
        if 'selected_tickers_for_regime' not in st.session_state:
            st.session_state.selected_tickers_for_regime = []
        if 'regime_analysis_mode' not in st.session_state:
            st.session_state.regime_analysis_mode = 'all'
        if 'rf_ticker' not in st.session_state:
            st.session_state.rf_ticker = '^IRX'
        if 'default_rf_rate' not in st.session_state:
            st.session_state.default_rf_rate = 0.02
        
        # 전체 프리셋 목록 (한국 시장 확장 버전)
        self.presets = {
            'S&P 500 Sectors': PresetManager.get_sp500_sectors(),
            'KOSPI 200 Sectors (Large Cap)': PresetManager.get_kospi_sectors(),
            'KOSPI Full Market Sectors': PresetManager.get_kospi_full_sectors(),
            'KOSDAQ Sectors': PresetManager.get_kosdaq_sectors(),
            'Korea Comprehensive Market': PresetManager.get_korea_comprehensive(),
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
        st.title("🚀 Universal RS Strategy Dashboard - Dynamic Risk-Free Rate Edition")
        st.markdown("### Real-time Market Monitoring & Signal Generation (All ETFs + Dynamic RF)")
        
        # Risk-Free Rate 상태 표시
        rf_status = "📊 동적" if HAS_RF_UTILS else "📌 고정"
        st.markdown(f"**🏦 Risk-Free Rate**: {st.session_state.rf_ticker} ({rf_status}) | **🎯 Training**: 2005-2024 | **🔮 Inference**: 2025")
        
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
        
        # Risk-Free Rate 설정
        st.sidebar.subheader("🏦 Risk-Free Rate Settings")
        
        if HAS_RF_UTILS:
            # RF 티커 선택
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
            
            # RF 상태 표시
            if st.sidebar.button("🔍 RF 데이터 테스트"):
                with st.sidebar.spinner("RF 데이터 확인 중..."):
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
                        else:
                            st.sidebar.error("❌ 데이터 없음")
                    except Exception as e:
                        st.sidebar.error(f"❌ 오류: {e}")
        else:
            # 고정 RF 설정
            default_rf_pct = st.sidebar.number_input(
                "Default RF Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.default_rf_rate * 100,
                step=0.1
            )
            st.session_state.default_rf_rate = default_rf_pct / 100
            st.sidebar.info(f"고정 RF: {default_rf_pct:.1f}%")
        
        # 프리셋 선택
        st.sidebar.subheader("Strategy Selection")
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
        
        # Regime 분석 모드 선택
        st.sidebar.markdown("#### 🌍 Regime Analysis Settings")
        regime_mode = st.sidebar.radio(
            "Analysis Mode",
            ["All Markets", "Selected Tickers"],
            index=0 if st.session_state.regime_analysis_mode == 'all' else 1
        )
        st.session_state.regime_analysis_mode = 'all' if regime_mode == "All Markets" else 'selected'
        
        # 선택적 분석 모드일 때 티커 선택
        if st.session_state.regime_analysis_mode == 'selected':
            # 모든 사용 가능한 티커 수집
            all_tickers = {}
            for strategy_name, preset in self.presets.items():
                # 벤치마크 추가
                benchmark = preset['benchmark']
                all_tickers[benchmark] = f"{strategy_name} Benchmark"
                
                # 구성요소들 추가
                for ticker, name in preset['components'].items():
                    if ticker not in all_tickers:
                        all_tickers[ticker] = name
            
            # 티커 선택 위젯
            selected_tickers = st.sidebar.multiselect(
                "Select Tickers to Analyze",
                options=list(all_tickers.keys()),
                default=st.session_state.selected_tickers_for_regime,
                format_func=lambda x: f"{x} - {all_tickers[x]}"
            )
            st.session_state.selected_tickers_for_regime = selected_tickers
            
            if selected_tickers:
                st.sidebar.info(f"Selected {len(selected_tickers)} ticker(s)")
            else:
                st.sidebar.warning("Please select at least one ticker")
        
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
            self.display_rf_info()
        
        # 탭 생성 (동적 RF 분석 탭 추가)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Market Status", "🎯 Current Signals", "🌍 All Market Regimes", "📊 Backtest Results", "🏦 RF Analysis"
        ])
        
        with tab1:
            self.display_market_status()
        
        with tab2:
            self.display_current_signals()
        
        with tab3:
            self.display_all_market_regimes()
        
        with tab4:
            self.display_backtest_results()
        
        with tab5:
            self.display_rf_analysis()
    
    def display_rf_info(self):
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
            else:
                st.markdown(f"""
                <div class="rf-info">
                    <strong>⚠️ Risk-Free Rate Data Unavailable</strong><br>
                    Using default rate: {st.session_state.default_rf_rate*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class="rf-info">
                <strong>❌ Risk-Free Rate Error</strong><br>
                {str(e)[:100]}...
            </div>
            """, unsafe_allow_html=True)
    
    def display_market_status(self):
        """시장 상태 표시 (동적 RF 지원)"""
        st.subheader("Market Regime Analysis (Dynamic Risk-Free Rate)")
        st.markdown("**Training Period**: 2005-2024 | **Inference Period**: 2025 (Out-of-Sample)")
        
        preset = st.session_state.selected_preset
        
        if st.button("🔍 Analyze Market Regime"):
            with st.spinner("Analyzing market regime with dynamic RF..."):
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
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            regime_emoji = "🟢" if current_regime['regime'] == 'BULL' else "🔴"
                            oos_indicator = "🔮" if current_regime.get('is_out_of_sample', False) else "📚"
                            st.metric("Current Regime", f"{regime_emoji} {current_regime['regime']} {oos_indicator}")
                        
                        with col2:
                            confidence = safe_get_value(current_regime['confidence'], 0.5)
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        with col3:
                            current_rf = current_regime.get('current_rf_rate', st.session_state.default_rf_rate * 100)
                            st.metric("Current RF", f"{current_rf:.3f}%")
                        
                        with col4:
                            rf_status = "📊 Dynamic" if current_regime.get('dynamic_rf_used', False) else "📌 Fixed"
                            st.metric("RF Type", rf_status)
                        
                        with col5:
                            features = current_regime.get('features', {})
                            risk_adj = safe_get_value(features.get('risk_adjusted_return', 0), 0)
                            st.metric("Risk-Adj Return", f"{risk_adj:.3f}")
                        
                        # 추가 정보 표시
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.info(f"📅 Analysis Date: {current_regime['date'].strftime('%Y-%m-%d')}")
                        with col2:
                            oos_status = "Out-of-Sample Prediction" if current_regime.get('is_out_of_sample', False) else "In-Sample Analysis"
                            st.info(f"🔮 Status: {oos_status}")
                        with col3:
                            rf_ticker = current_regime.get('rf_ticker', st.session_state.rf_ticker)
                            st.info(f"🏦 RF Ticker: {rf_ticker}")
                        
                        # RF 수준별 추가 분석
                        if current_regime.get('dynamic_rf_used', False):
                            rf_level = current_regime.get('current_rf_rate', 0)
                            if rf_level > 4.0:
                                st.warning(f"🔶 높은 금리 환경 ({rf_level:.2f}%) - 보수적 투자 권고")
                            elif rf_level < 1.0:
                                st.success(f"🔷 낮은 금리 환경 ({rf_level:.2f}%) - 적극적 투자 기회")
                            else:
                                st.info(f"🔸 보통 금리 환경 ({rf_level:.2f}%) - 표준 투자 환경")
                        
                        st.success("✅ Market regime analysis completed with dynamic RF!")
                        st.caption(f"🔮 = Out-of-Sample (2025 data) | 📚 = In-Sample (≤2024 data) | 📊 = Dynamic RF")
                    else:
                        st.error("❌ Unable to analyze market regime")
                        
                except Exception as e:
                    st.error(f"Market regime analysis failed: {str(e)}")
                    st.info("💡 Check your internet connection or try again later")
    
    def display_current_signals(self):
        """현재 투자 신호 표시 (동적 RF 지원)"""
        st.subheader("Current Investment Signals (Dynamic Risk-Free Rate)")
        st.markdown("**Model Training**: 2005-2024 | **Current Analysis**: Out-of-Sample Prediction")
        
        preset = st.session_state.selected_preset
        
        if st.button("🎯 Analyze Investment Signals"):
            with st.spinner('Analyzing components with dynamic RF...'):
                try:
                    # 먼저 시장 체제 확인 (2024년까지 학습, 동적 RF 사용)
                    jump_model = UniversalJumpModel(
                        benchmark_ticker=preset['benchmark'],
                        benchmark_name=preset['name'],
                        training_cutoff_date=datetime(2024, 12, 31),
                        rf_ticker=st.session_state.rf_ticker,
                        default_rf_rate=st.session_state.default_rf_rate
                    )
                    
                    current_regime = jump_model.get_current_regime_with_training_cutoff()
                    
                    if current_regime:
                        # 체제 정보 표시
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            regime_emoji = "🟢" if current_regime['regime'] == 'BULL' else "🔴"
                            oos_indicator = "🔮" if current_regime.get('is_out_of_sample', False) else "📚"
                            st.metric("Market Regime", f"{regime_emoji} {current_regime['regime']} {oos_indicator}")
                        with col2:
                            confidence = safe_get_value(current_regime['confidence'], 0.5)
                            st.metric("Confidence", f"{confidence:.1%}")
                        with col3:
                            current_rf = current_regime.get('current_rf_rate', st.session_state.default_rf_rate * 100)
                            st.metric("Current RF", f"{current_rf:.3f}%")
                        with col4:
                            analysis_status = "Out-of-Sample" if current_regime.get('is_out_of_sample', False) else "In-Sample"
                            st.metric("Prediction Type", analysis_status)
                        
                        # Dynamic RF 상태 표시
                        if current_regime.get('dynamic_rf_used', False):
                            st.markdown(f"""
                            <div class="dynamic-rf">
                                📊 Dynamic Risk-Free Rate Active: {current_regime.get('rf_ticker', st.session_state.rf_ticker)} 
                                (Current: {current_rf:.3f}%, 30-day Avg: {current_regime.get('avg_rf_rate_30d', current_rf):.3f}%)
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # BEAR 체제인 경우 투자 중단 권고
                        if current_regime['regime'] == 'BEAR':
                            st.error("🔴 **BEAR Market Detected** - Investment suspension recommended")
                            st.markdown("The model suggests avoiding new investments in current market conditions.")
                            if current_regime.get('dynamic_rf_used', False):
                                st.info(f"💰 Consider cash position with RF return: {current_rf:.3f}%")
                            return
                    
                    # RS 전략 분석 (BULL 체제이거나 체제 분석이 불가능한 경우, 동적 RF 사용)
                    strategy = UniversalRSStrategy(
                        benchmark=preset['benchmark'],
                        components=preset['components'],
                        name=preset['name'],
                        rf_ticker=st.session_state.rf_ticker,
                        default_rf_rate=st.session_state.default_rf_rate
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
                            
                            # 투자 권고 메시지 (RF 수준 고려)
                            if current_regime and current_regime['regime'] == 'BULL':
                                current_rf = current_regime.get('current_rf_rate', 0)
                                
                                if current_rf > 4.0:
                                    st.warning(f"🟡 **BULL Market + High RF ({current_rf:.2f}%)** - Conservative investment recommended")
                                    st.info(f"💡 Consider smaller position sizes due to high opportunity cost")
                                elif current_rf < 1.0:
                                    st.success(f"🟢 **BULL Market + Low RF ({current_rf:.2f}%)** - Aggressive investment opportunity!")
                                    st.info(f"💡 Low opportunity cost supports higher risk-taking")
                                else:
                                    st.success(f"🟢 **BULL Market + Normal RF ({current_rf:.2f}%)** - Standard investment execution")
                                    
                                st.info(f"📊 {len(selected)} Strong Components identified with dynamic RF-adjusted analysis")
                            else:
                                st.info(f"📊 **{len(selected)} Components** meet RS criteria (Market regime analysis unavailable)")
                            
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
                                    title='Top Components by RS-Ratio (Dynamic RF Adjusted)',
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
                                    title='Top Components by RS-Momentum (Dynamic RF Adjusted)',
                                    color='rs_momentum',
                                    color_continuous_scale='RdYlGn'
                                )
                                fig_momentum.add_hline(y=100, line_dash="dash", line_color="black")
                                fig_momentum.update_xaxes(tickangle=45)
                                st.plotly_chart(fig_momentum, use_container_width=True)
                            
                            # 동적 RF 기반 추가 분석
                            if HAS_RF_UTILS and current_regime and current_regime.get('dynamic_rf_used', False):
                                st.subheader("🏦 Risk-Free Rate Impact Analysis")
                                
                                current_rf = current_regime.get('current_rf_rate', 0)
                                
                                # RF 변화가 투자 매력도에 미치는 영향
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Current RF Impact", 
                                            "High Cost" if current_rf > 4.0 else "Low Cost" if current_rf < 1.0 else "Normal Cost")
                                
                                with col2:
                                    # 예상 초과수익률 필요치
                                    required_excess = current_rf + 3.0  # RF + 3% 위험프리미엄
                                    st.metric("Required Excess Return", f"{required_excess:.1f}%")
                                
                                with col3:
                                    # 현금 대비 매력도
                                    cash_vs_equity = "Equity Favorable" if current_rf < 2.0 else "Cash Competitive" if current_rf > 4.0 else "Balanced"
                                    st.metric("Cash vs Equity", cash_vs_equity)
                        else:
                            st.warning("⚠️ No components currently meet the investment criteria")
                            if current_regime and current_regime['regime'] == 'BULL':
                                current_rf = current_regime.get('current_rf_rate', 0)
                                if current_rf > 4.0:
                                    st.info(f"💡 High RF environment ({current_rf:.2f}%) may be contributing to lack of attractive opportunities")
                                else:
                                    st.info("Even in BULL market, no strong RS signals detected. Consider waiting for better opportunities.")
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
        """단일 ETF의 시장 체제 분석 - 2024년까지 학습, 동적 RF 지원"""
        try:
            jump_model = UniversalJumpModel(
                benchmark_ticker=ticker,
                benchmark_name=name,
                jump_penalty=50.0,
                training_cutoff_date=datetime(2024, 12, 31),
                rf_ticker=st.session_state.rf_ticker,
                default_rf_rate=st.session_state.default_rf_rate
            )
            
            current_regime = jump_model.get_current_regime_with_training_cutoff()
            
            if current_regime:
                return {
                    'ticker': ticker,
                    'name': name,
                    'regime': current_regime['regime'],
                    'confidence': current_regime['confidence'],
                    'is_out_of_sample': current_regime.get('is_out_of_sample', False),
                    'analysis_date': current_regime['date'].strftime('%Y-%m-%d'),
                    'rf_ticker': current_regime.get('rf_ticker', st.session_state.rf_ticker),
                    'current_rf_rate': current_regime.get('current_rf_rate', st.session_state.default_rf_rate * 100),
                    'dynamic_rf_used': current_regime.get('dynamic_rf_used', False),
                    'status': 'success'
                }
            else:
                return {
                    'ticker': ticker,
                    'name': name,
                    'regime': 'UNKNOWN',
                    'confidence': 0.0,
                    'is_out_of_sample': False,
                    'analysis_date': 'N/A',
                    'rf_ticker': st.session_state.rf_ticker,
                    'current_rf_rate': st.session_state.default_rf_rate * 100,
                    'dynamic_rf_used': False,
                    'status': 'no_data'
                }
        except Exception as e:
            return {
                'ticker': ticker,
                'name': name,
                'regime': 'ERROR',
                'confidence': 0.0,
                'is_out_of_sample': False,
                'analysis_date': 'N/A',
                'rf_ticker': st.session_state.rf_ticker,
                'current_rf_rate': st.session_state.default_rf_rate * 100,
                'dynamic_rf_used': False,
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_all_etf_regimes(self, selected_tickers_only=None):
        """모든 ETF의 시장 체제 병렬 분석 (선택적 분석 지원, 동적 RF)"""
        # 분석할 ETF 결정
        if selected_tickers_only:
            # 선택된 티커들만 분석
            all_etfs = {}
            benchmarks = {}
            
            # 모든 프리셋에서 선택된 티커 정보 수집
            for ticker in selected_tickers_only:
                found = False
                
                # 벤치마크인지 확인
                for strategy_name, preset in self.presets.items():
                    if preset['benchmark'] == ticker:
                        benchmarks[ticker] = f"{strategy_name} Benchmark"
                        found = True
                        break
                
                # 구성요소인지 확인
                if not found:
                    for strategy_name, preset in self.presets.items():
                        if ticker in preset['components']:
                            if ticker not in all_etfs:
                                all_etfs[ticker] = {
                                    'name': preset['components'][ticker],
                                    'strategies': [strategy_name]
                                }
                            else:
                                all_etfs[ticker]['strategies'].append(strategy_name)
                            found = True
                
                # 찾지 못한 경우 기본값으로 추가
                if not found:
                    all_etfs[ticker] = {
                        'name': ticker,
                        'strategies': ['Custom']
                    }
        else:
            # 모든 ETF 분석 (기존 로직)
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
        
        # 캐시 확인 (선택적 분석일 경우 캐시 사용 안 함)
        now = datetime.now()
        if (not selected_tickers_only and
            st.session_state.cache_timestamp and 
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
        status_text.text("Analyzing benchmarks with dynamic RF...")
        for ticker, name in benchmarks.items():
            result = self.analyze_single_etf_regime(ticker, name)
            results[ticker] = result
            results[ticker]['type'] = 'benchmark'
            
            processed += 1
            progress_bar.progress(processed / total_items)
        
        # ETF 분석 (배치로 처리)
        status_text.text("Analyzing ETFs with dynamic RF...")
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
                            'strategies': info['strategies'],
                            'rf_ticker': st.session_state.rf_ticker,
                            'current_rf_rate': st.session_state.default_rf_rate * 100,
                            'dynamic_rf_used': False
                        }
                    
                    processed += 1
                    progress_bar.progress(processed / total_items)
        
        # 캐시 업데이트 (전체 분석일 경우에만)
        if not selected_tickers_only:
            st.session_state.regime_cache = results
            st.session_state.cache_timestamp = now
        
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def display_all_market_regimes(self):
        """모든 시장 체제 현황 표시 (동적 RF 정보 포함)"""
        st.subheader("🌍 All Market Regimes Overview (Dynamic Risk-Free Rate)")
        
        # 분석 모드 표시
        if st.session_state.regime_analysis_mode == 'selected':
            if st.session_state.selected_tickers_for_regime:
                st.markdown(f"**Selected Tickers Analysis** ({len(st.session_state.selected_tickers_for_regime)} tickers)")
                with st.expander("Selected Tickers", expanded=False):
                    for ticker in st.session_state.selected_tickers_for_regime:
                        st.write(f"• {ticker}")
            else:
                st.warning("No tickers selected. Please select tickers from the sidebar.")
                return
        else:
            st.markdown("**Full Market Analysis** (All ETFs across all strategies)")
        
        st.markdown("Current Bull/Bear status with dynamic Risk-Free Rate analysis")
        st.markdown(f"**🏦 RF Ticker**: {st.session_state.rf_ticker} | **🎯 Training**: 2005-2024 | **🔮 Inference**: 2025")
        
        button_text = "🔄 Analyze Selected Tickers" if st.session_state.regime_analysis_mode == 'selected' else "🔄 Analyze All Market Regimes"
        
        if st.button(button_text, type="primary"):
            with st.spinner("Analyzing market regimes with dynamic RF... This may take a few minutes"):
                # 선택 모드에 따라 분석 실행
                if st.session_state.regime_analysis_mode == 'selected':
                    if not st.session_state.selected_tickers_for_regime:
                        st.error("Please select at least one ticker from the sidebar")
                        return
                    results = self.analyze_all_etf_regimes(selected_tickers_only=st.session_state.selected_tickers_for_regime)
                else:
                    results = self.analyze_all_etf_regimes()
                
                if results:
                    # 통계 요약
                    bull_count = sum(1 for r in results.values() if r['regime'] == 'BULL')
                    bear_count = sum(1 for r in results.values() if r['regime'] == 'BEAR')
                    unknown_count = sum(1 for r in results.values() if r['regime'] in ['UNKNOWN', 'ERROR', 'TIMEOUT'])
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
                        st.metric("⚠️ Unknown", unknown_count)
                    with col5:
                        st.metric("🔮 Out-of-Sample", oos_count)
                    with col6:
                        st.metric("📊 Dynamic RF", dynamic_rf_count)
                    
                    # RF 통계
                    if dynamic_rf_count > 0:
                        rf_rates = [r['current_rf_rate'] for r in results.values() if r.get('dynamic_rf_used', False)]
                        if rf_rates:
                            avg_rf = np.mean(rf_rates)
                            min_rf = np.min(rf_rates)
                            max_rf = np.max(rf_rates)
                            
                            st.markdown(f"""
                            <div class="rf-info">
                                <strong>📊 Dynamic Risk-Free Rate Statistics</strong><br>
                                Average: {avg_rf:.3f}% | Range: {min_rf:.3f}% - {max_rf:.3f}% | 
                                Assets using dynamic RF: {dynamic_rf_count}/{len(results)}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # 전략별 정리
                    st.subheader("📊 By Strategy (with RF Info)")
                    
                    # 선택 모드일 경우 관련 전략만 표시
                    strategies_to_show = []
                    if st.session_state.regime_analysis_mode == 'selected':
                        # 선택된 티커가 포함된 전략만 찾기
                        for strategy_name, preset in self.presets.items():
                            has_selected_ticker = False
                            
                            # 벤치마크 확인
                            if preset['benchmark'] in st.session_state.selected_tickers_for_regime:
                                has_selected_ticker = True
                            
                            # 구성요소 확인
                            if not has_selected_ticker:
                                for ticker in st.session_state.selected_tickers_for_regime:
                                    if ticker in preset['components']:
                                        has_selected_ticker = True
                                        break
                            
                            if has_selected_ticker:
                                strategies_to_show.append((strategy_name, preset))
                    else:
                        strategies_to_show = list(self.presets.items())
                    
                    # 전략별 표시
                    for strategy_name, preset in strategies_to_show:
                        # 이 전략에 포함된 선택 티커 수 계산
                        if st.session_state.regime_analysis_mode == 'selected':
                            selected_in_strategy = []
                            if preset['benchmark'] in st.session_state.selected_tickers_for_regime:
                                selected_in_strategy.append(preset['benchmark'])
                            for ticker in st.session_state.selected_tickers_for_regime:
                                if ticker in preset['components']:
                                    selected_in_strategy.append(ticker)
                            
                            if not selected_in_strategy:
                                continue
                            
                            display_title = f"{strategy_name} ({len(selected_in_strategy)} selected tickers)"
                        else:
                            display_title = f"{strategy_name} ({len(preset['components'])} ETFs)"
                        
                        with st.expander(display_title):
                            
                            # 벤치마크 상태
                            benchmark_ticker = preset['benchmark']
                            show_benchmark = True
                            if st.session_state.regime_analysis_mode == 'selected':
                                show_benchmark = benchmark_ticker in st.session_state.selected_tickers_for_regime
                            
                            if show_benchmark and benchmark_ticker in results:
                                benchmark_result = results[benchmark_ticker]
                                regime_class = f"{benchmark_result['regime'].lower()}-card" if benchmark_result['regime'] in ['BULL', 'BEAR'] else "unknown-card"
                                
                                oos_indicator = "🔮" if benchmark_result.get('is_out_of_sample', False) else "📚"
                                rf_indicator = "📊" if benchmark_result.get('dynamic_rf_used', False) else "📌"
                                confidence_text = f"(Confidence: {benchmark_result['confidence']:.1%})" if benchmark_result['confidence'] > 0 else ""
                                analysis_date = benchmark_result.get('analysis_date', 'N/A')
                                current_rf = benchmark_result.get('current_rf_rate', 0)
                                
                                st.markdown(f"""
                                <div class="regime-card {regime_class}">
                                    <div class="strategy-header">📊 Benchmark: {benchmark_result['name']} {oos_indicator} {rf_indicator}</div>
                                    <div><strong>Regime:</strong> {benchmark_result['regime']} {confidence_text}</div>
                                    <div><strong>Analysis Date:</strong> {analysis_date}</div>
                                    <div><strong>Risk-Free Rate:</strong> {current_rf:.3f}% ({benchmark_result.get('rf_ticker', 'N/A')})</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # ETF 목록
                            st.markdown("**Components:**")
                            
                            bull_etfs = []
                            bear_etfs = []
                            unknown_etfs = []
                            
                            # 표시할 구성요소 결정
                            components_to_show = preset['components'].items()
                            if st.session_state.regime_analysis_mode == 'selected':
                                components_to_show = [(t, n) for t, n in preset['components'].items() 
                                                    if t in st.session_state.selected_tickers_for_regime]
                            
                            for ticker, name in components_to_show:
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
                                        'confidence': 0.0,
                                        'is_out_of_sample': False,
                                        'analysis_date': 'N/A',
                                        'current_rf_rate': st.session_state.default_rf_rate * 100,
                                        'dynamic_rf_used': False
                                    })
                            
                            # BULL ETFs
                            if bull_etfs:
                                st.markdown("🟢 **BULL Regime:**")
                                for etf in bull_etfs:
                                    confidence_text = f" (Confidence: {etf['confidence']:.1%})" if etf['confidence'] > 0 else ""
                                    oos_indicator = " 🔮" if etf.get('is_out_of_sample', False) else " 📚"
                                    rf_indicator = " 📊" if etf.get('dynamic_rf_used', False) else " 📌"
                                    current_rf = etf.get('current_rf_rate', 0)
                                    
                                    st.markdown(f"""
                                    <div class="etf-item etf-bull">
                                        <span><strong>{etf['ticker']}</strong> - {etf['name']}{oos_indicator}{rf_indicator}</span>
                                        <span>{etf['regime']}{confidence_text} (RF: {current_rf:.3f}%)</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # BEAR ETFs
                            if bear_etfs:
                                st.markdown("🔴 **BEAR Regime:**")
                                for etf in bear_etfs:
                                    confidence_text = f" (Confidence: {etf['confidence']:.1%})" if etf['confidence'] > 0 else ""
                                    oos_indicator = " 🔮" if etf.get('is_out_of_sample', False) else " 📚"
                                    rf_indicator = " 📊" if etf.get('dynamic_rf_used', False) else " 📌"
                                    current_rf = etf.get('current_rf_rate', 0)
                                    
                                    st.markdown(f"""
                                    <div class="etf-item etf-bear">
                                        <span><strong>{etf['ticker']}</strong> - {etf['name']}{oos_indicator}{rf_indicator}</span>
                                        <span>{etf['regime']}{confidence_text} (RF: {current_rf:.3f}%)</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Unknown ETFs
                            if unknown_etfs:
                                st.markdown("⚠️ **Unknown/Error:**")
                                for etf in unknown_etfs:
                                    oos_indicator = " 🔮" if etf.get('is_out_of_sample', False) else ""
                                    current_rf = etf.get('current_rf_rate', 0)
                                    
                                    st.markdown(f"""
                                    <div class="etf-item etf-unknown">
                                        <span><strong>{etf['ticker']}</strong> - {etf['name']}{oos_indicator}</span>
                                        <span>{etf['regime']} (RF: {current_rf:.3f}%)</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # 종합 차트
                    st.subheader("📈 Regime Distribution (Dynamic RF)")
                    
                    # Out-of-Sample vs In-Sample 분석
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # 파이 차트
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=['BULL', 'BEAR', 'Unknown'],
                            values=[bull_count, bear_count, unknown_count],
                            marker_colors=['#28a745', '#dc3545', '#ffc107']
                        )])
                        fig_pie.update_layout(title="Overall Market Regime Distribution")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Out-of-Sample vs In-Sample 분포
                        in_sample_count = len(results) - oos_count
                        fig_oos = go.Figure(data=[go.Pie(
                            labels=['Out-of-Sample (2025)', 'In-Sample (≤2024)'],
                            values=[oos_count, in_sample_count],
                            marker_colors=['#17a2b8', '#6c757d']
                        )])
                        fig_oos.update_layout(title="Sample Distribution")
                        st.plotly_chart(fig_oos, use_container_width=True)
                    
                    with col3:
                        # Dynamic RF vs Fixed RF 분포
                        fixed_rf_count = len(results) - dynamic_rf_count
                        fig_rf = go.Figure(data=[go.Pie(
                            labels=['Dynamic RF', 'Fixed RF'],
                            values=[dynamic_rf_count, fixed_rf_count],
                            marker_colors=['#9c27b0', '#795548']
                        )])
                        fig_rf.update_layout(title="Risk-Free Rate Type")
                        st.plotly_chart(fig_rf, use_container_width=True)
                    
                    # RF 수준 분포 히스토그램 (동적 RF 사용 자산만)
                    if dynamic_rf_count > 0:
                        st.subheader("📊 Risk-Free Rate Distribution")
                        rf_rates = [r['current_rf_rate'] for r in results.values() if r.get('dynamic_rf_used', False)]
                        
                        fig_rf_hist = px.histogram(
                            x=rf_rates,
                            nbins=20,
                            title=f"Current Risk-Free Rate Distribution ({st.session_state.rf_ticker})",
                            labels={'x': 'Risk-Free Rate (%)', 'y': 'Count'}
                        )
                        fig_rf_hist.update_layout(showlegend=False)
                        st.plotly_chart(fig_rf_hist, use_container_width=True)
                    
                    # 범례 설명
                    st.markdown("---")
                    st.markdown("**Legend:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("🔮 **Out-of-Sample**: Model trained on 2005-2024, predicting 2025")
                        st.markdown("🟢 **BULL**: Favorable market conditions")
                        st.markdown("📊 **Dynamic RF**: Using real-time risk-free rate data")
                    with col2:
                        st.markdown("📚 **In-Sample**: Analysis using training period data")
                        st.markdown("🔴 **BEAR**: Unfavorable market conditions")
                        st.markdown("📌 **Fixed RF**: Using default risk-free rate")
                    
                    if st.session_state.regime_analysis_mode == 'selected':
                        st.success(f"✅ Analysis completed! {len(results)} selected assets analyzed. {oos_count} out-of-sample predictions. {dynamic_rf_count} dynamic RF.")
                    else:
                        st.success(f"✅ Analysis completed! {len(results)} assets analyzed. {oos_count} out-of-sample predictions. {dynamic_rf_count} dynamic RF.")
                else:
                    st.error("❌ Failed to analyze market regimes")
        else:
            # 캐시된 결과 표시
            if st.session_state.regime_cache and st.session_state.cache_timestamp and st.session_state.regime_analysis_mode == 'all':
                cache_age = datetime.now() - st.session_state.cache_timestamp
                st.info(f"📋 Cached results available (Updated {cache_age.seconds//60} minutes ago). Click 'Analyze' to refresh.")
            elif st.session_state.regime_analysis_mode == 'selected':
                st.info("📋 Click 'Analyze Selected Tickers' to analyze your selected tickers with dynamic RF.")
            else:
                st.info("📋 Click 'Analyze All Market Regimes' to start analysis with dynamic RF.")
    
    def display_rf_analysis(self):
        """Risk-Free Rate 전용 분석 탭"""
        st.subheader("🏦 Risk-Free Rate Analysis")
        
        if not HAS_RF_UTILS:
            st.warning("⚠️ Dynamic Risk-Free Rate analysis requires risk_free_rate_utils.py")
            st.info(f"Currently using fixed rate: {st.session_state.default_rf_rate*100:.1f}%")
            return
        
        st.markdown(f"**Current RF Ticker**: {st.session_state.rf_ticker}")
        
        # RF 데이터 분석 기간 선택
        col1, col2 = st.columns(2)
        with col1:
            period_options = {
                30: "1 Month",
                90: "3 Months", 
                180: "6 Months",
                365: "1 Year",
                730: "2 Years"
            }
            selected_days = st.selectbox("Analysis Period", options=list(period_options.keys()), 
                                       format_func=lambda x: period_options[x], index=3)
        
        with col2:
            if st.button("📊 Analyze RF Data", type="primary"):
                with st.spinner("Analyzing Risk-Free Rate data..."):
                    try:
                        rf_manager = RiskFreeRateManager(st.session_state.rf_ticker, st.session_state.default_rf_rate)
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=selected_days)
                        
                        rf_data = rf_manager.download_risk_free_rate(start_date, end_date)
                        
                        if rf_data is not None and not rf_data.empty:
                            # RF 통계
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
                            
                            # RF 추이 차트
                            fig_rf = go.Figure()
                            fig_rf.add_trace(go.Scatter(
                                x=rf_data.index,
                                y=rf_data.values * 100,
                                mode='lines',
                                name=f'Risk-Free Rate ({st.session_state.rf_ticker})',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # 평균선 추가
                            fig_rf.add_hline(y=stats['mean_rate'], line_dash="dash", 
                                           line_color="red", annotation_text=f"Average: {stats['mean_rate']:.3f}%")
                            
                            fig_rf.update_layout(
                                title=f"Risk-Free Rate Trend ({st.session_state.rf_ticker})",
                                xaxis_title="Date",
                                yaxis_title="Rate (%)",
                                height=400
                            )
                            
                            st.plotly_chart(fig_rf, use_container_width=True)
                            
                            # RF 분포 히스토그램
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_hist = px.histogram(
                                    x=rf_data.values * 100,
                                    nbins=30,
                                    title="Rate Distribution",
                                    labels={'x': 'Rate (%)', 'y': 'Frequency'}
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            with col2:
                                # 월별 평균 (충분한 데이터가 있을 때)
                                if selected_days >= 90:
                                    monthly_avg = rf_data.resample('M').mean() * 100
                                    fig_monthly = px.bar(
                                        x=monthly_avg.index,
                                        y=monthly_avg.values,
                                        title="Monthly Average Rates",
                                        labels={'x': 'Month', 'y': 'Rate (%)'}
                                    )
                                    st.plotly_chart(fig_monthly, use_container_width=True)
                                else:
                                    st.info("월별 분석을 위해서는 더 긴 기간을 선택하세요.")
                            
                            # RF 레벨별 투자 시사점
                            current_rate = stats['end_rate']
                            st.subheader("💡 Investment Implications")
                            
                            if current_rate > 4.0:
                                st.error(f"🔴 **High RF Environment** ({current_rate:.2f}%)")
                                st.markdown("""
                                - **Cash becomes attractive**: High opportunity cost for risky assets
                                - **Bond competition**: Fixed-income alternatives more appealing  
                                - **Equity hurdle**: Stocks need higher expected returns to justify risk
                                - **Strategy**: Consider conservative positioning, high-quality assets
                                """)
                            elif current_rate < 1.0:
                                st.success(f"🟢 **Low RF Environment** ({current_rate:.2f}%)")
                                st.markdown("""
                                - **Risk asset friendly**: Low opportunity cost encourages risk-taking
                                - **Growth favorable**: Low discount rates benefit growth stocks
                                - **Leverage attractive**: Cheap borrowing costs support leveraged strategies
                                - **Strategy**: Consider aggressive positioning, growth assets
                                """)
                            else:
                                st.info(f"🟡 **Normal RF Environment** ({current_rate:.2f}%)")
                                st.markdown("""
                                - **Balanced outlook**: Moderate opportunity cost for risky assets
                                - **Neutral positioning**: Standard risk-return relationships apply
                                - **Selective approach**: Focus on individual asset merit
                                - **Strategy**: Maintain balanced portfolio allocation
                                """)
                            
                            # RF 변화 트렌드 분석
                            if len(rf_data) > 30:
                                recent_trend = rf_data.iloc[-30:].values
                                trend_slope = np.polyfit(range(len(recent_trend)), recent_trend, 1)[0] * 100 * 365
                                
                                st.subheader("📈 Recent Trend Analysis (30 days)")
                                if abs(trend_slope) < 0.1:
                                    st.info(f"📊 **Stable**: Rate trend is flat ({trend_slope:+.2f}% annually)")
                                elif trend_slope > 0.1:
                                    st.warning(f"📈 **Rising**: Rate trend is upward ({trend_slope:+.2f}% annually)")
                                    st.markdown("- Consider reducing duration risk in bonds")
                                    st.markdown("- Monitor for impact on growth stocks")
                                else:
                                    st.success(f"📉 **Falling**: Rate trend is downward ({trend_slope:+.2f}% annually)")
                                    st.markdown("- Favorable for risk assets")
                                    st.markdown("- Duration assets may benefit")
                        else:
                            st.error(f"❌ Failed to download RF data for {st.session_state.rf_ticker}")
                            
                    except Exception as e:
                        st.error(f"RF analysis failed: {str(e)}")
        
        # RF 시나리오 분석
        st.subheader("🔮 Risk-Free Rate Scenario Analysis")
        
        with st.expander("Impact on Sharpe Ratios", expanded=False):
            st.markdown("**How different RF levels affect Sharpe ratios for a 10% return, 15% volatility strategy:**")
            
            # 시나리오 데이터 생성
            scenarios = []
            base_return = 10.0
            volatility = 15.0
            
            for rf in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                sharpe = (base_return - rf) / volatility
                scenarios.append({'RF_Rate': rf, 'Sharpe_Ratio': sharpe})
            
            scenario_df = pd.DataFrame(scenarios)
            
            fig_scenario = px.line(
                scenario_df, 
                x='RF_Rate', 
                y='Sharpe_Ratio',
                title='Sharpe Ratio vs Risk-Free Rate',
                labels={'RF_Rate': 'Risk-Free Rate (%)', 'Sharpe_Ratio': 'Sharpe Ratio'}
            )
            
            # 현재 RF 위치 표시
            if HAS_RF_UTILS:
                try:
                    rf_manager = RiskFreeRateManager(st.session_state.rf_ticker, st.session_state.default_rf_rate)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)
                    rf_data = rf_manager.download_risk_free_rate(start_date, end_date)
                    
                    if rf_data is not None and not rf_data.empty:
                        current_rf = rf_data.iloc[-1] * 100
                        current_sharpe = (base_return - current_rf) / volatility
                        
                        fig_scenario.add_trace(go.Scatter(
                            x=[current_rf],
                            y=[current_sharpe],
                            mode='markers',
                            marker=dict(size=12, color='red'),
                            name=f'Current ({current_rf:.2f}%)'
                        ))
                except:
                    pass
            
            fig_scenario.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                                 annotation_text="Sharpe = 1.0")
            
            st.plotly_chart(fig_scenario, use_container_width=True)
    
    def refresh_all_regimes(self):
        """모든 체제 정보 새로고침"""
        try:
            st.session_state.regime_cache = {}
            st.session_state.cache_timestamp = None
            # 선택된 티커는 유지 (사용자가 원할 수 있으므로)
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
        """백테스트 실행 - 2024년까지 학습, 동적 RF 지원"""
        preset = st.session_state.selected_preset
        
        with st.spinner('Running backtest with dynamic RF... This may take a few minutes'):
            try:
                # Jump Model을 사용하는 경우 2024년까지만 학습하도록 설정
                if use_jump:
                    st.info("🎯 Jump Model will be trained on data up to 2024-12-31")
                
                st.info(f"🏦 Using Risk-Free Rate: {st.session_state.rf_ticker}")
                
                strategy = UniversalRSWithJumpModel(
                    preset_config=preset,
                    rs_length=rs_length,
                    rs_timeframe=timeframe,
                    rs_recent_cross_days=cross_days,
                    use_jump_model=use_jump,
                    rf_ticker=st.session_state.rf_ticker,
                    default_rf_rate=st.session_state.default_rf_rate,
                    training_cutoff_date=datetime(2024, 12, 31)
                )
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365*years)
                
                portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)
                
                if safe_data_check(portfolio_df):
                    st.session_state.portfolio_data = {
                        'portfolio': portfolio_df,
                        'trades': trades_df if safe_data_check(trades_df) else pd.DataFrame(),
                        'regime': regime_df if safe_data_check(regime_df) else pd.DataFrame(),
                        'metrics': strategy.calculate_performance_metrics(portfolio_df),
                        'use_jump_model': use_jump,
                        'training_cutoff': '2024-12-31' if use_jump else 'N/A',
                        'rf_ticker': st.session_state.rf_ticker,
                        'dynamic_rf_used': HAS_RF_UTILS
                    }
                    
                    success_msg = "✅ Backtest completed!"
                    if use_jump:
                        success_msg += " (Jump Model trained on 2005-2024 data)"
                    if HAS_RF_UTILS:
                        success_msg += f" with dynamic RF ({st.session_state.rf_ticker})"
                    st.success(success_msg)
                else:
                    st.error("❌ Backtest failed - no results generated")
                    
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                st.info("💡 Try reducing the backtest period or check your internet connection")
    
    def display_backtest_results(self):
        """백테스트 결과 표시 (동적 RF 정보 포함)"""
        st.subheader("Backtest Results (Dynamic Risk-Free Rate)")
        
        if not safe_data_check(st.session_state.portfolio_data):
            st.info("💡 Run a backtest to see results")
            return
        
        data = st.session_state.portfolio_data
        metrics = data.get('metrics', {})
        use_jump = data.get('use_jump_model', False)
        training_cutoff = data.get('training_cutoff', 'N/A')
        rf_ticker = data.get('rf_ticker', st.session_state.rf_ticker)
        dynamic_rf_used = data.get('dynamic_rf_used', False)
        
        # 백테스트 설정 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            jump_status = "Enabled" if use_jump else "Disabled"
            st.metric("Jump Model", jump_status)
        with col2:
            rf_status = "📊 Dynamic" if dynamic_rf_used else "📌 Fixed"
            st.metric("Risk-Free Rate", f"{rf_status}")
        with col3:
            if use_jump:
                st.metric("Training Cutoff", training_cutoff)
            else:
                st.metric("Strategy Type", "Standard RS")
        
        if dynamic_rf_used:
            st.markdown(f"""
            <div class="rf-info">
                <strong>🏦 Dynamic Risk-Free Rate Information</strong><br>
                Ticker: {rf_ticker} | Performance metrics calculated using real-time RF data
            </div>
            """, unsafe_allow_html=True)
        
        # 핵심 지표
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_return = metrics.get('총 수익률', 'N/A')
            st.metric("Total Return", total_return)
        with col2:
            annual_return = metrics.get('연율화 수익률', 'N/A')
            st.metric("Annual Return", annual_return)
        with col3:
            sharpe_key = '샤프 비율 (동적)' if dynamic_rf_used else '샤프 비율 (기본)'
            sharpe_ratio = metrics.get(sharpe_key, metrics.get('샤프 비율', 'N/A'))
            st.metric("Sharpe Ratio", sharpe_ratio)
        with col4:
            max_dd = metrics.get('최대 낙폭', 'N/A')
            st.metric("Max Drawdown", max_dd)
        
        # 동적 RF 관련 추가 지표
        if dynamic_rf_used:
            st.subheader("🏦 Dynamic Risk-Free Rate Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sortino_ratio = metrics.get('소르티노 비율 (동적)', 'N/A')
                st.metric("Sortino Ratio", sortino_ratio)
            with col2:
                avg_rf = metrics.get('평균 Risk-Free Rate', 'N/A')
                st.metric("Avg RF Rate", avg_rf)
            with col3:
                rf_range = metrics.get('Risk-Free Rate 범위', 'N/A')
                st.metric("RF Range", rf_range)
            with col4:
                rf_ticker_info = metrics.get('Risk-Free Rate 티커', rf_ticker)
                st.metric("RF Source", rf_ticker_info)
        
        # Jump Model 관련 추가 지표
        if use_jump and 'BULL 기간' in metrics:
            st.subheader("🔄 Regime Analysis")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("BULL Period", metrics.get('BULL 기간', 'N/A'))
            with col2:
                st.metric("BULL Return", metrics.get('BULL 수익률', 'N/A'))
            with col3:
                st.metric("BEAR Period", metrics.get('BEAR 기간', 'N/A'))
            with col4:
                st.metric("BEAR Return", metrics.get('BEAR 수익률', 'N/A'))
        
        # Out-of-sample 분석
        if use_jump and 'Out-of-Sample Days' in metrics:
            st.subheader("🔮 Out-of-Sample Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("OOS Days", metrics.get('Out-of-Sample Days', 'N/A'))
            with col2:
                st.metric("OOS Return", metrics.get('Out-of-Sample Return', 'N/A'))
            with col3:
                oos_sharpe = metrics.get('Out-of-Sample Sharpe (동적)', 'N/A')
                st.metric("OOS Sharpe", oos_sharpe)
        
        # 포트폴리오 가치 차트
        portfolio_df = data.get('portfolio')
        
        if safe_data_check(portfolio_df):
            fig = go.Figure()
            
            # 포트폴리오 가치 라인
            fig.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            # Jump Model 사용시 체제별 배경색 추가
            if use_jump and 'regime' in portfolio_df.columns:
                bull_periods = portfolio_df[portfolio_df['regime'] == 'BULL']
                bear_periods = portfolio_df[portfolio_df['regime'] == 'BEAR']
                
                if not bull_periods.empty:
                    for i in range(len(bull_periods)):
                        fig.add_vrect(
                            x0=bull_periods.index[i], x1=bull_periods.index[i],
                            fillcolor="green", opacity=0.1, line_width=0
                        )
                
                if not bear_periods.empty:
                    for i in range(len(bear_periods)):
                        fig.add_vrect(
                            x0=bear_periods.index[i], x1=bear_periods.index[i],
                            fillcolor="red", opacity=0.1, line_width=0
                        )
            
            title_suffix = ""
            if use_jump:
                title_suffix += " (with Regime Background)"
            if dynamic_rf_used:
                title_suffix += f" - Dynamic RF: {rf_ticker}"
            
            fig.update_layout(
                title=f"Portfolio Value Over Time{title_suffix}",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 성과 분해 차트 (동적 RF 사용시)
            if dynamic_rf_used and HAS_RF_UTILS:
                st.subheader("📊 Performance Decomposition (Dynamic RF)")
                
                try:
                    # 간단한 성과 분해 시뮬레이션
                    returns = portfolio_df['value'].pct_change().dropna()
                    
                    # RF 데이터 다운로드
                    rf_manager = RiskFreeRateManager(rf_ticker, st.session_state.default_rf_rate)
                    start_date = portfolio_df.index[0]
                    end_date = portfolio_df.index[-1]
                    rf_data = rf_manager.download_risk_free_rate(start_date, end_date)
                    
                    if rf_data is not None and not rf_data.empty:
                        # RF 정렬
                        aligned_rf = rf_data.reindex(returns.index, method='ffill').fillna(st.session_state.default_rf_rate)
                        daily_rf = aligned_rf / 252
                        
                        # 초과 수익률 계산
                        excess_returns = returns - daily_rf
                        
                        # 누적 성과 분해
                        cumulative_total = (1 + returns).cumprod()
                        cumulative_rf = (1 + daily_rf).cumprod()
                        cumulative_excess = (1 + excess_returns).cumprod()
                        
                        fig_decomp = go.Figure()
                        
                        fig_decomp.add_trace(go.Scatter(
                            x=returns.index,
                            y=(cumulative_total - 1) * 100,
                            mode='lines',
                            name='Total Return',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_decomp.add_trace(go.Scatter(
                            x=returns.index,
                            y=(cumulative_rf - 1) * 100,
                            mode='lines',
                            name=f'Risk-Free Return ({rf_ticker})',
                            line=dict(color='green', width=1)
                        ))
                        
                        fig_decomp.add_trace(go.Scatter(
                            x=returns.index,
                            y=(cumulative_excess - 1) * 100,
                            mode='lines',
                            name='Excess Return',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig_decomp.update_layout(
                            title="Return Decomposition: Total = Risk-Free + Excess",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Return (%)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_decomp, use_container_width=True)
                        
                        # 성과 기여도 표시
                        final_total = (cumulative_total.iloc[-1] - 1) * 100
                        final_rf = (cumulative_rf.iloc[-1] - 1) * 100
                        final_excess = (cumulative_excess.iloc[-1] - 1) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Return", f"{final_total:.2f}%")
                        with col2:
                            st.metric("RF Contribution", f"{final_rf:.2f}%")
                        with col3:
                            st.metric("Excess Return", f"{final_excess:.2f}%")
                            
                except Exception as e:
                    st.warning(f"Performance decomposition failed: {str(e)}")
            
            # 상세 메트릭스 테이블
            if metrics:
                st.subheader("📋 Detailed Metrics")
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                st.dataframe(metrics_df, use_container_width=True)
                
                # Training cutoff 정보 추가
                if use_jump:
                    st.caption(f"💡 Jump Model was trained on data up to {training_cutoff}, providing out-of-sample predictions for 2025")
                
                if dynamic_rf_used:
                    st.caption(f"🏦 Performance metrics calculated using dynamic Risk-Free Rate ({rf_ticker})")
        else:
            st.warning("Portfolio data not available for charting")
    
    def download_results(self):
        """결과 다운로드 (동적 RF 정보 포함)"""
        try:
            if safe_data_check(st.session_state.portfolio_data):
                portfolio_df = st.session_state.portfolio_data['portfolio']
                if safe_data_check(portfolio_df):
                    # 메타데이터 추가
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    rf_info = f"_RF_{st.session_state.rf_ticker}" if HAS_RF_UTILS else "_FixedRF"
                    
                    csv = portfolio_df.to_csv()
                    st.download_button(
                        label="📥 Download Portfolio Data",
                        data=csv,
                        file_name=f"portfolio{rf_info}_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                    # 메트릭스도 다운로드 옵션 제공
                    metrics = st.session_state.portfolio_data.get('metrics', {})
                    if metrics:
                        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                        metrics_csv = metrics_df.to_csv()
                        st.download_button(
                            label="📊 Download Performance Metrics",
                            data=metrics_csv,
                            file_name=f"metrics{rf_info}_{timestamp}.csv",
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
            st.session_state.selected_tickers_for_regime = []
            st.session_state.regime_analysis_mode = 'all'
            # RF 설정은 유지
            st.success("✅ All cache cleared! (RF settings preserved)")
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
        
        # 디버깅 정보 표시
        if st.checkbox("Show Debug Info"):
            st.exception(e)


if __name__ == "__main__":
    main()
