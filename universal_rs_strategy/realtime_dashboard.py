"""
실시간 모니터링 대시보드
웹 기반 인터랙티브 대시보드 (Streamlit 사용)
완전히 수정된 버전 - pandas Series truth value 문제 해결
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
.colab-warning {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Colab 환경 알림
st.markdown("""
<div class="colab-warning">
🔬 <strong>Running on Google Colab</strong><br>
• 데이터 로딩에 시간이 걸릴 수 있습니다<br>
• 세션 유지를 위해 정기적으로 상호작용 해주세요<br>
• 중요한 백테스트 결과는 다운로드 받으세요
</div>
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
            
        # 기타 객체는 None이 아니면 True
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

class RealtimeDashboard:
    """실시간 모니터링 대시보드 - 안전한 버전"""
    
    def __init__(self):
        # 세션 상태 초기화
        if 'selected_preset' not in st.session_state:
            st.session_state.selected_preset = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = None
        
        # Colab 최적화된 프리셋 목록
        self.presets = {
            'S&P 500 Sectors (Top 6)': self.get_limited_sp500(),
            'KOSPI Sectors (Top 4)': self.get_limited_kospi(),
            'Major Countries': self.get_major_countries(),
            'Global Sectors (Top 4)': self.get_limited_global(),
            'Single ETF Test': self.get_single_etf()
        }
    
    def get_limited_sp500(self):
        """Colab용 제한된 S&P 500 섹터"""
        try:
            full_preset = PresetManager.get_sp500_sectors()
            limited_components = dict(list(full_preset['components'].items())[:6])
            return {
                'name': 'S&P 500 Sectors (Limited)',
                'benchmark': full_preset['benchmark'],
                'components': limited_components
            }
        except Exception:
            return self.get_single_etf()
    
    def get_limited_kospi(self):
        """Colab용 제한된 KOSPI 섹터"""
        try:
            full_preset = PresetManager.get_kospi_sectors()
            limited_components = dict(list(full_preset['components'].items())[:4])
            return {
                'name': 'KOSPI Sectors (Limited)',
                'benchmark': full_preset['benchmark'],
                'components': limited_components
            }
        except Exception:
            return self.get_single_etf()
    
    def get_major_countries(self):
        """주요 국가 ETF만"""
        return {
            'name': 'Major Countries Strategy',
            'benchmark': 'URTH',
            'components': {
                'EWJ': 'Japan',
                'EWG': 'Germany',
                'EWU': 'United Kingdom',
                'EWC': 'Canada'
            }
        }
    
    def get_limited_global(self):
        """제한된 글로벌 섹터"""
        try:
            full_preset = PresetManager.get_global_sectors()
            limited_components = dict(list(full_preset['components'].items())[:4])
            return {
                'name': 'Global Sectors (Limited)',
                'benchmark': full_preset['benchmark'],
                'components': limited_components
            }
        except Exception:
            return self.get_single_etf()
    
    def get_single_etf(self):
        """단일 ETF 테스트"""
        return {
            'name': 'Single ETF Test',
            'benchmark': '^GSPC',
            'components': {'SPY': 'SPDR S&P 500 ETF'}
        }
    
    def run(self):
        """대시보드 실행"""
        st.title("🚀 Universal RS Strategy Dashboard")
        st.markdown("### Real-time Market Monitoring & Signal Generation (Colab Edition)")
        
        # 사이드바
        self.create_sidebar()
        
        # 메인 컨텐츠
        if st.session_state.selected_preset:
            self.display_main_content()
        else:
            st.info("👈 Please select a strategy preset from the sidebar to begin")
    
    def create_sidebar(self):
        """사이드바 생성 - Colab 최적화"""
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
        
        # 전략 파라미터 (Colab 최적화)
        st.sidebar.subheader("Strategy Parameters")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            rs_length = st.number_input("RS Length", value=15, min_value=10, max_value=30)
            use_jump = st.checkbox("Use Jump Model", value=False)  # 기본값 비활성화
        
        with col2:
            timeframe = st.selectbox("Timeframe", ["daily"])  # weekly 제거
            use_cross = st.checkbox("Use Cross Filter", value=False)
        
        if use_cross:
            cross_days = st.sidebar.number_input("Cross Days", value=30, min_value=5, max_value=90)
        else:
            cross_days = None
        
        # 백테스트 설정 (기간 단축)
        st.sidebar.subheader("Backtest Settings")
        backtest_years = st.sidebar.slider("Backtest Period (Years)", 1, 3, 2)
        
        # 실행 버튼
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Update", type="primary"):
                self.update_data(rs_length, timeframe, cross_days, use_jump)
        
        with col2:
            if st.button("📊 Backtest"):
                self.run_backtest(rs_length, timeframe, cross_days, use_jump, backtest_years)
        
        # Colab 전용 기능
        st.sidebar.subheader("Colab Utils")
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
        
        # 탭 생성 (Colab에서는 3개만)
        tab1, tab2, tab3 = st.tabs([
            "📈 Market Status", "🎯 Current Signals", "📊 Backtest Results"
        ])
        
        with tab1:
            self.display_market_status()
        
        with tab2:
            self.display_current_signals()
        
        with tab3:
            self.display_backtest_results()
    
    def display_market_status(self):
        """시장 상태 표시 - 안전한 버전"""
        st.subheader("Market Regime Analysis")
        
        preset = st.session_state.selected_preset
        
        # 안전한 Jump Model 분석
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
                    st.info("💡 Try using 'Single ETF Test' preset for more stable analysis")
    
    def display_current_signals(self):
        """현재 투자 신호 표시 - 완전히 안전한 버전"""
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
                    
                    # 데이터 가져오기 (기간 단축)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=60)
                    
                    price_data, benchmark_data = strategy.get_price_data(start_date, end_date)
                    
                    # 🔧 핵심 수정: 안전한 데이터 검증
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
                                    signals_df.head(10),
                                    x='name',
                                    y='rs_ratio',
                                    title='Top Components by RS-Ratio',
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
                                    title='Top Components by RS-Momentum',
                                    color='rs_momentum',
                                    color_continuous_scale='RdYlGn'
                                )
                                fig_momentum.add_hline(y=100, line_dash="dash", line_color="black")
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
                    st.info("💡 Try using a simpler preset or check your internet connection")
    
    def update_data(self, rs_length, timeframe, cross_days, use_jump):
        """데이터 업데이트"""
        try:
            st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("✅ Data updated successfully!")
        except Exception as e:
            st.error(f"Update failed: {str(e)}")
    
    def run_backtest(self, rs_length, timeframe, cross_days, use_jump, years):
        """백테스트 실행 - 안전한 버전"""
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
                
                # 안전한 결과 검증
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
                st.info("💡 Try reducing the backtest period or using a simpler preset")
    
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
                        file_name=f"colab_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
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
            st.success("✅ Cache cleared!")
        except Exception as e:
            st.error(f"Cache clear failed: {str(e)}")


# Streamlit 앱 실행
def main():
    try:
        dashboard = RealtimeDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard initialization failed: {str(e)}")
        st.info("💡 Try refreshing the page or checking your file paths")


if __name__ == "__main__":
    main()
