import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf

class RealtimeRSWithJumpModel:
    """실시간 RS 신호 + Jump Model 통합 분석"""
    
    def __init__(self):
        from rs_kospijump import RSStrategyWithJumpModel
        from kospijumpmodel import KospiJumpModel
        from enhancedkospi_visualize import RealtimeKospiSectorRS
        
        # Jump Model
        self.jump_model = KospiJumpModel(jump_penalty=50.0)
        
        # RS 신호 생성기
        self.rs_analyzer = RealtimeKospiSectorRS(
            length=20,
            timeframe='daily',
            recent_cross_days=30
        )
        
    def get_current_market_status(self):
        """현재 시장 상태 종합 분석"""
        print("=== 현재 시장 상태 분석 ===\n")
        
        # 1. Jump Model로 체제 확인
        current_regime_info = self.jump_model.get_current_regime()
        
        if current_regime_info:
            current_regime = current_regime_info['regime']
            regime_date = current_regime_info['date']
            
            print(f"시장 체제: {current_regime}")
            print(f"기준일: {regime_date.strftime('%Y-%m-%d')}")
            print(f"현재 시장 지표:")
            
            features = current_regime_info['features']
            print(f"  - 실현 변동성: {features['realized_vol']*100:.1f}%")
            print(f"  - 하방 변동성: {features['downside_vol']*100:.1f}%")
            print(f"  - 최대 낙폭: {features['max_drawdown']*100:.1f}%")
            print(f"  - 상승일 비율: {features['up_days_ratio']*100:.1f}%")
        else:
            current_regime = 'UNKNOWN'
            print("시장 체제 확인 실패")
        
        # 2. RS 신호 확인 (BULL 체제일 때만)
        if current_regime == 'BULL':
            print("\n=== RS 신호 분석 ===")
            signals = self.rs_analyzer.get_realtime_signals()
            
            if signals is not None and not signals.empty:
                # 매수 신호 섹터
                buy_signals = signals[signals['signal'] == 'BUY']
                
                if not buy_signals.empty:
                    print(f"\n매수 가능 섹터 ({len(buy_signals)}개):")
                    for _, signal in buy_signals.iterrows():
                        print(f"  - {signal['name']}: "
                              f"RS-Ratio={signal['rs_ratio']:.1f}, "
                              f"RS-Momentum={signal['rs_momentum']:.1f}, "
                              f"강도={signal['strength']:.0f}")
                else:
                    print("\n매수 가능 섹터 없음")
                
                # 관찰 대상 섹터
                watch_signals = signals[signals['signal'] == 'WATCH']
                if not watch_signals.empty:
                    print(f"\n관찰 대상 섹터 ({len(watch_signals)}개):")
                    for _, signal in watch_signals.head(3).iterrows():
                        print(f"  - {signal['name']}: "
                              f"RS-Ratio={signal['rs_ratio']:.1f}, "
                              f"RS-Momentum={signal['rs_momentum']:.1f}")
                
                return {
                    'regime': current_regime,
                    'regime_info': current_regime_info,
                    'signals': signals,
                    'buy_count': len(buy_signals)
                }
        else:
            print(f"\n{current_regime} 체제 - 투자 중단 권고")
            return {
                'regime': current_regime,
                'regime_info': current_regime_info,
                'signals': None,
                'buy_count': 0
            }
    
    def visualize_regime_and_performance(self, lookback_days=365):
        """체제 변화와 시장 성과 시각화"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # 체제 이력 가져오기
        regime_history = self.jump_model.get_regime_history(start_date, end_date)
        
        if regime_history is None:
            print("체제 이력을 가져올 수 없습니다.")
            return
        
        # KOSPI 200 데이터
        kospi_data = yf.download('069500.KS', start=start_date, end=end_date, progress=False)
        
        if kospi_data.empty:
            print("KOSPI 200 데이터를 가져올 수 없습니다.")
            return
        
        # 시각화
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        fig.suptitle('KOSPI 200 체제 분석', fontsize=16)
        
        # 1. KOSPI 200 지수와 체제
        ax1 = axes[0]
        
        # 체제별 배경색
        for i in range(len(regime_history) - 1):
            if regime_history.iloc[i]['regime'] == 'BEAR':
                ax1.axvspan(
                    regime_history.index[i],
                    regime_history.index[i + 1],
                    alpha=0.3,
                    color='red',
                    label='BEAR' if i == 0 else ""
                )
            else:
                ax1.axvspan(
                    regime_history.index[i],
                    regime_history.index[i + 1],
                    alpha=0.3,
                    color='green',
                    label='BULL' if i == 0 else ""
                )
        
        # KOSPI 200 지수
        ax1.plot(kospi_data.index, kospi_data['Close'], 'b-', linewidth=2, label='KOSPI 200')
        ax1.set_ylabel('KOSPI 200 지수')
        ax1.set_title('KOSPI 200 지수와 시장 체제')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 일일 수익률
        ax2 = axes[1]
        returns = kospi_data['Close'].pct_change().dropna()
        
        # 체제별 수익률
        bull_returns = []
        bear_returns = []
        
        for date, ret in returns.items():
            if date in regime_history.index:
                if regime_history.loc[date, 'regime'] == 'BULL':
                    bull_returns.append((date, ret))
                    bear_returns.append((date, 0))
                else:
                    bear_returns.append((date, ret))
                    bull_returns.append((date, 0))
        
        if bull_returns:
            bull_dates, bull_vals = zip(*bull_returns)
            ax2.bar(bull_dates, bull_vals, color='green', alpha=0.6, label='BULL 수익률')
        
        if bear_returns:
            bear_dates, bear_vals = zip(*bear_returns)
            ax2.bar(bear_dates, bear_vals, color='red', alpha=0.6, label='BEAR 수익률')
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('일일 수익률 (%)')
        ax2.set_title('체제별 일일 수익률')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 20일 이동 변동성
        ax3 = axes[2]
        volatility = returns.rolling(20).std() * np.sqrt(252) * 100
        
        ax3.plot(volatility.index, volatility.values, 'purple', linewidth=2)
        ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='고변동성 기준')
        ax3.set_ylabel('변동성 (%)')
        ax3.set_xlabel('날짜')
        ax3.set_title('20일 이동 변동성')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regime_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 체제별 통계
        print("\n=== 체제별 성과 통계 ===")
        
        # BULL 체제 통계
        bull_mask = regime_history['regime'] == 'BULL'
        bull_dates = regime_history.index[bull_mask]
        bull_period_returns = returns[returns.index.isin(bull_dates)]
        
        if not bull_period_returns.empty:
            print(f"\nBULL 체제:")
            print(f"  - 기간: {len(bull_dates)}일 ({len(bull_dates)/len(regime_history)*100:.1f}%)")
            print(f"  - 평균 일일 수익률: {bull_period_returns.mean()*100:.3f}%")
            print(f"  - 연율화 수익률: {bull_period_returns.mean()*252*100:.1f}%")
            print(f"  - 변동성: {bull_period_returns.std()*np.sqrt(252)*100:.1f}%")
        
        # BEAR 체제 통계
        bear_mask = regime_history['regime'] == 'BEAR'
        bear_dates = regime_history.index[bear_mask]
        bear_period_returns = returns[returns.index.isin(bear_dates)]
        
        if not bear_period_returns.empty:
            print(f"\nBEAR 체제:")
            print(f"  - 기간: {len(bear_dates)}일 ({len(bear_dates)/len(regime_history)*100:.1f}%)")
            print(f"  - 평균 일일 수익률: {bear_period_returns.mean()*100:.3f}%")
            print(f"  - 연율화 수익률: {bear_period_returns.mean()*252*100:.1f}%")
            print(f"  - 변동성: {bear_period_returns.std()*np.sqrt(252)*100:.1f}%")
    
    def create_dashboard(self):
        """종합 대시보드 생성"""
        # 현재 상태
        status = self.get_current_market_status()
        
        # 대시보드 생성
        fig = plt.figure(figsize=(16, 10))
        
        # 레이아웃 설정
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 현재 체제 표시 (큰 박스)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        regime_color = 'green' if status['regime'] == 'BULL' else 'red'
        regime_text = f"현재 시장 체제: {status['regime']}"
        ax1.text(0.5, 0.5, regime_text, fontsize=24, fontweight='bold',
                ha='center', va='center', color=regime_color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor=regime_color, linewidth=3))
        
        # 2. 시장 지표
        ax2 = fig.add_subplot(gs[1, 0])
        if status['regime_info']:
            features = status['regime_info']['features']
            indicators = ['realized_vol', 'downside_vol', 'max_drawdown', 'up_days_ratio']
            labels = ['실현 변동성', '하방 변동성', '최대 낙폭', '상승일 비율']
            values = [features[ind]*100 for ind in indicators]
            
            colors = ['blue', 'orange', 'red', 'green']
            bars = ax2.bar(labels, values, color=colors, alpha=0.7)
            ax2.set_ylabel('값 (%)')
            ax2.set_title('현재 시장 지표')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 값 표시
            for bar, val in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom')
        
        # 3. 투자 가능 섹터
        ax3 = fig.add_subplot(gs[1, 1:])
        if status['signals'] is not None and status['buy_count'] > 0:
            buy_signals = status['signals'][status['signals']['signal'] == 'BUY'].head(5)
            
            sectors = buy_signals['name'].str.replace('TIGER 200 ', '')
            rs_ratios = buy_signals['rs_ratio'].values
            rs_momentums = buy_signals['rs_momentum'].values
            
            x = np.arange(len(sectors))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, rs_ratios - 100, width, label='RS-Ratio', alpha=0.8)
            bars2 = ax3.bar(x + width/2, rs_momentums - 100, width, label='RS-Momentum', alpha=0.8)
            
            ax3.set_xlabel('섹터')
            ax3.set_ylabel('100 대비 초과 수준')
            ax3.set_title(f'투자 가능 섹터 TOP 5 (총 {status["buy_count"]}개)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(sectors, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '투자 가능 섹터 없음', 
                    fontsize=16, ha='center', va='center')
            ax3.set_title('투자 가능 섹터')
        
        # 4. 최근 체제 변화
        ax4 = fig.add_subplot(gs[2, :])
        
        # 최근 60일 체제 이력
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        recent_regime = self.jump_model.get_regime_history(start_date, end_date)
        
        if recent_regime is not None:
            # 체제 변화 시각화
            dates = recent_regime.index
            regimes = recent_regime['regime'].map({'BULL': 1, 'BEAR': 0})
            
            ax4.fill_between(dates, 0, regimes, where=(regimes == 1), 
                           color='green', alpha=0.3, label='BULL')
            ax4.fill_between(dates, 0, regimes, where=(regimes == 0), 
                           color='red', alpha=0.3, label='BEAR')
            ax4.plot(dates, regimes, 'k-', linewidth=2)
            
            ax4.set_ylim(-0.1, 1.1)
            ax4.set_yticks([0, 1])
            ax4.set_yticklabels(['BEAR', 'BULL'])
            ax4.set_xlabel('날짜')
            ax4.set_title('최근 60일 체제 변화')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('market_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

# 사용 예시
if __name__ == "__main__":
    # 실시간 분석기 생성
    analyzer = RealtimeRSWithJumpModel()
    
    # 1. 현재 시장 상태 확인
    print("="*60)
    status = analyzer.get_current_market_status()
    
    # 2. 투자 결정
    print("\n" + "="*60)
    print("=== 투자 결정 ===")
    
    if status['regime'] == 'BULL' and status['buy_count'] > 0:
        print(f"\n✅ 투자 권고: {status['buy_count']}개 섹터에 분산 투자")
    elif status['regime'] == 'BEAR':
        print("\n❌ 투자 중단 권고: BEAR 체제로 현금 보유 권장")
    else:
        print("\n⚠️ 대기 권고: 투자 가능한 섹터가 없음")
    
    # 3. 체제 분석 시각화
    print("\n체제 분석 차트 생성 중...")
    analyzer.visualize_regime_and_performance(lookback_days=365)
    
    # 4. 종합 대시보드
    print("\n종합 대시보드 생성 중...")
    analyzer.create_dashboard()
    
    print("\n분석이 완료되었습니다.")