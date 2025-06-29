import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class RealtimeKospiSectorRS:
    def __init__(self, length=20, timeframe='daily', recent_cross_days=None):
        """
        실시간 RS 신호 생성기
        
        Parameters:
        - length: RS 계산 기간
        - timeframe: 'daily' 또는 'weekly'
        - recent_cross_days: 최근 N일 내 크로스 필터링
        """
        self.sector_etfs = {
            '139220.KS': 'TIGER 200 IT',
            '139230.KS': 'TIGER 200 산업재',
            '139240.KS': 'TIGER 200 소비재',
            '139250.KS': 'TIGER 200 금융',
            '139260.KS': 'TIGER 200 중공업',
            '139270.KS': 'TIGER 200 경기소비재',
            '139280.KS': 'TIGER 200 에너지화학',
            '139290.KS': 'TIGER 200 철강소재',
            '227540.KS': 'TIGER 200 건강관리',
            '227550.KS': 'TIGER 200 커뮤니케이션서비스'
        }
        
        self.benchmark = '069500.KS'
        self.length = length
        self.timeframe = timeframe
        self.recent_cross_days = recent_cross_days
        
    def get_realtime_signals(self, lookback_days=120):
        """현재 시점의 실시간 신호 생성"""
        print(f"실시간 신호 생성 중... (timeframe: {self.timeframe}, recent_cross: {self.recent_cross_days})")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # 전략 인스턴스 생성
        from kospi_sector_rs_enhanced import EnhancedKospiSectorRSStrategy
        strategy = EnhancedKospiSectorRSStrategy(
            length=self.length,
            timeframe=self.timeframe,
            recent_cross_days=self.recent_cross_days
        )
        
        # 데이터 다운로드
        price_data, benchmark_data = strategy.get_price_data(start_date, end_date)
        
        if price_data is None:
            return None
        
        # 각 섹터의 RS 지표 계산
        signals = []
        cross_info = {}
        
        for ticker, name in self.sector_etfs.items():
            if ticker not in price_data:
                continue
                
            try:
                # RS 컴포넌트 계산
                rs_components = strategy.calculate_rs_components(
                    price_data[ticker], benchmark_data
                )
                
                if rs_components.empty:
                    continue
                
                # 최신 값
                latest = rs_components.iloc[-1]
                
                # 추세 분석
                rs_ratio_trend = rs_components['rs_ratio'].rolling(5).mean().iloc[-1]
                rs_momentum_trend = rs_components['rs_momentum'].rolling(5).mean().iloc[-1]
                
                # 크로스 정보 계산
                cross_days_ratio = self.find_cross_days(rs_components['rs_ratio'], 100)
                cross_days_momentum = self.find_cross_days(rs_components['rs_momentum'], 100)
                
                # 신호 강도 계산
                signal_strength = self.calculate_signal_strength(
                    latest['rs_ratio'], 
                    latest['rs_momentum'],
                    cross_days_ratio,
                    cross_days_momentum
                )
                
                # 최근 크로스 확인
                if self.recent_cross_days is not None:
                    recent_cross = strategy.check_recent_cross(
                        rs_components, 
                        end_date, 
                        self.recent_cross_days
                    )
                else:
                    recent_cross = True
                
                signal = {
                    'ticker': ticker,
                    'name': name,
                    'rs_ratio': latest['rs_ratio'],
                    'rs_momentum': latest['rs_momentum'],
                    'rs_ratio_trend': rs_ratio_trend,
                    'rs_momentum_trend': rs_momentum_trend,
                    'cross_days_ratio': cross_days_ratio,
                    'cross_days_momentum': cross_days_momentum,
                    'signal': self.determine_signal(latest, recent_cross),
                    'strength': signal_strength,
                    'recent_cross': recent_cross,
                    'current_price': price_data[ticker].iloc[-1],
                    'date': end_date.strftime('%Y-%m-%d %H:%M')
                }
                
                signals.append(signal)
                
                # 크로스 정보 저장
                if cross_days_ratio is not None or cross_days_momentum is not None:
                    cross_info[name] = {
                        'ratio_cross': cross_days_ratio,
                        'momentum_cross': cross_days_momentum
                    }
                
            except Exception as e:
                print(f"{name} 신호 생성 실패: {e}")
                continue
        
        # DataFrame 생성 및 정렬
        signals_df = pd.DataFrame(signals)
        
        if not signals_df.empty:
            # 신호 강도로 정렬
            signals_df = signals_df.sort_values('strength', ascending=False)
            
            # 크로스 정보 출력
            if cross_info:
                print("\n=== 최근 크로스 정보 ===")
                for name, info in cross_info.items():
                    ratio_days = info['ratio_cross']
                    momentum_days = info['momentum_cross']
                    
                    if ratio_days is not None and ratio_days < 30:
                        print(f"{name}: RS-Ratio가 {ratio_days}일 전에 100 돌파")
                    if momentum_days is not None and momentum_days < 30:
                        print(f"{name}: RS-Momentum이 {momentum_days}일 전에 100 돌파")
        
        return signals_df
    
    def find_cross_days(self, series, threshold, max_days=60):
        """특정 임계값을 넘은 날짜 찾기"""
        try:
            # 최근 max_days 동안의 데이터
            recent_data = series.tail(max_days)
            
            # 임계값 이상인 첫 번째 인덱스 찾기
            above_threshold = recent_data >= threshold
            
            if above_threshold.any():
                # 처음으로 임계값을 넘은 위치
                first_above = above_threshold.idxmax()
                first_above_idx = recent_data.index.get_loc(first_above)
                
                # 그 이전에 임계값 아래였는지 확인
                if first_above_idx > 0:
                    if recent_data.iloc[first_above_idx - 1] < threshold:
                        # 크로스가 발생한 날로부터 경과일
                        days_since_cross = len(recent_data) - first_above_idx - 1
                        return days_since_cross
            
            return None
            
        except:
            return None
    
    def calculate_signal_strength(self, rs_ratio, rs_momentum, cross_days_ratio, cross_days_momentum):
        """신호 강도 계산 (0-100)"""
        strength = 0
        
        # 기본 점수 (RS 값 기반)
        if rs_ratio > 100:
            strength += min((rs_ratio - 100) * 2, 30)  # 최대 30점
        if rs_momentum > 100:
            strength += min((rs_momentum - 100) * 2, 30)  # 최대 30점
        
        # 최근 크로스 보너스
        if cross_days_ratio is not None and cross_days_ratio < 10:
            strength += 20  # 10일 이내 크로스
        elif cross_days_ratio is not None and cross_days_ratio < 20:
            strength += 10  # 20일 이내 크로스
            
        if cross_days_momentum is not None and cross_days_momentum < 10:
            strength += 20  # 10일 이내 크로스
        elif cross_days_momentum is not None and cross_days_momentum < 20:
            strength += 10  # 20일 이내 크로스
        
        return min(strength, 100)
    
    def determine_signal(self, latest, recent_cross):
        """신호 결정"""
        if latest['rs_ratio'] >= 100 and latest['rs_momentum'] >= 100:
            if self.recent_cross_days is None or recent_cross:
                return 'BUY'
            else:
                return 'HOLD_OLD'  # 조건은 만족하지만 오래됨
        elif latest['rs_ratio'] >= 95 or latest['rs_momentum'] >= 95:
            return 'WATCH'  # 관찰 대상
        else:
            return 'HOLD'
    
    def get_portfolio_recommendation(self, signals_df, max_positions=5):
        """포트폴리오 추천"""
        if signals_df is None or signals_df.empty:
            return None
        
        # BUY 신호만 필터링
        buy_signals = signals_df[signals_df['signal'] == 'BUY'].copy()
        
        if buy_signals.empty:
            # WATCH 신호 확인
            watch_signals = signals_df[signals_df['signal'] == 'WATCH']
            
            return {
                'action': 'HOLD_CASH',
                'reason': '매수 조건을 만족하는 섹터가 없습니다.',
                'watch_list': watch_signals[['name', 'rs_ratio', 'rs_momentum']].to_dict('records') if not watch_signals.empty else []
            }
        
        # 상위 섹터 선택
        selected = buy_signals.head(max_positions)
        
        # 신호 강도별 그룹
        strong_signals = selected[selected['strength'] >= 70]
        normal_signals = selected[selected['strength'] < 70]
        
        return {
            'action': 'INVEST',
            'sectors': selected.to_dict('records'),
            'allocation': 1.0 / len(selected),
            'strong_count': len(strong_signals),
            'normal_count': len(normal_signals),
            'total_strength': selected['strength'].mean()
        }
    
    def analyze_market_condition(self, signals_df):
        """전체 시장 상황 분석"""
        if signals_df is None or signals_df.empty:
            return None
        
        total_sectors = len(signals_df)
        buy_sectors = len(signals_df[signals_df['signal'] == 'BUY'])
        watch_sectors = len(signals_df[signals_df['signal'] == 'WATCH'])
        
        # 평균 RS 값
        avg_rs_ratio = signals_df['rs_ratio'].mean()
        avg_rs_momentum = signals_df['rs_momentum'].mean()
        
        # 100 이상 섹터 비율
        ratio_above_100 = len(signals_df[signals_df['rs_ratio'] >= 100]) / total_sectors * 100
        momentum_above_100 = len(signals_df[signals_df['rs_momentum'] >= 100]) / total_sectors * 100
        
        # 시장 상태 판단
        if buy_sectors >= 5:
            market_state = "강세"
        elif buy_sectors >= 3:
            market_state = "중립-강세"
        elif buy_sectors >= 1:
            market_state = "중립-약세"
        else:
            market_state = "약세"
        
        return {
            'market_state': market_state,
            'buy_sectors': buy_sectors,
            'watch_sectors': watch_sectors,
            'avg_rs_ratio': avg_rs_ratio,
            'avg_rs_momentum': avg_rs_momentum,
            'ratio_above_100_pct': ratio_above_100,
            'momentum_above_100_pct': momentum_above_100
        }

# 사용 예시
if __name__ == "__main__":
    print("=== 코스피 섹터 RS 실시간 분석 ===\n")
    
    # 1. 일봉 기본 분석
    print("1. 일봉 기본 분석")
    analyzer_daily = RealtimeKospiSectorRS(
        length=20,
        timeframe='daily',
        recent_cross_days=None
    )
    
    signals_daily = analyzer_daily.get_realtime_signals()
    
    if signals_daily is not None:
        print("\n전체 섹터 현황:")
        print(signals_daily[['name', 'rs_ratio', 'rs_momentum', 'signal', 'strength']].round(2))
        
        # 시장 상황 분석
        market_condition = analyzer_daily.analyze_market_condition(signals_daily)
        print(f"\n시장 상태: {market_condition['market_state']}")
        print(f"매수 신호 섹터: {market_condition['buy_sectors']}개")
        print(f"관찰 대상 섹터: {market_condition['watch_sectors']}개")
        
        # 포트폴리오 추천
        recommendation = analyzer_daily.get_portfolio_recommendation(signals_daily)
        if recommendation['action'] == 'INVEST':
            print(f"\n추천 포트폴리오 ({len(recommendation['sectors'])}개 섹터):")
            for sector in recommendation['sectors']:
                print(f"  - {sector['name']}: 강도={sector['strength']:.0f}")
    
    print("\n" + "="*60 + "\n")
    
    # 2. 최근 30일 크로스 필터링
    print("2. 최근 30일 크로스 필터링 분석")
    analyzer_cross = RealtimeKospiSectorRS(
        length=20,
        timeframe='daily',
        recent_cross_days=30
    )
    
    signals_cross = analyzer_cross.get_realtime_signals()
    
    if signals_cross is not None:
        buy_signals = signals_cross[signals_cross['signal'] == 'BUY']
        
        if not buy_signals.empty:
            print("\n최근 30일 내 크로스한 매수 신호:")
            for _, signal in buy_signals.iterrows():
                print(f"  - {signal['name']}: "
                      f"RS-Ratio={signal['rs_ratio']:.1f} "
                      f"(크로스: {signal['cross_days_ratio']}일 전), "
                      f"RS-Momentum={signal['rs_momentum']:.1f} "
                      f"(크로스: {signal['cross_days_momentum']}일 전)")
        else:
            print("\n최근 30일 내 크로스한 매수 신호가 없습니다.")
    
    # 결과 저장
    if signals_daily is not None:
        signals_daily.to_csv('realtime_signals.csv', index=False)
        print("\n신호가 realtime_signals.csv에 저장되었습니다.")