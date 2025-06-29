import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enhancedkospi import EnhancedKospiSectorRSStrategy
from kospijumpmodel import KospiJumpModel
import warnings
warnings.filterwarnings('ignore')

class RSStrategyWithJumpModel:
    """
    RS 전략 + Jump Model 통합
    - KOSPI 200이 BEAR 체제일 때 모든 섹터 투자 중단
    - BULL 체제일 때만 RS 전략 실행
    """
    
    def __init__(self, rs_length=20, rs_timeframe='daily', rs_recent_cross_days=None,
                 jump_penalty=50.0, regime_lookback=20):
        """
        Parameters:
        - rs_length: RS 계산 기간
        - rs_timeframe: RS 계산 주기 ('daily' or 'weekly')
        - rs_recent_cross_days: 최근 크로스 필터링 기간
        - jump_penalty: Jump Model의 체제 전환 페널티
        - regime_lookback: 체제 판단을 위한 lookback 기간
        """
        # RS 전략 초기화
        self.rs_strategy = EnhancedKospiSectorRSStrategy(
            length=rs_length,
            timeframe=rs_timeframe,
            recent_cross_days=rs_recent_cross_days
        )
        
        # Jump Model 초기화
        self.jump_model = KospiJumpModel(
            n_states=2,
            lookback_window=regime_lookback,
            jump_penalty=jump_penalty
        )
        
        self.regime_history = None
        
    def prepare_regime_data(self, start_date, end_date):
        """백테스트를 위한 체제 데이터 준비"""
        # Jump Model 학습을 위해 추가 기간 확보
        extended_start = start_date - timedelta(days=100)
        
        # 체제 이력 계산
        print("KOSPI 200 체제 분석 중...")
        self.regime_history = self.jump_model.get_regime_history(extended_start, end_date)
        
        if self.regime_history is not None:
            # 체제별 통계
            bull_pct = (self.regime_history['regime'] == 'BULL').mean() * 100
            bear_pct = (self.regime_history['regime'] == 'BEAR').mean() * 100
            
            print(f"체제 분포: BULL {bull_pct:.1f}%, BEAR {bear_pct:.1f}%")
            
            # 체제 전환 횟수
            regime_changes = self.regime_history['regime'] != self.regime_history['regime'].shift()
            n_changes = regime_changes.sum() - 1
            
            print(f"체제 전환 횟수: {n_changes}회")
        
        return self.regime_history
    
    def get_regime_on_date(self, date):
        """특정 날짜의 체제 확인"""
        if self.regime_history is None:
            return 'BULL'  # 기본값
        
        # 해당 날짜 또는 가장 가까운 이전 날짜의 체제
        try:
            if date in self.regime_history.index:
                return self.regime_history.loc[date, 'regime']
            else:
                # 이전 날짜 중 가장 가까운 날짜
                prev_dates = self.regime_history.index[self.regime_history.index <= date]
                if len(prev_dates) > 0:
                    return self.regime_history.loc[prev_dates[-1], 'regime']
                else:
                    return 'BULL'
        except:
            return 'BULL'
    
    def backtest(self, start_date, end_date, initial_capital=10000000):
        """Jump Model을 적용한 백테스트"""
        
        # 1. 체제 데이터 준비
        self.prepare_regime_data(start_date, end_date)
        
        # 2. RS 전략용 데이터 준비
        extra_days = 200 if self.rs_strategy.timeframe == 'weekly' else 100
        price_data, benchmark_data = self.rs_strategy.get_price_data(
            start_date - timedelta(days=extra_days),
            end_date
        )
        
        if price_data is None or benchmark_data is None:
            print("데이터 다운로드 실패")
            return None, None
        
        # 3. 리밸런싱 날짜 생성
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # 4. 포트폴리오 초기화
        portfolio_value = initial_capital
        portfolio_history = []
        holdings = {}
        trade_history = []
        regime_log = []
        
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"\n{rebal_date.strftime('%Y-%m-%d')} 리밸런싱")
            
            # 현재 체제 확인
            current_regime = self.get_regime_on_date(rebal_date)
            print(f"현재 시장 체제: {current_regime}")
            
            regime_log.append({
                'date': rebal_date,
                'regime': current_regime
            })
            
            # BEAR 체제인 경우 모든 포지션 청산
            if current_regime == 'BEAR':
                print("BEAR 체제 - 모든 투자 중단")
                
                # 기존 포지션 청산
                if holdings:
                    for ticker, holding in holdings.items():
                        trade_history.append({
                            'date': rebal_date,
                            'ticker': ticker,
                            'name': holding['name'],
                            'action': 'SELL_BEAR',
                            'shares': holding['shares'],
                            'price': price_data[ticker].loc[rebal_date] if rebal_date in price_data[ticker].index else holding['buy_price']
                        })
                    holdings = {}
                
                # 다음 리밸런싱까지 현금 보유
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                else:
                    next_date = end_date
                
                dates = pd.date_range(start=rebal_date, end=next_date, freq='D')
                for date in dates:
                    if date <= end_date:
                        portfolio_history.append({
                            'date': date,
                            'value': portfolio_value,
                            'holdings': 0,
                            'regime': current_regime
                        })
                continue
            
            # BULL 체제인 경우 RS 전략 실행
            analysis_date = rebal_date
            if self.rs_strategy.timeframe == 'weekly':
                days_ahead = 4 - rebal_date.weekday()
                if days_ahead < 0:
                    days_ahead += 7
                analysis_date = rebal_date + timedelta(days=days_ahead)
            
            # 섹터 선택 (RS 전략)
            selected_sectors = self.rs_strategy.select_sectors(price_data, benchmark_data, analysis_date)
            
            if not selected_sectors:
                print("선택된 섹터가 없습니다.")
                # 현금 보유
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                else:
                    next_date = end_date
                
                dates = pd.date_range(start=rebal_date, end=next_date, freq='D')
                for date in dates:
                    if date <= end_date:
                        portfolio_history.append({
                            'date': date,
                            'value': portfolio_value,
                            'holdings': 0,
                            'regime': current_regime
                        })
                continue
            
            # 선택된 섹터 출력
            print(f"선택된 섹터 수: {len(selected_sectors)}")
            for sector in selected_sectors:
                print(f"  - {sector['name']}: RS-Ratio={sector['rs_ratio']:.2f}, RS-Momentum={sector['rs_momentum']:.2f}")
            
            # 동일 가중 투자
            investment_per_sector = portfolio_value / len(selected_sectors)
            
            # 새로운 포지션 구성
            new_holdings = {}
            for sector in selected_sectors:
                ticker = sector['ticker']
                try:
                    if analysis_date in price_data[ticker].index:
                        current_price = price_data[ticker].loc[analysis_date]
                    else:
                        nearest_date = price_data[ticker].index[price_data[ticker].index.get_indexer([analysis_date], method='nearest')[0]]
                        current_price = price_data[ticker].loc[nearest_date]
                    
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    
                    shares = int(investment_per_sector / current_price)
                    if shares > 0:
                        new_holdings[ticker] = {
                            'shares': shares,
                            'buy_price': float(current_price),
                            'name': sector['name']
                        }
                        
                        trade_history.append({
                            'date': rebal_date,
                            'ticker': ticker,
                            'name': sector['name'],
                            'action': 'BUY',
                            'shares': shares,
                            'price': float(current_price)
                        })
                
                except Exception as e:
                    print(f"{ticker} 매수 실패: {e}")
                    continue
            
            holdings = new_holdings
            
            # 다음 리밸런싱까지 포트폴리오 추적
            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]
            else:
                next_date = end_date
            
            # 일별 포트폴리오 가치 계산
            dates = pd.date_range(start=rebal_date, end=next_date, freq='D')
            for date in dates:
                if date <= end_date:
                    # 현재 체제 확인
                    daily_regime = self.get_regime_on_date(date)
                    
                    # BEAR 체제로 전환된 경우 즉시 청산
                    if daily_regime == 'BEAR' and holdings:
                        print(f"\n{date.strftime('%Y-%m-%d')} BEAR 체제 감지 - 긴급 청산")
                        
                        portfolio_value = 0
                        for ticker, holding in holdings.items():
                            try:
                                if date in price_data[ticker].index:
                                    exit_price = price_data[ticker].loc[date]
                                else:
                                    exit_price = holding['buy_price']
                                
                                portfolio_value += holding['shares'] * exit_price
                                
                                trade_history.append({
                                    'date': date,
                                    'ticker': ticker,
                                    'name': holding['name'],
                                    'action': 'SELL_BEAR_EMERGENCY',
                                    'shares': holding['shares'],
                                    'price': float(exit_price)
                                })
                            except:
                                portfolio_value += holding['shares'] * holding['buy_price']
                        
                        holdings = {}
                    
                    # 포트폴리오 가치 계산
                    elif holdings:
                        daily_value = 0
                        for ticker, holding in holdings.items():
                            try:
                                if self.rs_strategy.timeframe == 'weekly':
                                    if date in price_data[ticker].index:
                                        current_price = price_data[ticker].loc[date]
                                    else:
                                        available_dates = price_data[ticker].index[price_data[ticker].index <= date]
                                        if len(available_dates) > 0:
                                            current_price = price_data[ticker].loc[available_dates[-1]]
                                        else:
                                            current_price = holding['buy_price']
                                else:
                                    if date in price_data[ticker].index:
                                        current_price = price_data[ticker].loc[date]
                                    else:
                                        current_price = holding['buy_price']
                                
                                daily_value += holding['shares'] * current_price
                            except:
                                pass
                        
                        if daily_value > 0:
                            portfolio_value = daily_value
                    
                    portfolio_history.append({
                        'date': date,
                        'value': portfolio_value,
                        'holdings': len(holdings),
                        'regime': daily_regime
                    })
        
        # 결과 정리
        portfolio_df = pd.DataFrame(portfolio_history).drop_duplicates(subset='date').set_index('date')
        trades_df = pd.DataFrame(trade_history)
        regime_df = pd.DataFrame(regime_log).set_index('date')
        
        return portfolio_df, trades_df, regime_df
    
    def calculate_performance_metrics(self, portfolio_df):
        """성과 지표 계산 (체제별 분석 포함)"""
        if portfolio_df.empty:
            return {}
        
        # 기본 성과 지표
        metrics = self.rs_strategy.calculate_performance_metrics(portfolio_df)
        
        # 체제별 성과 분석
        if 'regime' in portfolio_df.columns:
            bull_df = portfolio_df[portfolio_df['regime'] == 'BULL']
            bear_df = portfolio_df[portfolio_df['regime'] == 'BEAR']
            
            if not bull_df.empty:
                bull_days = len(bull_df)
                bull_return = (bull_df['value'].iloc[-1] / bull_df['value'].iloc[0] - 1) * 100 if len(bull_df) > 1 else 0
                metrics['BULL 기간'] = f"{bull_days}일 ({bull_days/len(portfolio_df)*100:.1f}%)"
                metrics['BULL 수익률'] = f"{bull_return:.2f}%"
            
            if not bear_df.empty:
                bear_days = len(bear_df)
                bear_return = (bear_df['value'].iloc[-1] / bear_df['value'].iloc[0] - 1) * 100 if len(bear_df) > 1 else 0
                metrics['BEAR 기간'] = f"{bear_days}일 ({bear_days/len(portfolio_df)*100:.1f}%)"
                metrics['BEAR 수익률'] = f"{bear_return:.2f}%"
        
        return metrics

# 사용 예시
if __name__ == "__main__":
    print("=== 코스피 섹터 RS 전략 + Jump Model ===\n")
    
    # 전략 생성
    strategy = RSStrategyWithJumpModel(
        rs_length=20,
        rs_timeframe='daily',
        rs_recent_cross_days=30,  # 최근 30일 크로스
        jump_penalty=50.0,
        regime_lookback=20
    )
    
    # 백테스트 기간
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    print(f"백테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print("초기 자본: 10,000,000원")
    print("-" * 60)
    
    # 백테스트 실행
    portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)
    
    if portfolio_df is not None and not portfolio_df.empty:
        # 성과 지표 계산
        metrics = strategy.calculate_performance_metrics(portfolio_df)
        
        print("\n=== 백테스트 결과 ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # 거래 통계
        if not trades_df.empty:
            print(f"\n=== 거래 통계 ===")
            print(f"총 거래 횟수: {len(trades_df)}")
            
            # 액션별 거래 횟수
            action_counts = trades_df['action'].value_counts()
            for action, count in action_counts.items():
                print(f"  - {action}: {count}회")
            
            # BEAR 청산 통계
            bear_sells = trades_df[trades_df['action'].str.contains('BEAR')]
            if not bear_sells.empty:
                print(f"\nBEAR 체제 청산: {len(bear_sells)}회")
        
        # 결과 저장
        portfolio_df.to_csv('portfolio_with_jump_model.csv')
        if not trades_df.empty:
            trades_df.to_csv('trades_with_jump_model.csv')
        if not regime_df.empty:
            regime_df.to_csv('regime_history.csv')
        
        print("\n결과가 CSV 파일로 저장되었습니다.")