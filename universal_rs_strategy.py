import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class UniversalRSStrategy:
    """
    범용 RS (Relative Strength) 전략
    다양한 벤치마크와 구성요소로 사용 가능
    """
    
    def __init__(self, benchmark, components, name="Custom Strategy", 
                 length=20, timeframe='daily', recent_cross_days=None):
        """
        Parameters:
        - benchmark: 벤치마크 티커 (예: '^GSPC' for S&P 500)
        - components: 구성요소 딕셔너리 {ticker: name}
        - name: 전략 이름
        - length: RS 계산 기간
        - timeframe: 'daily' 또는 'weekly'
        - recent_cross_days: 최근 크로스 필터링 기간
        """
        self.benchmark = benchmark
        self.components = components
        self.strategy_name = name
        self.length = length
        self.timeframe = timeframe.lower()
        self.recent_cross_days = recent_cross_days
        
        # 유효성 검사
        if self.timeframe not in ['daily', 'weekly']:
            raise ValueError("timeframe은 'daily' 또는 'weekly'여야 합니다.")
        
        print(f"\n{self.strategy_name} 초기화")
        print(f"벤치마크: {self.benchmark}")
        print(f"구성요소: {len(self.components)}개")
        print(f"설정: timeframe={self.timeframe}, length={self.length}, recent_cross_days={self.recent_cross_days}")
    
    def resample_to_weekly(self, data):
        """일봉 데이터를 주봉으로 변환"""
        if isinstance(data, pd.Series):
            weekly = data.resample('W').last()
        else:
            weekly = data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
        return weekly.dropna()
    
    def calculate_wma(self, data, period):
        """가중이동평균 계산"""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum() if len(x) == period else np.nan
        )
    
    def calculate_rs_components(self, price_data, benchmark_data):
        """RS-Ratio와 RS-Momentum 계산"""
        try:
            if isinstance(price_data, pd.DataFrame):
                price_data = price_data.squeeze()
            if isinstance(benchmark_data, pd.DataFrame):
                benchmark_data = benchmark_data.squeeze()
            
            aligned_data = pd.DataFrame({
                'price': price_data,
                'benchmark': benchmark_data
            }).dropna()
            
            if len(aligned_data) < self.length * 2:
                return pd.DataFrame(columns=['rs_ratio', 'rs_momentum'])
            
            # RS 계산
            rs = aligned_data['price'] / aligned_data['benchmark']
            
            # WMA of RS
            wma_rs = self.calculate_wma(rs, self.length)
            
            # RS Ratio
            rs_ratio_raw = rs / wma_rs
            rs_ratio = self.calculate_wma(rs_ratio_raw, self.length) * 100
            
            # RS Momentum
            wma_rs_ratio = self.calculate_wma(rs_ratio, self.length)
            rs_momentum = (rs_ratio / wma_rs_ratio * 100).fillna(100)
            
            result_df = pd.DataFrame({
                'rs_ratio': rs_ratio.values if hasattr(rs_ratio, 'values') else rs_ratio,
                'rs_momentum': rs_momentum.values if hasattr(rs_momentum, 'values') else rs_momentum
            }, index=rs_ratio.index if hasattr(rs_ratio, 'index') else aligned_data.index)
            
            return result_df
            
        except Exception as e:
            print(f"RS 계산 중 오류 발생: {e}")
            return pd.DataFrame(columns=['rs_ratio', 'rs_momentum'])
    
    def check_recent_cross(self, rs_components, analysis_date, days_back):
        """최근 N일 내에 100을 크로스했는지 확인"""
        if rs_components.empty or days_back is None:
            return True
        
        try:
            end_date = analysis_date
            
            if self.timeframe == 'weekly':
                periods_back = days_back // 7 + 1
            else:
                periods_back = days_back
            
            recent_data = rs_components.loc[:end_date].tail(periods_back + 1)
            
            if len(recent_data) < 2:
                return False
            
            rs_ratio_cross = False
            rs_momentum_cross = False
            
            for i in range(len(recent_data) - 1):
                if recent_data['rs_ratio'].iloc[i] <= 100 < recent_data['rs_ratio'].iloc[i + 1]:
                    rs_ratio_cross = True
                    break
            
            for i in range(len(recent_data) - 1):
                if recent_data['rs_momentum'].iloc[i] <= 100 < recent_data['rs_momentum'].iloc[i + 1]:
                    rs_momentum_cross = True
                    break
            
            return rs_ratio_cross or rs_momentum_cross
            
        except Exception as e:
            print(f"크로스 확인 중 오류: {e}")
            return True
    
    def get_price_data(self, start_date, end_date):
        """가격 데이터 다운로드"""
        price_data = {}
        
        def extract_close_series(data):
            """DataFrame에서 Close Series를 안전하게 추출"""
            if isinstance(data, pd.Series):
                return data
            
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns:
                    close_data = data['Close']
                    if isinstance(close_data, pd.DataFrame):
                        return close_data.squeeze()
                    return close_data
                elif len(data.columns) > 0:
                    return data.iloc[:, 0]
            
            return pd.Series()
        
        # 벤치마크 데이터 다운로드
        try:
            print(f"\n벤치마크 데이터 다운로드: {self.benchmark}")
            benchmark_df = yf.download(
                self.benchmark, 
                start=start_date, 
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            benchmark_data = extract_close_series(benchmark_df)
            
            if self.timeframe == 'weekly' and not benchmark_data.empty:
                benchmark_data = self.resample_to_weekly(benchmark_data)
            
            if benchmark_data.empty:
                raise ValueError("벤치마크 데이터를 가져올 수 없습니다.")
                
            print(f"벤치마크 데이터: {len(benchmark_data)}개 {self.timeframe} 데이터")
                
        except Exception as e:
            print(f"벤치마크 데이터 다운로드 실패: {e}")
            return None, None
        
        # 각 구성요소 데이터 다운로드
        for ticker, name in self.components.items():
            try:
                print(f"{name} 데이터 다운로드 중...")
                
                df = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
                
                data = extract_close_series(df)
                
                if self.timeframe == 'weekly' and not data.empty:
                    data = self.resample_to_weekly(data)
                
                if not data.empty:
                    price_data[ticker] = data
                    print(f"{name}: {len(data)}개 {self.timeframe} 데이터")
                else:
                    print(f"{name} 데이터가 비어있습니다.")
                    
            except Exception as e:
                print(f"{name} 다운로드 실패: {e}")
                continue
        
        print(f"\n총 {len(price_data)}개 구성요소 데이터 다운로드 완료")
        return price_data, benchmark_data
    
    def select_components(self, price_data, benchmark_data, date):
        """특정 날짜에 RS-Ratio와 RS-Momentum이 모두 100 이상인 구성요소 선택"""
        selected_components = []
        
        if self.timeframe == 'weekly':
            lookback_days = self.length * 7 * 4
        else:
            lookback_days = self.length * 4
        start_date = date - timedelta(days=lookback_days)
        
        for ticker in price_data.keys():
            try:
                ticker_prices = price_data[ticker][start_date:date]
                bench_prices = benchmark_data[start_date:date]
                
                if isinstance(ticker_prices, pd.DataFrame):
                    ticker_prices = ticker_prices.squeeze()
                if isinstance(bench_prices, pd.DataFrame):
                    bench_prices = bench_prices.squeeze()
                
                if len(ticker_prices) < self.length * 2 or len(bench_prices) < self.length * 2:
                    continue
                
                rs_components = self.calculate_rs_components(ticker_prices, bench_prices)
                
                if not rs_components.empty and len(rs_components) > 0:
                    latest_rs_ratio = rs_components['rs_ratio'].iloc[-1]
                    latest_rs_momentum = rs_components['rs_momentum'].iloc[-1]
                    
                    if pd.isna(latest_rs_ratio) or pd.isna(latest_rs_momentum):
                        continue
                    
                    if latest_rs_ratio >= 100 and latest_rs_momentum >= 100:
                        selected_components.append({
                            'ticker': ticker,
                            'name': self.components[ticker],
                            'rs_ratio': float(latest_rs_ratio),
                            'rs_momentum': float(latest_rs_momentum)
                        })
                        
            except Exception as e:
                print(f"{ticker} 분석 중 오류: {e}")
                continue
        
        return selected_components
    
    def backtest(self, start_date, end_date, initial_capital=10000000):
        """백테스트 실행"""
        if self.timeframe == 'weekly':
            extra_days = 200
        else:
            extra_days = 100
            
        price_data, benchmark_data = self.get_price_data(
            start_date - timedelta(days=extra_days),
            end_date
        )
        
        if price_data is None or benchmark_data is None:
            print("데이터 다운로드 실패")
            return None, None
        
        if self.timeframe == 'weekly':
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        else:
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        portfolio_value = initial_capital
        cash_balance = initial_capital
        portfolio_history = []
        holdings = {}
        trade_history = []
        
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"\n{rebal_date.strftime('%Y-%m-%d')} 리밸런싱")
            
            if self.timeframe == 'weekly':
                days_ahead = 4 - rebal_date.weekday()
                if days_ahead < 0:
                    days_ahead += 7
                analysis_date = rebal_date + timedelta(days=days_ahead)
            else:
                analysis_date = rebal_date
            
            # 기존 포지션 청산
            if holdings:
                cash_balance = 0
                for ticker, holding in holdings.items():
                    try:
                        if analysis_date in price_data[ticker].index:
                            sell_price = float(price_data[ticker].loc[analysis_date])
                        else:
                            nearest_idx = price_data[ticker].index.get_indexer([analysis_date], method='nearest')[0]
                            sell_price = float(price_data[ticker].iloc[nearest_idx])
                        
                        sell_value = holding['shares'] * sell_price
                        cash_balance += sell_value
                        
                        trade_history.append({
                            'date': rebal_date,
                            'ticker': ticker,
                            'name': holding['name'],
                            'action': 'SELL',
                            'shares': holding['shares'],
                            'price': sell_price
                        })
                        
                        print(f"  청산: {holding['name']} - {holding['shares']}주 @ {sell_price:,.0f}원")
                        
                    except Exception as e:
                        print(f"  {ticker} 청산 실패: {e}")
                        cash_balance += holding['shares'] * holding['buy_price']
                
                portfolio_value = cash_balance
                holdings = {}
            
            # 구성요소 선택
            selected_components = self.select_components(price_data, benchmark_data, analysis_date)
            
            if not selected_components:
                print("선택된 구성요소가 없습니다.")
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                else:
                    next_date = end_date
                
                dates = pd.date_range(start=rebal_date, end=next_date, freq='D')
                for date in dates:
                    if date <= end_date:
                        portfolio_history.append({
                            'date': date,
                            'value': cash_balance,
                            'holdings': 0
                        })
                continue
            
            print(f"선택된 구성요소 수: {len(selected_components)}")
            for comp in selected_components:
                print(f"  - {comp['name']}: RS-Ratio={comp['rs_ratio']:.2f}, RS-Momentum={comp['rs_momentum']:.2f}")
            
            # 동일 가중 투자
            investment_per_component = cash_balance / len(selected_components)
            
            # 새로운 포지션 구성
            new_holdings = {}
            total_invested = 0
            
            for comp in selected_components:
                ticker = comp['ticker']
                try:
                    if analysis_date in price_data[ticker].index:
                        current_price = float(price_data[ticker].loc[analysis_date])
                    else:
                        nearest_idx = price_data[ticker].index.get_indexer([analysis_date], method='nearest')[0]
                        current_price = float(price_data[ticker].iloc[nearest_idx])
                    
                    if pd.isna(current_price) or current_price <= 0:
                        print(f"{ticker}: 유효하지 않은 가격")
                        continue
                        
                    shares = int(investment_per_component / current_price)
                    if shares > 0:
                        actual_investment = shares * current_price
                        total_invested += actual_investment
                        
                        new_holdings[ticker] = {
                            'shares': shares,
                            'buy_price': current_price,
                            'name': comp['name']
                        }
                        
                        trade_history.append({
                            'date': rebal_date,
                            'ticker': ticker,
                            'name': comp['name'],
                            'action': 'BUY',
                            'shares': shares,
                            'price': current_price
                        })
                        
                        print(f"  매수: {comp['name']} - {shares}주 @ {current_price:,.0f}원")
                    
                except Exception as e:
                    print(f"{ticker} 매수 실패: {e}")
                    continue
            
            holdings = new_holdings
            remaining_cash = cash_balance - total_invested
            
            # 다음 리밸런싱까지 포트폴리오 가치 추적
            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]
            else:
                next_date = end_date
            
            dates = pd.date_range(start=rebal_date, end=next_date, freq='D')
            
            for date in dates:
                if date <= end_date:
                    daily_value = remaining_cash
                    
                    for ticker, holding in holdings.items():
                        try:
                            if date in price_data[ticker].index:
                                current_price = float(price_data[ticker].loc[date])
                            else:
                                available_dates = price_data[ticker].index[price_data[ticker].index <= date]
                                if len(available_dates) > 0:
                                    current_price = float(price_data[ticker].loc[available_dates[-1]])
                                else:
                                    current_price = holding['buy_price']
                            
                            if pd.isna(current_price) or current_price <= 0:
                                current_price = holding['buy_price']
                            
                            position_value = holding['shares'] * current_price
                            daily_value += position_value
                            
                        except Exception as e:
                            position_value = holding['shares'] * holding['buy_price']
                            daily_value += position_value
                    
                    portfolio_value = daily_value
                    
                    portfolio_history.append({
                        'date': date,
                        'value': portfolio_value,
                        'holdings': len(holdings)
                    })
        
        # 결과 정리
        portfolio_df = pd.DataFrame(portfolio_history).drop_duplicates(subset='date').set_index('date')
        trades_df = pd.DataFrame(trade_history) if trade_history else pd.DataFrame()
        
        # 최종 통계 출력
        if not portfolio_df.empty:
            print(f"\n=== 백테스트 완료 ===")
            print(f"시작 가치: {portfolio_df['value'].iloc[0]:,.0f}")
            print(f"종료 가치: {portfolio_df['value'].iloc[-1]:,.0f}")
            print(f"수익률: {(portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0] - 1) * 100:.2f}%")
            print(f"총 거래 횟수: {len(trades_df) if not trades_df.empty else 0}")
        
        return portfolio_df, trades_df
    
    def calculate_performance_metrics(self, portfolio_df):
        """성과 지표 계산"""
        if portfolio_df.empty:
            return {}
        
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        
        total_return = (portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0] - 1) * 100
        
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = (np.power(1 + total_return/100, 1/years) - 1) * 100
        
        annual_volatility = portfolio_df['returns'].std() * np.sqrt(252) * 100
        
        sharpe_ratio = (annual_return - 2) / annual_volatility if annual_volatility > 0 else 0
        
        cumulative_returns = (1 + portfolio_df['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        return {
            '총 수익률': f"{total_return:.2f}%",
            '연율화 수익률': f"{annual_return:.2f}%",
            '연율화 변동성': f"{annual_volatility:.2f}%",
            '샤프 비율': f"{sharpe_ratio:.2f}",
            '최대 낙폭': f"{max_drawdown:.2f}%"
        }