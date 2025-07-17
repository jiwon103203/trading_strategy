import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..core.universal_rs_strategy import UniversalRSStrategy
from ..core.universal_jump_model import UniversalJumpModel
from ..core.utils import safe_float, safe_extract_close, validate_data, calculate_basic_metrics, print_status
import warnings
warnings.filterwarnings('ignore')

# Risk-free rate 유틸리티 import
try:
    from ..advanced.risk_free_rate_utils import RiskFreeRateManager, calculate_dynamic_sharpe_ratio, calculate_dynamic_sortino_ratio
    HAS_RF_UTILS = True
except ImportError:
    print_status("risk_free_rate_utils.py가 없습니다. 기본 risk-free rate (2%) 사용", "WARNING")
    HAS_RF_UTILS = False

class UniversalRSWithJumpModel:
    """
    범용 RS 전략 + Jump Model 통합 (통합된 Jump Model 사용)
    - 시장이 BEAR 체제일 때 모든 투자 중단
    - BULL 체제일 때만 RS 전략 실행
    - Jump Model은 2024년까지만 학습, 2025년은 추론용
    - 동적 Risk-Free Rate (^IRX) 사용한 성과 지표 계산
    - 통합된 특징 계산 코드 사용
    """
    
    def __init__(self, preset_config, rs_length=20, rs_timeframe='daily', 
                 rs_recent_cross_days=None, jump_penalty=50.0, 
                 regime_lookback=20, use_jump_model=True,
                 training_cutoff_date=None, rf_ticker='^IRX', default_rf_rate=0.02,
                 use_paper_features_only=True):
        """
        Parameters:
        - preset_config: 프리셋 설정 딕셔너리 (benchmark, components, name)
        - rs_length: RS 계산 기간
        - rs_timeframe: RS 계산 주기
        - rs_recent_cross_days: 최근 크로스 필터링 기간
        - jump_penalty: Jump Model의 체제 전환 페널티 (기본: 50.0)
        - regime_lookback: 체제 판단을 위한 lookback 기간
        - use_jump_model: Jump Model 사용 여부
        - training_cutoff_date: Jump Model 학습 마감일 (None이면 2024-12-31)
        - rf_ticker: Risk-free rate 티커 (기본: ^IRX)
        - default_rf_rate: 기본 risk-free rate (기본: 2%)
        - use_paper_features_only: 논문 정확한 3특징만 사용 여부 (기본: True)
        """
        # RS 전략 초기화 (동적 Risk-Free Rate 지원)
        self.rs_strategy = UniversalRSStrategy(
            benchmark=preset_config['benchmark'],
            components=preset_config['components'],
            name=preset_config['name'],
            length=rs_length,
            timeframe=rs_timeframe
        )
        
        # Jump Model 사용 여부
        self.use_jump_model = use_jump_model
        
        # 통합 모델 설정
        self.use_paper_features_only = use_paper_features_only
        self.jump_penalty = jump_penalty
        
        # Risk-Free Rate 설정
        self.rf_ticker = rf_ticker
        self.default_rf_rate = default_rf_rate
        
        # Risk-free rate 관리자 초기화
        if HAS_RF_UTILS:
            self.rf_manager = RiskFreeRateManager(rf_ticker, default_rf_rate)
        else:
            self.rf_manager = None
        
        # Training cutoff 설정
        if training_cutoff_date is None:
            self.training_cutoff_date = datetime(2024, 12, 31)
        else:
            self.training_cutoff_date = training_cutoff_date
        
        if self.use_jump_model:
            # 통합된 Jump Model 초기화
            self.jump_model = UniversalJumpModel(
                benchmark_ticker=preset_config['benchmark'],
                benchmark_name=preset_config.get('name', 'Market'),
                jump_penalty=self.jump_penalty,
                training_cutoff_date=self.training_cutoff_date
            )
            print_status(f"통합된 Jump Model 초기화:")
            print_status(f"  - Feature Type: {'논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가'}")
            print_status(f"  - Jump Penalty: {self.jump_penalty}")
            print_status(f"  - 학습 마감일: {self.training_cutoff_date.strftime('%Y-%m-%d')}")
        else:
            self.jump_model = None
        
        self.regime_history = None
        self.preset_config = preset_config
        
        print_status(f"통합 전략 초기화: Risk-Free Rate = {self.rf_ticker}")
        
    def prepare_regime_data(self, start_date, end_date):
        """백테스트를 위한 체제 데이터 준비 - 통합 모델 사용"""
        if not self.use_jump_model:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            self.regime_history = pd.DataFrame({
                'state': 0,
                'regime': 'BULL'
            }, index=dates)
            print_status("Jump Model 비활성화 - 항상 BULL 체제")
            return self.regime_history
        
        # 체제 이력 계산 (통합 모델 + training cutoff 고려)
        print_status(f"{self.preset_config['name']} 체제 분석 중... (통합 모델)")
        print_status(f"Feature Type: {'논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가'}")
        print_status(f"Jump Penalty: {self.jump_penalty}")
        print_status(f"학습 마감일: {self.training_cutoff_date.strftime('%Y-%m-%d')}")
        
        # 현재 체제만 분석 (간단한 버전)
        current_regime = self.jump_model.get_current_regime_with_training_cutoff()
        
        if current_regime:
            # 간단한 체제 데이터 생성
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            regime_value = current_regime['regime']
            
            self.regime_history = pd.DataFrame({
                'state': 0 if regime_value == 'BULL' else 1,
                'regime': regime_value
            }, index=dates)
            
            # 체제별 통계
            bull_pct = (self.regime_history['regime'] == 'BULL').mean() * 100
            bear_pct = (self.regime_history['regime'] == 'BEAR').mean() * 100
            
            print_status(f"체제 분포 (통합 모델): BULL {bull_pct:.1f}%, BEAR {bear_pct:.1f}%")
        
        return self.regime_history
    
    def get_regime_on_date(self, date):
        """특정 날짜의 체제 확인"""
        if not self.use_jump_model:
            return 'BULL'
        
        if self.regime_history is None:
            return 'BULL'
        
        try:
            if date in self.regime_history.index:
                return self.regime_history.loc[date, 'regime']
            else:
                prev_dates = self.regime_history.index[self.regime_history.index <= date]
                if len(prev_dates) > 0:
                    return self.regime_history.loc[prev_dates[-1], 'regime']
                else:
                    return 'BULL'
        except:
            return 'BULL'
    
    def backtest(self, start_date, end_date, initial_capital=10000000):
        """Jump Model을 적용한 백테스트 (통합 모델 + Training Cutoff + 동적 Risk-Free Rate)"""
        
        # Jump Model 비활성화시 기본 RS 전략 실행
        if not self.use_jump_model:
            print_status("Jump Model 비활성화 - 기본 RS 전략 실행")
            portfolio_df, trades_df = self.rs_strategy.backtest(start_date, end_date, initial_capital)
            
            if portfolio_df is not None and not portfolio_df.empty:
                portfolio_df['regime'] = 'BULL'
            
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
            regime_df = pd.DataFrame({
                'regime': 'BULL'
            }, index=rebalance_dates)
            
            return portfolio_df, trades_df, regime_df
        
        # 1. 체제 데이터 준비 (통합 모델 사용)
        print_status(f"통합 모델 백테스트 시작:")
        print_status(f"  - Feature Type: {'논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가'}")
        print_status(f"  - Jump Penalty: {self.jump_penalty}")
        print_status(f"  - Training cutoff: {self.training_cutoff_date.strftime('%Y-%m-%d')}")
        print_status(f"  - Risk-Free Rate: {self.rf_ticker}")
        
        self.prepare_regime_data(start_date, end_date)
        
        # 2. RS 전략용 데이터 준비
        extra_days = 200 if self.rs_strategy.timeframe == 'weekly' else 100
        price_data, benchmark_data = self.rs_strategy.get_price_data(
            start_date - timedelta(days=extra_days),
            end_date
        )
        
        if not validate_data(price_data) or not validate_data(benchmark_data):
            print_status("데이터 다운로드 실패", "ERROR")
            return None, None, None
        
        # 3. 리밸런싱 날짜 생성
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # 4. 포트폴리오 초기화
        portfolio_value = initial_capital
        portfolio_history = []
        holdings = {}
        trade_history = []
        regime_log = []
        
        for i, rebal_date in enumerate(rebalance_dates):
            print_status(f"{rebal_date.strftime('%Y-%m-%d')} 리밸런싱 (통합 모델)")
            
            # 현재 체제 확인
            current_regime = self.get_regime_on_date(rebal_date)
            
            # Out-of-sample 여부 확인
            is_oos = rebal_date > self.training_cutoff_date
            oos_indicator = " (Out-of-Sample)" if is_oos else " (In-Sample)"
            
            print_status(f"현재 시장 체제: {current_regime}{oos_indicator}")
            
            regime_log.append({
                'date': rebal_date,
                'regime': current_regime,
                'is_out_of_sample': is_oos,
                'unified_model_used': True,
                'feature_type': '논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가',
                'jump_penalty': self.jump_penalty
            })
            
            # BEAR 체제인 경우 모든 포지션 청산
            if current_regime == 'BEAR':
                print_status("BEAR 체제 - 모든 투자 중단 (통합 모델)")
                
                # 기존 포지션 청산
                if holdings:
                    for ticker, holding in holdings.items():
                        try:
                            if rebal_date in price_data[ticker].index:
                                exit_price = price_data[ticker].loc[rebal_date]
                            else:
                                exit_price = holding['buy_price']
                            
                            trade_history.append({
                                'date': rebal_date,
                                'ticker': ticker,
                                'name': holding['name'],
                                'action': 'SELL_BEAR',
                                'shares': holding['shares'],
                                'price': safe_float(exit_price),
                                'unified_model': True
                            })
                        except:
                            pass
                    holdings = {}
                
                # 다음 리밸런싱까지 현금 보유
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                else:
                    next_date = end_date
                
                dates = pd.date_range(start=rebal_date, end=next_date, freq='D')
                for date in dates:
                    if date <= end_date:
                        date_is_oos = date > self.training_cutoff_date
                        portfolio_history.append({
                            'date': date,
                            'value': portfolio_value,
                            'holdings': 0,
                            'regime': current_regime,
                            'is_out_of_sample': date_is_oos,
                            'unified_model_used': True
                        })
                continue
            
            # BULL 체제인 경우 RS 전략 실행
            analysis_date = rebal_date
            if self.rs_strategy.timeframe == 'weekly':
                days_ahead = 4 - rebal_date.weekday()
                if days_ahead < 0:
                    days_ahead += 7
                analysis_date = rebal_date + timedelta(days=days_ahead)
            
            # 구성요소 선택 (RS 전략)
            selected_components = self.rs_strategy.select_components(price_data, benchmark_data, analysis_date)
            
            if not selected_components:
                print_status("선택된 구성요소가 없습니다.")
                # 현금 보유
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                else:
                    next_date = end_date
                
                dates = pd.date_range(start=rebal_date, end=next_date, freq='D')
                for date in dates:
                    if date <= end_date:
                        date_is_oos = date > self.training_cutoff_date
                        portfolio_history.append({
                            'date': date,
                            'value': portfolio_value,
                            'holdings': 0,
                            'regime': current_regime,
                            'is_out_of_sample': date_is_oos,
                            'unified_model_used': True
                        })
                continue
            
            # 선택된 구성요소 출력
            print_status(f"선택된 구성요소 수: {len(selected_components)}")
            for comp in selected_components:
                print_status(f"  - {comp['name']}: RS-Ratio={comp['rs_ratio']:.2f}, RS-Momentum={comp['rs_momentum']:.2f}")
            
            # 동일 가중 투자
            investment_per_component = portfolio_value / len(selected_components)
            
            # 새로운 포지션 구성
            new_holdings = {}
            for comp in selected_components:
                ticker = comp['ticker']
                try:
                    if analysis_date in price_data[ticker].index:
                        current_price = price_data[ticker].loc[analysis_date]
                    else:
                        nearest_date = price_data[ticker].index[price_data[ticker].index.get_indexer([analysis_date], method='nearest')[0]]
                        current_price = price_data[ticker].loc[nearest_date]
                    
                    current_price = safe_float(current_price)
                    if current_price <= 0:
                        continue
                    
                    shares = int(investment_per_component / current_price)
                    if shares > 0:
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
                            'price': current_price,
                            'unified_model': True
                        })
                
                except Exception as e:
                    print_status(f"{ticker} 매수 실패: {e}", "ERROR")
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
                    daily_regime = self.get_regime_on_date(date) if self.use_jump_model else 'BULL'
                    date_is_oos = date > self.training_cutoff_date
                    
                    # BEAR 체제로 전환된 경우 즉시 청산
                    if self.use_jump_model and daily_regime == 'BEAR' and holdings:
                        print_status(f"{date.strftime('%Y-%m-%d')} BEAR 체제 감지 - 긴급 청산 (통합 모델)")
                        
                        portfolio_value = 0
                        for ticker, holding in holdings.items():
                            try:
                                if date in price_data[ticker].index:
                                    exit_price = price_data[ticker].loc[date]
                                else:
                                    exit_price = holding['buy_price']
                                
                                portfolio_value += holding['shares'] * safe_float(exit_price)
                                
                                trade_history.append({
                                    'date': date,
                                    'ticker': ticker,
                                    'name': holding['name'],
                                    'action': 'SELL_BEAR_EMERGENCY',
                                    'shares': holding['shares'],
                                    'price': safe_float(exit_price),
                                    'unified_model': True
                                })
                            except:
                                portfolio_value += holding['shares'] * holding['buy_price']
                        
                        holdings = {}
                    
                    # 포트폴리오 가치 계산
                    elif holdings:
                        daily_value = 0
                        for ticker, holding in holdings.items():
                            try:
                                if date in price_data[ticker].index:
                                    current_price = price_data[ticker].loc[date]
                                else:
                                    available_dates = price_data[ticker].index[price_data[ticker].index <= date]
                                    if len(available_dates) > 0:
                                        current_price = price_data[ticker].loc[available_dates[-1]]
                                    else:
                                        current_price = holding['buy_price']
                                
                                daily_value += holding['shares'] * safe_float(current_price)
                            except:
                                pass
                        
                        if daily_value > 0:
                            portfolio_value = daily_value
                    
                    portfolio_history.append({
                        'date': date,
                        'value': portfolio_value,
                        'holdings': len(holdings),
                        'regime': daily_regime,
                        'is_out_of_sample': date_is_oos,
                        'unified_model_used': True
                    })
        
        # 결과 정리
        portfolio_df = pd.DataFrame(portfolio_history).drop_duplicates(subset='date').set_index('date')
        trades_df = pd.DataFrame(trade_history)
        regime_df = pd.DataFrame(regime_log).set_index('date')
        
        # 통합 모델 사용 통계 출력
        if self.use_jump_model and not portfolio_df.empty:
            oos_df = portfolio_df[portfolio_df['is_out_of_sample'] == True]
            if not oos_df.empty:
                print_status(f"통합 모델 백테스트 완료:", "SUCCESS")
                print_status(f"  - Out-of-sample 기간: {len(oos_df)}일")
                print_status(f"  - Out-of-sample 비율: {len(oos_df) / len(portfolio_df) * 100:.1f}%")
                print_status(f"  - Feature Type: {'논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가'}")
                print_status(f"  - Jump Penalty: {self.jump_penalty}")
        
        return portfolio_df, trades_df, regime_df
    
    def calculate_performance_metrics(self, portfolio_df):
        """성과 지표 계산 (통합 모델 정보 + Out-of-sample + 동적 Risk-Free Rate 포함)"""
        if portfolio_df.empty:
            return {}
        
        print_status(f"성과 지표 계산 중... (통합 모델 + Risk-Free Rate: {self.rf_ticker})")
        
        # 기본 성과 지표 계산 (utils 사용)
        metrics = calculate_basic_metrics(portfolio_df)
        
        # 동적 Risk-Free Rate를 사용한 성과 지표
        if HAS_RF_UTILS and self.rf_manager:
            try:
                # Risk-free rate 다운로드
                start_date = portfolio_df.index[0]
                end_date = portfolio_df.index[-1]
                self.rf_manager.download_risk_free_rate(start_date, end_date)
                
                # 수익률 계산
                returns = portfolio_df['value'].pct_change().dropna()
                
                # 동적 Sharpe ratio
                sharpe_ratio = self.rf_manager.calculate_sharpe_ratio(returns, portfolio_df.index)
                
                # 동적 Sortino ratio
                sortino_ratio = self.rf_manager.calculate_sortino_ratio(returns, portfolio_df.index)
                
                # Risk-free rate 통계
                rf_stats = self.rf_manager.get_risk_free_rate_stats(start_date, end_date)
                
                print_status(f"평균 Risk-Free Rate: {rf_stats['mean_rate']:.3f}%")
                print_status(f"Sharpe Ratio (동적): {sharpe_ratio:.3f}")
                print_status(f"Sortino Ratio (동적): {sortino_ratio:.3f}")
                
                # 메트릭스에 추가
                metrics.update({
                    '샤프 비율 (동적)': f"{sharpe_ratio:.3f}",
                    '소르티노 비율 (동적)': f"{sortino_ratio:.3f}" if sortino_ratio != float('inf') else "∞",
                    '평균 Risk-Free Rate': f"{rf_stats['mean_rate']:.3f}%",
                    'Risk-Free Rate 티커': self.rf_ticker,
                    'Risk-Free Rate 범위': f"{rf_stats['min_rate']:.3f}% ~ {rf_stats['max_rate']:.3f}%"
                })
                
            except Exception as e:
                print_status(f"동적 성과 지표 계산 실패: {e}", "WARNING")
                # 기본 방식으로 fallback
                metrics['Risk-Free Rate'] = f"{self.default_rf_rate*100:.1f}% (기본값)"
        else:
            # 기본 방식 (2% 고정)
            metrics['Risk-Free Rate'] = f"{self.default_rf_rate*100:.1f}% (기본값)"
        
        # 통합 모델 정보 추가
        if self.use_jump_model:
            metrics['통합 모델 사용'] = '예'
            metrics['Feature Type'] = '논문 정확한 3특징' if self.use_paper_features_only else '논문 기반 + 추가'
            metrics['Jump Penalty'] = f"{self.jump_penalty}"
            metrics['Training Cutoff'] = self.training_cutoff_date.strftime('%Y-%m-%d')
            
            # Out-of-sample 기간 분석
            if 'is_out_of_sample' in portfolio_df.columns:
                oos_df = portfolio_df[portfolio_df['is_out_of_sample'] == True]
                in_sample_df = portfolio_df[portfolio_df['is_out_of_sample'] == False]
                
                if not oos_df.empty:
                    oos_days = len(oos_df)
                    total_days = len(portfolio_df)
                    metrics['Out-of-Sample Days'] = f"{oos_days}일 ({oos_days/total_days*100:.1f}%)"
                    
                    if len(oos_df) > 1:
                        oos_return = (oos_df['value'].iloc[-1] / oos_df['value'].iloc[0] - 1) * 100
                        metrics['Out-of-Sample Return'] = f"{oos_return:.2f}%"
                        
                        # Out-of-sample 기간의 동적 성과 지표
                        if HAS_RF_UTILS and self.rf_manager and len(oos_df) > 10:
                            try:
                                oos_returns = oos_df['value'].pct_change().dropna()
                                oos_sharpe = self.rf_manager.calculate_sharpe_ratio(oos_returns, oos_df.index)
                                oos_sortino = self.rf_manager.calculate_sortino_ratio(oos_returns, oos_df.index)
                                
                                metrics['Out-of-Sample Sharpe (동적)'] = f"{oos_sharpe:.3f}"
                                metrics['Out-of-Sample Sortino (동적)'] = f"{oos_sortino:.3f}" if oos_sortino != float('inf') else "∞"
                            except Exception as e:
                                print_status(f"Out-of-sample 성과 지표 계산 실패: {e}", "WARNING")
                
                if not in_sample_df.empty:
                    in_sample_days = len(in_sample_df)
                    metrics['In-Sample Days'] = f"{in_sample_days}일 ({in_sample_days/total_days*100:.1f}%)"
        else:
            metrics['통합 모델 사용'] = '아니오'
        
        # 체제별 성과 분석
        if self.use_jump_model and 'regime' in portfolio_df.columns:
            bull_df = portfolio_df[portfolio_df['regime'] == 'BULL']
            bear_df = portfolio_df[portfolio_df['regime'] == 'BEAR']
            
            if not bull_df.empty:
                bull_days = len(bull_df)
                bull_return = (bull_df['value'].iloc[-1] / bull_df['value'].iloc[0] - 1) * 100 if len(bull_df) > 1 else 0
                metrics['BULL 기간'] = f"{bull_days}일 ({bull_days/len(portfolio_df)*100:.1f}%)"
                metrics['BULL 수익률'] = f"{bull_return:.2f}%"
                
                # BULL 기간의 동적 성과 지표
                if HAS_RF_UTILS and self.rf_manager and len(bull_df) > 10:
                    try:
                        bull_returns = bull_df['value'].pct_change().dropna()
                        bull_sharpe = self.rf_manager.calculate_sharpe_ratio(bull_returns, bull_df.index)
                        metrics['BULL Sharpe (동적)'] = f"{bull_sharpe:.3f}"
                    except:
                        pass
            
            if not bear_df.empty:
                bear_days = len(bear_df)
                bear_return = (bear_df['value'].iloc[-1] / bear_df['value'].iloc[0] - 1) * 100 if len(bear_df) > 1 else 0
                metrics['BEAR 기간'] = f"{bear_days}일 ({bear_days/len(portfolio_df)*100:.1f}%)"
                metrics['BEAR 수익률'] = f"{bear_return:.2f}%"
        
        return metrics


# 편의 함수들 - 통합 모델 사용
def create_integrated_strategy_with_unified_model(preset_config, use_jump_model=True, 
                                                use_paper_features_only=True, jump_penalty=50.0,
                                                rf_ticker='^IRX', default_rf_rate=0.02, **kwargs):
    """통합 모델을 사용하는 통합 전략 생성 편의 함수"""
    return UniversalRSWithJumpModel(
        preset_config=preset_config,
        use_jump_model=use_jump_model,
        use_paper_features_only=use_paper_features_only,
        jump_penalty=jump_penalty,
        rf_ticker=rf_ticker,
        default_rf_rate=default_rf_rate,
        **kwargs
    )

def compare_unified_model_strategies(preset1, preset2, years=3, rf_ticker='^IRX', 
                                   use_paper_features_only=True, jump_penalty=50.0):
    """통합 모델을 사용한 전략 비교"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    
    results = {}
    
    for i, preset in enumerate([preset1, preset2], 1):
        print_status(f"전략 {i}: {preset['name']} (통합 모델)")
        
        strategy = UniversalRSWithJumpModel(
            preset_config=preset,
            use_jump_model=True,
            use_paper_features_only=use_paper_features_only,
            jump_penalty=jump_penalty,
            rf_ticker=rf_ticker
        )
        
        portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)
        
        if portfolio_df is not None and not portfolio_df.empty:
            metrics = strategy.calculate_performance_metrics(portfolio_df)
            results[preset['name']] = metrics
    
    # 결과 비교 출력
    if len(results) == 2:
        print_status(f"전략 비교 결과 (통합 모델 + 동적 Risk-Free Rate: {rf_ticker})", "SUCCESS")
        strategies = list(results.keys())
        
        print(f"{'지표':<30} {strategies[0]:<25} {strategies[1]:<25}")
        print("-" * 80)
        
        compare_metrics = ['총 수익률', '연율화 수익률', '샤프 비율 (동적)', '소르티노 비율 (동적)', 
                          'Feature Type', 'Jump Penalty', '평균 Risk-Free Rate']
        
        for metric in compare_metrics:
            val1 = results[strategies[0]].get(metric, 'N/A')
            val2 = results[strategies[1]].get(metric, 'N/A')
            print(f"{metric:<30} {str(val1):<25} {str(val2):<25}")
    
    return results


# 사용 예시
if __name__ == "__main__":
    from preset_manager import PresetManager
    
    # S&P 500 섹터 전략 (통합 모델 사용)
    sp500_preset = PresetManager.get_sp500_sectors()
    
    strategy = UniversalRSWithJumpModel(
        preset_config=sp500_preset,
        use_jump_model=True,
        use_paper_features_only=True,
        jump_penalty=50.0,
        rf_ticker='^IRX',
        default_rf_rate=0.02,
        training_cutoff_date=datetime(2024, 12, 31)
    )
    
    # 백테스트
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3년
    
    print_status(f"백테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print_status(f"통합 모델 설정:")
    print_status(f"  - Feature Type: 논문 정확한 3특징")
    print_status(f"  - Jump Penalty: 50.0")
    print_status(f"  - Training Cutoff: 2024-12-31")
    print_status(f"  - Risk-Free Rate: ^IRX (미국 3개월물 국채)")
    
    portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)
    
    if portfolio_df is not None and not portfolio_df.empty:
        metrics = strategy.calculate_performance_metrics(portfolio_df)
        
        print_status(f"{sp500_preset['name']} 성과 결과 (통합 모델)", "SUCCESS")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # 추가 분석
        if HAS_RF_UTILS:
            print_status(f"동적 Risk-Free Rate 추가 분석", "SUCCESS")
            quick_sharpe = calculate_dynamic_sharpe_ratio(portfolio_df, '^IRX')
            quick_sortino = calculate_dynamic_sortino_ratio(portfolio_df, '^IRX')
            
            print(f"빠른 Sharpe 계산: {quick_sharpe:.3f}")
            print(f"빠른 Sortino 계산: {quick_sortino:.3f}")
    
    print_status(f"통합 모델 + 동적 Risk-Free Rate 시스템 테스트 완료!", "SUCCESS")
