"""
최적화된 통합 전략
RS + Jump Model 통합 전략
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from optimized_rs_strategy import OptimizedRSStrategy
from optimized_jump_model import OptimizedJumpModel
from base_strategy import BaseStrategy
from config_manager import config_manager
from logger import get_logger
from memory_optimization import memory_monitor

class OptimizedIntegratedStrategy(BaseStrategy):
    """최적화된 RS + Jump Model 통합 전략"""
    
    def __init__(self, preset_config: Dict[str, Any],
                 rs_length: int = 20,
                 rs_timeframe: str = 'daily',
                 rs_recent_cross_days: Optional[int] = None,
                 use_jump_model: bool = True,
                 jump_penalty: float = 50.0,
                 regime_lookback: int = 20,
                 training_cutoff_date: Optional[datetime] = None):
        """
        Parameters:
        - preset_config: 프리셋 설정
        - rs_length: RS 계산 기간
        - rs_timeframe: RS 시간 프레임
        - rs_recent_cross_days: 최근 크로스 필터링 기간
        - use_jump_model: Jump Model 사용 여부
        - jump_penalty: 체제 전환 페널티
        - regime_lookback: 체제 판단 lookback
        - training_cutoff_date: Jump Model 학습 마감일
        """
        # 설정 통합
        config = preset_config.copy()
        config.update({
            'rs_length': rs_length,
            'rs_timeframe': rs_timeframe,
            'rs_recent_cross_days': rs_recent_cross_days,
            'use_jump_model': use_jump_model,
            'jump_penalty': jump_penalty,
            'regime_lookback': regime_lookback
        })
        
        super().__init__(config)
        
        # RS 전략 초기화
        self.rs_strategy = OptimizedRSStrategy(
            preset_config=preset_config,
            rs_length=rs_length,
            rs_timeframe=rs_timeframe,
            rs_recent_cross_days=rs_recent_cross_days,
            use_parallel=True
        )
        
        # Jump Model 사용 설정
        self.use_jump_model = use_jump_model
        
        if self.use_jump_model:
            # Jump Model 초기화
            if training_cutoff_date is None:
                self.training_cutoff_date = datetime(2024, 12, 31)
            else:
                self.training_cutoff_date = training_cutoff_date
            
            self.jump_model = OptimizedJumpModel(
                benchmark_ticker=preset_config['benchmark'],
                benchmark_name=preset_config.get('name', 'Market'),
                n_states=2,
                jump_penalty=jump_penalty,
                training_cutoff_date=self.training_cutoff_date,
                cache_features=True
            )
            
            self.logger.info(f"Jump Model 활성화 (학습 마감: {self.training_cutoff_date.strftime('%Y-%m-%d')})")
        else:
            self.jump_model = None
            self.logger.info("Jump Model 비활성화 - RS 전략만 사용")
        
        # 체제 이력
        self.regime_history = None
    
    def _validate_strategy_config(self):
        """전략 설정 검증"""
        # RS 전략이 자체적으로 검증 수행
        pass
    
    def prepare_regime_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """백테스트를 위한 체제 데이터 준비"""
        if not self.use_jump_model:
            # Jump Model 비활성화시 항상 BULL
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            self.regime_history = pd.DataFrame({
                'state': 0,
                'regime': 'BULL',
                'confidence': 1.0
            }, index=dates)
            self.logger.info("Jump Model 비활성화 - 항상 BULL 체제")
            return self.regime_history
        
        # Jump Model을 사용한 체제 분석
        self.logger.info(f"체제 분석 중: {self.benchmark_name}")
        
        # 체제 이력 가져오기
        regime_history = self.jump_model.get_regime_history(start_date, end_date)
        
        if regime_history is None:
            self.logger.warning("체제 분석 실패 - 기본값(BULL) 사용")
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            regime_history = pd.DataFrame({
                'state': 0,
                'regime': 'BULL',
                'confidence': 0.5
            }, index=dates)
        
        self.regime_history = regime_history
        
        # 체제 통계
        bull_pct = (regime_history['regime'] == 'BULL').mean() * 100
        bear_pct = (regime_history['regime'] == 'BEAR').mean() * 100
        
        self.logger.info(f"체제 분포: BULL {bull_pct:.1f}%, BEAR {bear_pct:.1f}%")
        
        # 체제 전환 횟수
        regime_changes = regime_history['regime'] != regime_history['regime'].shift()
        n_changes = regime_changes.sum() - 1
        
        self.logger.info(f"체제 전환 횟수: {n_changes}회")
        
        # Out-of-sample 기간 통계
        if self.use_jump_model and hasattr(self, 'training_cutoff_date'):
            oos_start = self.training_cutoff_date + timedelta(days=1)
            if end_date > self.training_cutoff_date:
                oos_regime = regime_history[oos_start:]
                if not oos_regime.empty:
                    oos_bull_pct = (oos_regime['regime'] == 'BULL').mean() * 100
                    self.logger.info(f"Out-of-sample 기간 BULL 비율: {oos_bull_pct:.1f}%")
        
        return regime_history
    
    def get_regime_on_date(self, date: datetime) -> str:
        """특정 날짜의 체제 확인"""
        if not self.use_jump_model:
            return 'BULL'
        
        if self.regime_history is None:
            return 'BULL'
        
        try:
            if date in self.regime_history.index:
                return self.regime_history.loc[date, 'regime']
            else:
                # 가장 가까운 이전 날짜의 체제 사용
                prev_dates = self.regime_history.index[self.regime_history.index <= date]
                if len(prev_dates) > 0:
                    return self.regime_history.loc[prev_dates[-1], 'regime']
                else:
                    return 'BULL'
        except:
            return 'BULL'
    
    def select_components(self, price_data: Dict[str, pd.Series], 
                         benchmark_data: pd.Series, 
                         date: datetime) -> List[Dict[str, Any]]:
        """구성요소 선택 (체제 기반)"""
        # 현재 체제 확인
        current_regime = self.get_regime_on_date(date)
        
        # Out-of-sample 여부 확인
        is_oos = False
        if self.use_jump_model and hasattr(self, 'training_cutoff_date'):
            is_oos = date > self.training_cutoff_date
        
        oos_indicator = " (Out-of-Sample)" if is_oos else " (In-Sample)"
        self.logger.debug(f"{date.strftime('%Y-%m-%d')} - 체제: {current_regime}{oos_indicator}")
        
        # BEAR 체제인 경우 빈 리스트 반환
        if current_regime == 'BEAR':
            self.logger.debug("BEAR 체제 - 투자 중단")
            return []
        
        # BULL 체제인 경우 RS 전략 실행
        return self.rs_strategy.select_components(price_data, benchmark_data, date)
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """신호 계산"""
        # RS 신호 계산
        rs_signals = self.rs_strategy.calculate_signals(data)
        
        if not self.use_jump_model:
            return rs_signals
        
        # 체제 신호 추가
        if not rs_signals.empty and self.regime_history is not None:
            # 인덱스 정렬
            common_dates = rs_signals.index.intersection(self.regime_history.index)
            if len(common_dates) > 0:
                rs_signals.loc[common_dates, 'regime'] = self.regime_history.loc[common_dates, 'regime']
                rs_signals.loc[common_dates, 'regime_confidence'] = self.regime_history.loc[common_dates, 'confidence']
                
                # 최종 신호 = RS 신호 AND BULL 체제
                rs_signals['final_signal'] = (
                    (rs_signals['signal'] == 1) & 
                    (rs_signals['regime'] == 'BULL')
                ).astype(int)
        
        return rs_signals
    
    def backtest(self, start_date: datetime, end_date: datetime, 
                 initial_capital: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """백테스트 실행 (체제 기반)"""
        if initial_capital is None:
            initial_capital = self.global_config.initial_capital
        
        self.logger.info(f"통합 백테스트 시작: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        with memory_monitor.memory_tracker("Integrated Backtest"):
            # 체제 데이터 준비
            self.prepare_regime_data(start_date, end_date)
            
            # 데이터 준비
            extra_days = 200 if self.global_config.rs_timeframe == 'weekly' else 100
            extended_start = start_date - timedelta(days=extra_days)
            
            benchmark_data, price_data = self.download_data(extended_start, end_date)
            
            # 리밸런싱 날짜
            rebalance_dates = pd.date_range(
                start=start_date, 
                end=end_date, 
                freq=self.global_config.rebalance_frequency
            )
            
            # 백테스트 실행 (체제 고려)
            portfolio_df, trades_df = self._run_backtest_with_regime(
                price_data, benchmark_data, rebalance_dates, initial_capital
            )
            
            # 성과 지표 계산
            if not portfolio_df.empty:
                metrics = self.calculate_performance_metrics_with_regime(portfolio_df)
                self.logger.log_backtest_summary(start_date, end_date, metrics)
            else:
                metrics = {}
            
            return portfolio_df, trades_df, metrics
    
    def _run_backtest_with_regime(self, price_data: Dict[str, pd.Series], 
                                 benchmark_data: pd.Series,
                                 rebalance_dates: pd.DatetimeIndex, 
                                 initial_capital: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """체제를 고려한 백테스트 실행"""
        portfolio_value = initial_capital
        cash = initial_capital
        holdings = {}
        portfolio_history = []
        trade_history = []
        
        # 거래 비용
        commission_rate = self.global_config.transaction_cost
        slippage_rate = self.global_config.slippage
        
        for i, rebal_date in enumerate(rebalance_dates):
            self.logger.debug(f"리밸런싱: {rebal_date.strftime('%Y-%m-%d')}")
            
            # 현재 체제 확인
            current_regime = self.get_regime_on_date(rebal_date)
            
            # 이전 보유 종목 가치 평가
            if holdings:
                portfolio_value = cash
                for ticker, position in holdings.items():
                    if ticker in price_data and rebal_date in price_data[ticker].index:
                        current_price = price_data[ticker].loc[rebal_date]
                        portfolio_value += position['shares'] * current_price
            
            # BEAR 체제인 경우 모든 포지션 청산
            if current_regime == 'BEAR':
                self.logger.debug("BEAR 체제 - 모든 포지션 청산")
                
                if holdings:
                    for ticker, position in holdings.items():
                        if ticker in price_data and rebal_date in price_data[ticker].index:
                            exit_price = price_data[ticker].loc[rebal_date]
                            exit_value = position['shares'] * exit_price * (1 - slippage_rate)
                            cash += exit_value * (1 - commission_rate)
                            
                            trade_history.append({
                                'date': rebal_date,
                                'ticker': ticker,
                                'name': position['name'],
                                'action': 'SELL_BEAR',
                                'shares': position['shares'],
                                'price': exit_price,
                                'value': exit_value,
                                'regime': current_regime
                            })
                            
                            self.logger.log_trade({
                                'action': 'SELL_BEAR',
                                'ticker': ticker,
                                'price': exit_price,
                                'shares': position['shares']
                            })
                    
                    holdings = {}
                
                portfolio_value = cash
            
            else:  # BULL 체제
                # 구성요소 선택
                selected = self.select_components(price_data, benchmark_data, rebal_date)
                
                if selected:
                    # 포트폴리오 리밸런싱
                    new_holdings = self._rebalance_portfolio(
                        selected, holdings, price_data, rebal_date, 
                        portfolio_value, cash, commission_rate, slippage_rate
                    )
                    
                    # 거래 기록
                    for trade in new_holdings['trades']:
                        trade['regime'] = current_regime
                        trade_history.append(trade)
                        self.logger.log_trade(trade)
                    
                    holdings = new_holdings['holdings']
                    cash = new_holdings['cash']
                else:
                    # 선택된 구성요소가 없으면 현금 보유
                    if holdings:
                        # 기존 포지션 청산
                        for ticker, position in holdings.items():
                            if ticker in price_data and rebal_date in price_data[ticker].index:
                                exit_price = price_data[ticker].loc[rebal_date]
                                exit_value = position['shares'] * exit_price * (1 - slippage_rate)
                                cash += exit_value * (1 - commission_rate)
                                
                                trade_history.append({
                                    'date': rebal_date,
                                    'ticker': ticker,
                                    'name': position['name'],
                                    'action': 'SELL_NO_SIGNAL',
                                    'shares': position['shares'],
                                    'price': exit_price,
                                    'value': exit_value,
                                    'regime': current_regime
                                })
                        holdings = {}
                    
                    portfolio_value = cash
            
            # 일별 포트폴리오 추적
            next_date = rebalance_dates[i + 1] if i < len(rebalance_dates) - 1 else end_date
            
            for date in pd.date_range(start=rebal_date, end=next_date, freq='D'):
                if date <= end_date:
                    # 일별 체제 확인 (체제 중간 변경 감지)
                    daily_regime = self.get_regime_on_date(date)
                    
                    # 체제가 BEAR로 변경된 경우 즉시 청산
                    if self.use_jump_model and daily_regime == 'BEAR' and holdings:
                        self.logger.warning(f"{date.strftime('%Y-%m-%d')} BEAR 체제 감지 - 긴급 청산")
                        
                        portfolio_value = cash
                        for ticker, position in holdings.items():
                            if ticker in price_data and date in price_data[ticker].index:
                                exit_price = price_data[ticker].loc[date]
                                exit_value = position['shares'] * exit_price * (1 - slippage_rate)
                                cash += exit_value * (1 - commission_rate)
                                portfolio_value = cash
                                
                                trade_history.append({
                                    'date': date,
                                    'ticker': ticker,
                                    'name': position['name'],
                                    'action': 'SELL_BEAR_EMERGENCY',
                                    'shares': position['shares'],
                                    'price': exit_price,
                                    'value': exit_value,
                                    'regime': daily_regime
                                })
                                
                                self.logger.log_regime_change('BULL', 'BEAR', 0.0)
                        
                        holdings = {}
                    
                    # 포트폴리오 가치 계산
                    elif holdings:
                        daily_value = cash
                        for ticker, position in holdings.items():
                            if ticker in price_data and date in price_data[ticker].index:
                                current_price = price_data[ticker].loc[date]
                                daily_value += position['shares'] * current_price
                            else:
                                daily_value += position['shares'] * position['last_price']
                        portfolio_value = daily_value
                    else:
                        portfolio_value = cash
                    
                    # 기록
                    portfolio_history.append({
                        'date': date,
                        'value': portfolio_value,
                        'cash': cash,
                        'holdings': len(holdings),
                        'invested': portfolio_value - cash,
                        'regime': daily_regime
                    })
        
        # 결과 정리
        portfolio_df = pd.DataFrame(portfolio_history).set_index('date')
        trades_df = pd.DataFrame(trade_history)
        
        # 메모리 최적화
        portfolio_df = self.memory_optimizer.optimize_dataframe(portfolio_df)
        if not trades_df.empty:
            trades_df = self.memory_optimizer.optimize_dataframe(trades_df)
        
        return portfolio_df, trades_df
    
    def calculate_performance_metrics_with_regime(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """체제를 고려한 성과 지표 계산"""
        # 기본 성과 지표
        metrics = self.calculate_performance_metrics(portfolio_df)
        
        # Jump Model 사용시 추가 지표
        if self.use_jump_model and 'regime' in portfolio_df.columns:
            # 체제별 성과 분석
            bull_df = portfolio_df[portfolio_df['regime'] == 'BULL']
            bear_df = portfolio_df[portfolio_df['regime'] == 'BEAR']
            
            if not bull_df.empty:
                bull_days = len(bull_df)
                if len(bull_df) > 1:
                    bull_return = (bull_df['value'].iloc[-1] / bull_df['value'].iloc[0] - 1) * 100
                else:
                    bull_return = 0
                
                metrics['BULL 기간'] = f"{bull_days}일 ({bull_days/len(portfolio_df)*100:.1f}%)"
                metrics['BULL 수익률'] = f"{bull_return:.2f}%"
                
                # BULL 기간 샤프 비율
                if len(bull_df) > 10:
                    bull_returns = bull_df['value'].pct_change().dropna()
                    bull_sharpe = self.calculator.calculate_sharpe_ratio_vectorized(
                        bull_returns.values, self.global_config.default_rf_rate
                    )
                    metrics['BULL 샤프 비율'] = f"{bull_sharpe:.3f}"
            
            if not bear_df.empty:
                bear_days = len(bear_df)
                if len(bear_df) > 1:
                    bear_return = (bear_df['value'].iloc[-1] / bear_df['value'].iloc[0] - 1) * 100
                else:
                    bear_return = 0
                
                metrics['BEAR 기간'] = f"{bear_days}일 ({bear_days/len(portfolio_df)*100:.1f}%)"
                metrics['BEAR 수익률'] = f"{bear_return:.2f}%"
            
            # Out-of-sample 분석
            if hasattr(self, 'training_cutoff_date'):
                oos_df = portfolio_df[portfolio_df.index > self.training_cutoff_date]
                if not oos_df.empty:
                    oos_days = len(oos_df)
                    metrics['Out-of-Sample 기간'] = f"{oos_days}일 ({oos_days/len(portfolio_df)*100:.1f}%)"
                    
                    if len(oos_df) > 1:
                        oos_return = (oos_df['value'].iloc[-1] / oos_df['value'].iloc[0] - 1) * 100
                        metrics['Out-of-Sample 수익률'] = f"{oos_return:.2f}%"
        
        return metrics
    
    def get_current_status(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """현재 전략 상태"""
        if date is None:
            date = datetime.now()
        
        status = {
            'date': date,
            'strategy': self.strategy_name
        }
        
        # 현재 체제
        if self.use_jump_model:
            regime_info = self.jump_model.get_current_regime(date)
            if regime_info:
                status['regime'] = regime_info['regime']
                status['regime_confidence'] = regime_info['confidence']
                status['is_out_of_sample'] = regime_info['is_out_of_sample']
            else:
                status['regime'] = 'UNKNOWN'
                status['regime_confidence'] = 0.0
        else:
            status['regime'] = 'BULL'
            status['regime_confidence'] = 1.0
        
        # 현재 투자 가능 구성요소
        if status['regime'] == 'BULL':
            # 간단한 분석을 위한 데이터 다운로드
            start_date = date - timedelta(days=100)
            try:
                benchmark_data, price_data = self.download_data(start_date, date, show_progress=False)
                selected = self.select_components(price_data, benchmark_data, date)
                
                status['selected_components'] = len(selected)
                status['top_components'] = selected[:5] if selected else []
            except:
                status['selected_components'] = 0
                status['top_components'] = []
        else:
            status['selected_components'] = 0
            status['top_components'] = []
            status['investment_status'] = 'SUSPENDED_BEAR_MARKET'
        
        return status

# 편의 함수
def create_integrated_strategy(preset_config: Dict[str, Any], 
                             use_jump_model: bool = True,
                             **kwargs) -> OptimizedIntegratedStrategy:
    """통합 전략 생성"""
    return OptimizedIntegratedStrategy(
        preset_config=preset_config,
        use_jump_model=use_jump_model,
        **kwargs
    )

def quick_status_check(preset_config: Dict[str, Any]) -> Dict[str, Any]:
    """빠른 상태 확인"""
    strategy = OptimizedIntegratedStrategy(preset_config)
    return strategy.get_current_status()
