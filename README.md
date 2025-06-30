# Universal RS Momentum Strategy + Jump Model

범용 RS (Relative Strength) 모멘텀 전략과 Jump Model을 결합한 투자 시스템입니다.
다양한 시장과 자산군에 적용 가능한 재사용 가능한 프레임워크입니다.

## 📋 주요 기능

- **다양한 시장 지원**: S&P 500, KOSPI, MSCI, 원자재, 암호화폐 등
- **RS 모멘텀 전략**: Pine Script RS 지표 기반 구성요소 선택
- **Jump Model**: 시장 체제(Bull/Bear) 자동 감지
- **백테스트**: 과거 성과 검증 및 최적화
- **실시간 모니터링**: 웹 기반 대시보드
- **성과 리포트**: HTML/PDF 형식의 전문적인 리포트

## 🚀 빠른 시작

### 1. 설치

```bash
# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. 기본 사용법

```python
from preset_manager import PresetManager
from universal_rs_with_jump import UniversalRSWithJumpModel
from datetime import datetime, timedelta

# 1. 프리셋 선택 (S&P 500 섹터)
sp500_preset = PresetManager.get_sp500_sectors()

# 2. 전략 생성
strategy = UniversalRSWithJumpModel(
    preset_config=sp500_preset,
    rs_length=20,
    rs_timeframe='daily',
    use_jump_model=True
)

# 3. 백테스트 실행
end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)  # 3년

portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)

# 4. 성과 확인
metrics = strategy.calculate_performance_metrics(portfolio_df)
for key, value in metrics.items():
    print(f"{key}: {value}")
```

### 3. 실시간 대시보드 실행

```bash
streamlit run realtime_dashboard.py
```

## 📁 파일 구조

```
universal-rs-strategy/
├── universal_rs_strategy.py      # 핵심 RS 전략 구현
├── universal_jump_model.py       # Jump Model 구현
├── universal_rs_with_jump.py     # RS + Jump Model 통합
├── preset_manager.py             # 시장 프리셋 관리
├── universal_main.py             # 메인 실행 스크립트
├── usage_examples.py             # 사용 예시
├── performance_reporter.py       # 성과 리포트 생성
├── realtime_dashboard.py         # Streamlit 대시보드
└── requirements.txt             # 필수 패키지
```

## 🎯 사용 가능한 프리셋

### 1. S&P 500 섹터
```python
preset = PresetManager.get_sp500_sectors()
# 벤치마크: ^GSPC
# 구성요소: XLK, XLF, XLV, XLY, XLI, XLE, XLP, XLB, XLRE, XLU, XLC
```

### 2. KOSPI 섹터
```python
preset = PresetManager.get_kospi_sectors()
# 벤치마크: 069500.KS (KODEX 200)
# 구성요소: TIGER 200 섹터 ETF들
```

### 3. MSCI 국가별 지수
```python
preset = PresetManager.get_msci_countries()
# 벤치마크: URTH (MSCI World)
# 구성요소: 주요 국가 ETF (EWZ, EWJ, EWG, EWU 등)
```

### 4. 사용자 정의 프리셋
```python
custom_preset = {
    'name': 'My Custom Strategy',
    'benchmark': 'SPY',
    'components': {
        'QQQ': 'Nasdaq 100',
        'IWM': 'Russell 2000',
        'DIA': 'Dow Jones'
    }
}

strategy = UniversalRSWithJumpModel(preset_config=custom_preset)
```

## 🔧 고급 설정

### RS 전략 파라미터
- `length`: RS 계산 기간 (기본: 20)
- `timeframe`: 'daily' 또는 'weekly'
- `recent_cross_days`: 최근 크로스 필터링 기간

### Jump Model 파라미터
- `jump_penalty`: 체제 전환 페널티 (기본: 50.0)
- `lookback_window`: 체제 판단 기간 (기본: 20)
- `n_states`: 상태 수 (기본: 2 - Bull/Bear)

## 📊 성과 리포트 생성

```python
from performance_reporter import PerformanceReporter

# 리포터 생성
reporter = PerformanceReporter(
    strategy_name="S&P 500 Sector RS Strategy",
    portfolio_df=portfolio_df,
    trades_df=trades_df
)

# HTML 리포트
reporter.generate_html_report()

# PDF 리포트
reporter.generate_pdf_report()

# 성과 지표 CSV
reporter.save_metrics_csv()
```

## 🔍 실시간 모니터링

```python
from usage_examples import realtime_monitoring

# 여러 전략 동시 모니터링
realtime_monitoring()
```

## 💡 사용 팁

1. **시장 선택**: 유동성이 높은 ETF를 사용하세요
2. **리밸런싱**: 월 1회 리밸런싱이 일반적입니다
3. **Jump Model**: 변동성이 높은 시장에서 특히 유용합니다
4. **백테스트**: 최소 3년 이상의 데이터로 검증하세요

## 📈 예시 결과

### S&P 500 섹터 전략 (3년)
- 총 수익률: 45.2%
- 연율화 수익률: 13.8%
- 샤프 비율: 0.92
- 최대 낙폭: -12.4%

### MSCI 국가별 전략 (3년)
- 총 수익률: 38.7%
- 연율화 수익률: 11.9%
- 샤프 비율: 0.85
- 최대 낙폭: -15.2%

## ⚠️ 주의사항

1. 이 코드는 교육 목적으로 제작되었습니다
2. 실제 투자 시 추가적인 리스크 관리가 필요합니다
3. 과거 성과가 미래 수익을 보장하지 않습니다
4. 거래 비용과 세금을 고려하세요

## 📄 라이선스

MIT License

---

## 필수 패키지 (requirements.txt)

```
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
streamlit>=1.28.0
scipy>=1.10.0
```
