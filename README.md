# Universal RS Momentum Strategy + Jump Model

범용 RS(Relative Strength) 모멘텀 전략과 Jump Model(체제 감지 모델)을 결합한 투자 리서치용 코드베이스입니다.
S&P 500, KOSPI, MSCI 등 다양한 시장/자산군 프리셋에 대해 섹터·국가 로테이션 백테스트와 실시간 대시보드를 제공합니다.

> ⚠️ **이 저장소는 현재 여러 핵심 기능이 정상 동작하지 않는 상태입니다.** 아래 [알려진 이슈](#-알려진-이슈-known-issues) 섹션을 반드시 먼저 읽어 주세요. 특히 RS 컴포넌트 선택과 Jump Model 백테스트 경로는 그대로 실행하면 예외가 발생하거나 잘못된 결과를 냅니다.

## 📁 실제 프로젝트 구조

```
trading_strategy/
├── __init__.py                    # ⚠️ 사실상 Streamlit 대시보드 코드가 그대로 들어있는 파일 (아래 이슈 참고)
├── requirements.txt                # 실제 의존성 목록 (아래 표와 동일)
└── universal_rs_strategy/          # 실제 구현 코드는 전부 이 폴더 안에 있습니다
    ├── __init__.py                 # 비어 있음 (패키지로서 export 되는 심볼 없음)
    ├── universal_rs_strategy.py    # RS 모멘텀 전략 코어
    ├── universal_jump_model.py     # Jump Model (체제 감지) 코어
    ├── universal_rs_with_jump.py   # RS 전략 + Jump Model 통합 백테스트
    ├── preset_manager.py           # 시장/자산군 프리셋 정의
    ├── risk_free_rate_utils.py     # 동적 무위험금리(^IRX) 유틸리티
    ├── performance_reporter.py     # HTML/PDF 성과 리포트 생성
    ├── realtime_dashboard.py       # Streamlit 실시간 대시보드
    ├── universal_main.py           # 콘솔 메뉴 기반 실행 스크립트
    └── usage_examples.py           # 사용 예시 모음
```

모든 모듈이 `from preset_manager import PresetManager`처럼 **상대 임포트가 아닌 bare import**로 서로를 참조합니다. 즉 이 코드는 패키지(`import trading_strategy...`)로 불러오는 것이 아니라, `universal_rs_strategy/` 디렉터리 안에서 스크립트를 직접 실행하거나 그 디렉터리를 `PYTHONPATH`/`sys.path`에 추가했을 때만 정상적으로 import됩니다.

## 🚀 설치

```bash
pip install -r requirements.txt
```

주요 의존성: `pandas`, `numpy`, `yfinance`, `scikit-learn`, `scipy`, `matplotlib`, `plotly`, `streamlit`, `numba`, `reportlab` 등. 전체 목록은 `requirements.txt`를 참고하세요.

## 🎬 실행 방법

모든 예시는 **`universal_rs_strategy/` 디렉터리 안에서** 실행해야 합니다 (모듈들이 서로를 bare import로 참조하기 때문).

### 콘솔 메뉴로 실행

```bash
cd universal_rs_strategy
python universal_main.py
```

### 실시간 대시보드 실행

```bash
cd universal_rs_strategy
streamlit run realtime_dashboard.py
```

> 저장소 루트의 `__init__.py`도 겉보기엔 대시보드 코드처럼 보이지만, `realtime_dashboard.py`의 예전 버전이 별도로 갈라져 나온 **중복 파일**이며 루트에서는 `preset_manager` 등을 import할 수 없어 실행되지 않습니다. 대시보드는 반드시 `universal_rs_strategy/realtime_dashboard.py`로 실행하세요.

### 코드에서 직접 사용

```python
import sys
sys.path.insert(0, "universal_rs_strategy")  # 또는 이 디렉터리에서 스크립트 실행

from preset_manager import PresetManager
from universal_rs_with_jump import UniversalRSWithJumpModel
from datetime import datetime, timedelta

preset = PresetManager.get_sp500_sectors()

strategy = UniversalRSWithJumpModel(
    preset_config=preset,
    rs_length=20,
    rs_timeframe='daily',
    use_jump_model=True,          # ⚠️ 현재 True/False 둘 다 결함이 있음 (아래 이슈 참고)
)

end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 3)
portfolio_df, trades_df, regime_df = strategy.backtest(start_date, end_date)
```

## 🎯 사용 가능한 프리셋 (`PresetManager`)

| 메서드 | 설명|
|---|---|
| `get_sp500_sectors()` | S&P 500 섹터 ETF (벤치마크 `^GSPC`) |
| `get_kospi_sectors()` / `get_kospi_full_sectors()` | KOSPI 섹터 ETF |
| `get_kosdaq_sectors()` | KOSDAQ 섹터 ETF |
| `get_korea_comprehensive()` | 한국 종합 프리셋 (벤치마크 티커 오류 있음, 이슈 참고) |
| `get_msci_countries()` | MSCI 국가별 지수 (벤치마크 `URTH`) |
| `get_europe_sectors()` | 유럽 섹터 |
| `get_global_sectors()` | 글로벌 섹터 |
| `get_emerging_markets()` | 신흥국 시장 |
| `get_commodity_sectors()` | 원자재 섹터 |
| `get_crypto_assets()` | 암호화폐 자산 |
| `get_factor_etfs()` | 팩터 ETF |
| `get_thematic_etfs()` | 테마 ETF |

사용자 정의 프리셋도 `{'name', 'benchmark', 'components'}` 딕셔너리로 직접 만들어 넘길 수 있습니다.

## 🔧 주요 파라미터

**RS 전략** (`UniversalRSStrategy` / `UniversalRSWithJumpModel`)
- `rs_length`: RS 계산 기간 (기본 20)
- `rs_timeframe`: `'daily'` 또는 `'weekly'`
- `rs_recent_cross_days`: 문서상 "최근 크로스 필터링" 파라미터이지만 **현재 코드에서 읽히지 않아 아무 효과가 없습니다.**

**Jump Model** (`UniversalJumpModel`)
- `jump_penalty`: 체제 전환 페널티 (기본 50.0) — 학습 시와 추론 시 스케일이 100배 다르게 적용되는 버그가 있습니다 (이슈 참고).
- `n_states`: 상태 수 (기본 2, Bull/Bear)
- `training_cutoff_date`: 학습 마감일 (기본 2024-12-31)
- `use_paper_features_only`: True면 논문 기준 3개 특징만 사용

## 📊 성과 리포트 생성

```python
from performance_reporter import PerformanceReporter

reporter = PerformanceReporter(
    strategy_name="S&P 500 Sector RS Strategy",
    portfolio_df=portfolio_df,
    trades_df=trades_df,
)
reporter.generate_html_report()
reporter.generate_pdf_report()
reporter.save_metrics_csv()
```

`strategy_name`이 이스케이프 없이 HTML에 삽입되므로, 신뢰할 수 없는 문자열을 그대로 넘기지 마세요 (이슈 참고).

## 🐛 알려진 이슈 (Known Issues)

코드 리뷰에서 발견된, 실제 사용에 영향을 주는 문제들입니다. 우선순위(치명적 → 낮음) 순으로 정리했습니다.

### 치명적 (핵심 기능이 동작하지 않음)

1. **RS 컴포넌트 선택이 항상 실패합니다.** `universal_rs_strategy.py`의 `get_price_data()`가 이미 `columns.droplevel(1)`로 컬럼을 평탄화한 데이터를, `safe_calculate_rs_components()`가 다시 `droplevel(1)`을 호출하면서 예외가 발생 → 모든 프리셋에서 컴포넌트가 하나도 선택되지 않습니다.
2. **`use_jump_model=True`(기본값)로 백테스트하면 `AttributeError`가 발생합니다.** `universal_rs_with_jump.py`가 호출하는 `UniversalJumpModel.get_regime_history()`가 어디에도 정의되어 있지 않습니다.
3. **`use_jump_model=False`일 때의 백테스트는 실제 전략과 무관한 난수입니다.** `UniversalRSStrategy.backtest()`가 `np.random.normal()`로 일별 수익률을 생성하여 반환합니다.
4. **저장소 루트 `__init__.py`는 임포트 시 깨집니다.** Streamlit 앱 코드가 모듈 최상단 부작용(`st.set_page_config()` 등)으로 들어 있고, 존재하지 않는 경로의 모듈을 import하려 시도합니다. `realtime_dashboard.py`의 갈라져 나온 구버전 사본이며, 표시되는 학습 마감일(`2025-06-30`)도 실제 모델 생성 코드의 값(`2024-12-31`)과 다릅니다.

### 정확성/보안

- **Jump penalty 스케일 불일치**: 학습 시 원값을 그대로 쓰고 추론 시 `/100`을 적용해, 같은 파라미터가 학습·추론에서 다르게 작동합니다 (`universal_jump_model.py`).
- **동일 봉(same-bar) 시그널/체결**로 인한 look-ahead bias: 리밸런싱 판단과 체결가가 동일 날짜의 종가를 사용합니다.
- **에러 발생 시 기본값이 항상 'BULL'(투자 유지)**로 처리되어, 장애 상황에서 가장 위험한 방향으로 폴백합니다.
- **HTML 리포트에 XSS 가능성**: `strategy_name` 등이 이스케이프 없이 HTML에 삽입됩니다.
- `performance_reporter.py`의 `turnover_rate` 계산에서 0으로 나눌 수 있는 경우가 있고, 최대 낙폭 지속기간 계산 로직이 상승장에서도 잘못된 값을 반환합니다.
- `preset_manager.py`의 `get_korea_comprehensive()` 벤치마크 티커(`1001.KS`)가 잘못되었습니다 (정상: `^KS11`).
- `recent_cross_days` 파라미터가 문서화만 되어 있고 실제로는 작동하지 않습니다.

### 성능

- Streamlit 대시보드에 캐싱(`@st.cache_data`)이 전혀 없어 위젯 조작마다 전체 데이터를 재다운로드합니다.
- 병렬 처리용 import(`concurrent.futures`, `Lock`)가 있지만 실제로는 모든 티커를 순차 처리합니다.
- 무위험금리·가격 데이터가 계산 경로마다 중복 다운로드됩니다.
- WMA/RS 비율/모멘텀 계산과 Jump penalty 최적화가 벡터화되지 않은 파이썬 루프로 구현되어 있습니다.

### 기타

- `universal_rs_strategy/__init__.py`가 비어 있어 패키지로 온전히 동작하지 않으며, 모든 모듈이 상대 임포트 없이 서로를 참조합니다.
- 사용자 입력이 그대로 파일명에 사용되는 부분이 있어 경로 조작에 취약할 수 있습니다.
- `fillna(method=...)`, `resample('M')` 등 향후 pandas 버전에서 제거될 API를 사용 중입니다.

이 항목들은 아직 수정되지 않았습니다. 수정 작업이 필요하면 이슈로 등록하거나 별도로 요청해 주세요.

## 💡 사용 팁

1. 유동성이 높은 ETF를 구성요소로 사용하세요.
2. 월 1회 리밸런싱이 일반적입니다.
3. 위 "알려진 이슈"의 백테스트 관련 결함이 수정되기 전까지는, 산출되는 성과 지표를 실제 투자 판단에 사용하지 마세요.

## ⚠️ 주의사항

1. 이 코드는 교육/리서치 목적으로 제작되었습니다.
2. 실제 투자 시 추가적인 리스크 관리가 필요합니다.
3. 과거 성과가 미래 수익을 보장하지 않습니다.
4. 거래 비용과 세금을 고려하세요.

## 📄 라이선스

MIT License
