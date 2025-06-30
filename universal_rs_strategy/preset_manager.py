import json
import os

class PresetManager:
    """
    다양한 시장 프리셋 관리 (KOSPI 전체 시장 커버리지 추가)
    """
    
    @staticmethod
    def get_sp500_sectors():
        """S&P 500 섹터 ETF"""
        return {
            'benchmark': '^GSPC',  # S&P 500 Index
            'name': 'S&P 500 Sector RS Strategy',
            'components': {
                'XLK': 'Technology Select Sector',
                'XLF': 'Financial Select Sector',
                'XLV': 'Health Care Select Sector',
                'XLY': 'Consumer Discretionary Select',
                'XLI': 'Industrial Select Sector',
                'XLE': 'Energy Select Sector',
                'XLP': 'Consumer Staples Select Sector',
                'XLB': 'Materials Select Sector',
                'XLRE': 'Real Estate Select Sector',
                'XLU': 'Utilities Select Sector',
                'XLC': 'Communication Services Select'
            }
        }
    
    @staticmethod
    def get_kospi_sectors():
        """코스피 200 섹터 ETF (대형주 중심)"""
        return {
            'benchmark': '069500.KS',  # KODEX 200
            'name': 'KOSPI 200 Sector RS Strategy (Large Cap)',
            'components': {
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
        }
    
    @staticmethod
    def get_kospi_full_sectors():
        """코스피 전체 시장 섹터 ETF (전체 시장 커버)"""
        return {
            'benchmark': '^KS11',  # KOSPI Composite Index
            'name': 'KOSPI Full Market Sector RS Strategy',
            'components': {
                '091230.KS': 'TIGER IT',
                '091170.KS': 'TIGER 금융',
                '102970.KS': 'TIGER 건설',
                '102780.KS': 'TIGER 철강소재',
                '102960.KS': 'TIGER 에너지화학',
                '102980.KS': 'TIGER 조선',
                '139230.KS': 'TIGER 200 산업재',  # 대체용
                '139240.KS': 'TIGER 200 소비재',  # 대체용
                '227540.KS': 'TIGER 200 건강관리',  # 대체용
                '148020.KS': 'TIGER 코스피고배당',
                '114800.KS': 'KODEX 인버스',
                '252670.KS': 'KODEX 200선물인버스2X',
                '233740.KS': 'KODEX 코스닥150선물인버스',
                '251340.KS': 'KODEX 코스닥150',
                '152100.KS': 'ARIRANG 코스피',
                '069660.KS': 'KOSEF 200',
                '278530.KS': 'KODEX 200TR',
                '130680.KS': 'TIGER 바이오헬스케어'
            }
        }
    
    @staticmethod
    def get_kosdaq_sectors():
        """코스닥 섹터 ETF (중소형주/성장주 중심)"""
        return {
            'benchmark': '229200.KS',  # KODEX 코스닥150
            'name': 'KOSDAQ Sector RS Strategy',
            'components': {
                '251340.KS': 'KODEX 코스닥150',
                '233740.KS': 'KODEX 코스닥150선물인버스',
                '130680.KS': 'TIGER 바이오헬스케어',
                '091160.KS': 'KODEX 하이일드',
                '182490.KS': 'TIGER 코스닥150',
                '261220.KS': 'KODEX WTI원유선물',
                '238720.KS': 'KODEX 코스닥150 레버리지',
                '130730.KS': 'KOSEF 단기자금',
                '148070.KS': 'KOSEF 국고채10년',
                '285130.KS': 'KODEX 코스닥150 선물',
                '252710.KS': 'TIGER 코스닥150 레버리지',
                '143850.KS': 'TIGER 코스닥150 ETN'
            }
        }
    
    @staticmethod
    def get_korea_comprehensive():
        """한국 종합 시장 전략 (대형주 + 중소형주 혼합)"""
        return {
            'benchmark': '1001.KS',  # KOSPI
            'name': 'Korea Comprehensive Market RS Strategy',
            'components': {
                # 대형주 대표
                '069500.KS': 'KODEX 200',
                '278530.KS': 'KODEX 200TR',
                '152100.KS': 'ARIRANG 코스피',
                
                # 중소형주 대표
                '251340.KS': 'KODEX 코스닥150',
                '182490.KS': 'TIGER 코스닥150',
                
                # 섹터별
                '139220.KS': 'TIGER 200 IT',
                '091230.KS': 'TIGER IT',
                '139250.KS': 'TIGER 200 금융',
                '091170.KS': 'TIGER 금융',
                '130680.KS': 'TIGER 바이오헬스케어',
                '102970.KS': 'TIGER 건설',
                '102780.KS': 'TIGER 철강소재',
                '102960.KS': 'TIGER 에너지화학',
                
                # 스타일
                '148020.KS': 'TIGER 코스피고배당',
                '114800.KS': 'KODEX 인버스'
            }
        }
    
    @staticmethod
    def get_msci_countries():
        """MSCI 국가별 ETF"""
        return {
            'benchmark': 'URTH',  # MSCI World ETF
            'name': 'MSCI Country RS Strategy',
            'components': {
                'EWZ': 'Brazil (MSCI Brazil)',
                'EWJ': 'Japan (MSCI Japan)',
                'EWG': 'Germany (MSCI Germany)',
                'EWU': 'United Kingdom (MSCI UK)',
                'EWA': 'Australia (MSCI Australia)',
                'EWC': 'Canada (MSCI Canada)',
                'EWY': 'South Korea (MSCI South Korea)',
                'EWH': 'Hong Kong (MSCI Hong Kong)',
                'EWS': 'Singapore (MSCI Singapore)',
                'EWT': 'Taiwan (MSCI Taiwan)',
                'INDA': 'India (MSCI India)',
                'FXI': 'China (FTSE China)',
                'EWW': 'Mexico (MSCI Mexico)',
                'EWI': 'Italy (MSCI Italy)',
                'EWP': 'Spain (MSCI Spain)',
                'EWQ': 'France (MSCI France)',
                'EWN': 'Netherlands (MSCI Netherlands)',
                'EWL': 'Switzerland (MSCI Switzerland)'
            }
        }
    
    @staticmethod
    def get_europe_sectors():
        """유럽 섹터 ETF"""
        return {
            'benchmark': 'FEZ',  # EURO STOXX 50 ETF
            'name': 'Europe Sector RS Strategy',
            'components': {
                'IEUS': 'Europe Small-Cap',
                'IEUR': 'Europe Real Estate',
                'EUFN': 'Europe Financials',
                'FEZ': 'Europe Large-Cap',
                'EZU': 'Eurozone',
                'HEDJ': 'Europe Hedged Equity',
                'IEV': 'Europe 350 Index',
                'VGK': 'Europe ETF'
            }
        }
    
    @staticmethod
    def get_global_sectors():
        """글로벌 섹터 ETF"""
        return {
            'benchmark': 'VT',  # Vanguard Total World Stock ETF
            'name': 'Global Sector RS Strategy',
            'components': {
                'IXC': 'Global Energy',
                'IXG': 'Global Financials',
                'IXJ': 'Global Healthcare',
                'IXN': 'Global Technology',
                'IXP': 'Global Telecom',
                'MXI': 'Global Materials',
                'KXI': 'Global Consumer Staples',
                'RXI': 'Global Consumer Discretionary',
                'JXI': 'Global Utilities',
                'REET': 'Global Real Estate'
            }
        }
    
    @staticmethod
    def get_emerging_markets():
        """신흥시장 ETF"""
        return {
            'benchmark': 'EEM',  # iShares MSCI Emerging Markets ETF
            'name': 'Emerging Markets RS Strategy',
            'components': {
                'EWZ': 'Brazil',
                'FXI': 'China Large-Cap',
                'INDA': 'India',
                'EWY': 'South Korea',
                'EWT': 'Taiwan',
                'EWW': 'Mexico',
                'RSX': 'Russia',
                'EZA': 'South Africa',
                'THD': 'Thailand',
                'EPHE': 'Philippines',
                'EWM': 'Malaysia',
                'EIDO': 'Indonesia',
                'TUR': 'Turkey',
                'ARGT': 'Argentina',
                'EPU': 'Peru',
                'ECH': 'Chile'
            }
        }
    
    @staticmethod
    def get_commodity_sectors():
        """원자재 섹터 ETF"""
        return {
            'benchmark': 'DJP',  # iPath Bloomberg Commodity Index
            'name': 'Commodity Sector RS Strategy',
            'components': {
                'GLD': 'Gold',
                'SLV': 'Silver',
                'USO': 'Oil',
                'UNG': 'Natural Gas',
                'DBA': 'Agriculture',
                'DBB': 'Base Metals',
                'DBC': 'Commodities Broad',
                'PDBC': 'Optimum Yield Commodities',
                'CORN': 'Corn',
                'WEAT': 'Wheat',
                'SOYB': 'Soybeans',
                'CPER': 'Copper',
                'PALL': 'Palladium',
                'PPLT': 'Platinum'
            }
        }
    
    @staticmethod
    def get_crypto_assets():
        """암호화폐 관련 자산 (주의: 높은 변동성)"""
        return {
            'benchmark': 'BITO',  # ProShares Bitcoin Strategy ETF
            'name': 'Crypto Assets RS Strategy',
            'components': {
                'GBTC': 'Grayscale Bitcoin Trust',
                'ETHE': 'Grayscale Ethereum Trust',
                'BITO': 'ProShares Bitcoin Strategy',
                'BITQ': 'Bitwise Crypto Industry Innovators',
                'BLOK': 'Amplify Transformational Data Sharing',
                'BLCN': 'Reality Shares Nasdaq NextGen Economy'
            }
        }
    
    @staticmethod
    def get_factor_etfs():
        """팩터 기반 ETF"""
        return {
            'benchmark': 'SPY',  # S&P 500 ETF
            'name': 'Factor-Based RS Strategy',
            'components': {
                'MTUM': 'Momentum Factor',
                'VLUE': 'Value Factor',
                'QUAL': 'Quality Factor',
                'SIZE': 'Size Factor',
                'USMV': 'Minimum Volatility',
                'VFMV': 'Minimum Volatility International',
                'DGRO': 'Dividend Growth',
                'NOBL': 'Dividend Aristocrats',
                'PKW': 'Buyback ETF',
                'SPLV': 'Low Volatility S&P 500'
            }
        }
    
    @staticmethod
    def get_thematic_etfs():
        """테마 ETF"""
        return {
            'benchmark': 'QQQ',  # Nasdaq 100
            'name': 'Thematic RS Strategy',
            'components': {
                'ARKK': 'ARK Innovation',
                'ICLN': 'Clean Energy',
                'ESPO': 'Video Games & Esports',
                'ROBO': 'Robotics & AI',
                'HACK': 'Cyber Security',
                'FINX': 'FinTech',
                'GNOM': 'Genomics',
                'DRIV': 'Autonomous Vehicles',
                'KOMP': 'AI & Big Data',
                'CLOU': 'Cloud Computing',
                'HERO': 'Video Game Tech',
                'NERD': 'Gaming & Esports',
                'BETZ': 'Sports Betting',
                'POTX': 'Cannabis'
            }
        }
    
    @staticmethod
    def save_custom_preset(name, benchmark, components, filename=None):
        """사용자 정의 프리셋 저장"""
        preset = {
            'name': name,
            'benchmark': benchmark,
            'components': components
        }
        
        if filename is None:
            filename = f"preset_{name.lower().replace(' ', '_')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(preset, f, ensure_ascii=False, indent=2)
        
        print(f"프리셋이 {filename}에 저장되었습니다.")
        return filename
    
    @staticmethod
    def load_custom_preset(filename):
        """사용자 정의 프리셋 로드"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                preset = json.load(f)
            return preset
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {filename}")
            return None
        except json.JSONDecodeError:
            print(f"JSON 파일 형식이 올바르지 않습니다: {filename}")
            return None
    
    @staticmethod
    def list_presets():
        """사용 가능한 모든 프리셋 목록"""
        presets = [
            "S&P 500 Sectors",
            "KOSPI 200 Sectors (Large Cap)",
            "KOSPI Full Market Sectors",
            "KOSDAQ Sectors",
            "Korea Comprehensive Market",
            "MSCI Countries",
            "Europe Sectors",
            "Global Sectors",
            "Emerging Markets",
            "Commodity Sectors",
            "Crypto Assets",
            "Factor ETFs",
            "Thematic ETFs"
        ]
        
        print("\n=== 사용 가능한 프리셋 ===")
        for i, preset in enumerate(presets, 1):
            print(f"{i}. {preset}")
        
        # 커스텀 프리셋 파일 찾기
        custom_files = [f for f in os.listdir('.') if f.startswith('preset_') and f.endswith('.json')]
        if custom_files:
            print("\n=== 사용자 정의 프리셋 ===")
            for file in custom_files:
                print(f"- {file}")
        
        return presets


# 예시: 사용자 정의 프리셋 생성
def create_custom_preset_example():
    """사용자 정의 프리셋 생성 예시"""
    # 예시 1: 아시아 주요 시장
    asia_preset = {
        'name': 'Asia Major Markets RS Strategy',
        'benchmark': 'AAXJ',  # iShares MSCI All Country Asia ex Japan ETF
        'components': {
            'EWJ': 'Japan',
            'FXI': 'China',
            'INDA': 'India',
            'EWY': 'South Korea',
            'EWT': 'Taiwan',
            'EWS': 'Singapore',
            'EWH': 'Hong Kong',
            'THD': 'Thailand',
            'EWM': 'Malaysia',
            'EIDO': 'Indonesia'
        }
    }
    
    PresetManager.save_custom_preset(
        asia_preset['name'],
        asia_preset['benchmark'],
        asia_preset['components']
    )
    
    # 예시 2: 미국 성장주 ETF
    growth_preset = {
        'name': 'US Growth ETFs RS Strategy',
        'benchmark': 'VUG',  # Vanguard Growth ETF
        'components': {
            'QQQ': 'Nasdaq 100',
            'VUG': 'Vanguard Growth',
            'IWF': 'iShares Russell 1000 Growth',
            'SCHG': 'Schwab US Large-Cap Growth',
            'MGK': 'Vanguard Mega Cap Growth',
            'VONG': 'Vanguard Russell 1000 Growth',
            'IWY': 'iShares Russell Top 200 Growth',
            'RPG': 'Invesco S&P 500 Pure Growth'
        }
    }
    
    PresetManager.save_custom_preset(
        growth_preset['name'],
        growth_preset['benchmark'],
        growth_preset['components']
    )

if __name__ == "__main__":
    # 프리셋 목록 표시
    PresetManager.list_presets()
    
    # 새로운 한국 시장 프리셋 테스트
    print("\n=== 새로운 한국 시장 프리셋 ===")
    
    kospi_full = PresetManager.get_kospi_full_sectors()
    print(f"\n{kospi_full['name']}")
    print(f"벤치마크: {kospi_full['benchmark']}")
    print(f"구성요소 수: {len(kospi_full['components'])}")
    
    kosdaq = PresetManager.get_kosdaq_sectors()
    print(f"\n{kosdaq['name']}")
    print(f"벤치마크: {kosdaq['benchmark']}")
    print(f"구성요소 수: {len(kosdaq['components'])}")
    
    korea_comp = PresetManager.get_korea_comprehensive()
    print(f"\n{korea_comp['name']}")
    print(f"벤치마크: {korea_comp['benchmark']}")
    print(f"구성요소 수: {len(korea_comp['components'])}")
    
    # 사용자 정의 프리셋 생성 예시
    create_custom_preset_example()
