from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 기존 금융 도구들
def calculate_returns(initial_investment: float, final_value: float) -> float:
    """투자 수익률(ROI)을 백분율로 계산합니다."""
    return ((final_value - initial_investment) / initial_investment) * 100

def calculate_compound_interest(principal: float, rate: float, time: float, n: float = 1) -> float:
    """복리를 계산합니다.
    
    Args:
        principal: 원금
        rate: 연 이자율 (소수점으로 표현, 예: 5%는 0.05)
        time: 기간 (년)
        n: 연간 복리 횟수
    """
    return principal * (1 + rate/n) ** (n * time)

def get_stock_info(symbol: str) -> str:
    """주어진 종목 코드의 현재 주식 정보를 가져옵니다."""
    # 시뮬레이션된 주식 데이터
    stock_data = {
        "AAPL": {"price": 188.25, "pe_ratio": 31.2, "market_cap": "2.95T", "dividend_yield": "0.44%"},
        "MSFT": {"price": 415.50, "pe_ratio": 35.8, "market_cap": "3.09T", "dividend_yield": "0.72%"},
        "JPM": {"price": 195.75, "pe_ratio": 11.2, "market_cap": "565B", "dividend_yield": "2.29%"},
        "GS": {"price": 475.30, "pe_ratio": 13.5, "market_cap": "148B", "dividend_yield": "2.10%"},
        "BRK.B": {"price": 420.15, "pe_ratio": 22.8, "market_cap": "878B", "dividend_yield": "0.00%"}
    }
    
    if symbol in stock_data:
        data = stock_data[symbol]
        return (
            f"{symbol} 주식 정보:\\n"
            f"- 현재 가격: ${data['price']}\\n"
            f"- P/E 비율: {data['pe_ratio']}\\n"
            f"- 시가총액: {data['market_cap']}\\n"
            f"- 배당 수익률: {data['dividend_yield']}"
        )
    else:
        return f"종목 코드 {symbol}에 대한 주식 데이터를 찾을 수 없습니다."

def get_economic_indicators() -> str:
    """현재 경제 지표를 가져옵니다."""
    return (
        "현재 경제 지표 (2024년 4분기):\\n"
        "1. **미국 인플레이션율 (CPI)**: 전년 대비 3.2%\\n"
        "2. **연방기금금리**: 5.25-5.50%\\n"
        "3. **미국 GDP 성장률**: 2.8% (연율)\\n"
        "4. **실업률**: 3.7%\\n"
        "5. **10년물 국채 수익률**: 4.25%\\n"
        "6. **S&P 500 연초 대비 수익률**: +24.5%"
    )

# 추가 금융 도구들
def calculate_portfolio_risk(volatility: float, beta: float) -> str:
    """포트폴리오 위험도를 계산합니다."""
    risk_score = (volatility * 0.6) + (beta * 0.4)
    if risk_score < 0.5:
        risk_level = "낮음"
    elif risk_score < 1.0:
        risk_level = "중간"
    else:
        risk_level = "높음"
    return f"위험도 점수: {risk_score:.2f} (위험 수준: {risk_level})"

def analyze_sector_performance(sector: str) -> str:
    """섹터별 성과를 분석합니다."""
    sector_data = {
        "기술": {"performance": "+28.5%", "outlook": "긍정적"},
        "금융": {"performance": "+15.2%", "outlook": "중립"},
        "헬스케어": {"performance": "+12.8%", "outlook": "긍정적"},
        "에너지": {"performance": "-5.3%", "outlook": "부정적"}
    }
    
    if sector in sector_data:
        data = sector_data[sector]
        return f"{sector} 섹터: 연초 대비 성과 {data['performance']}, 전망 {data['outlook']}"
    return f"{sector} 섹터 데이터를 찾을 수 없습니다."

def generate_investment_recommendation(risk_profile: str) -> str:
    """투자자 위험 성향에 따른 추천을 생성합니다."""
    recommendations = {
        "보수적": "채권 60%, 주식 30%, 현금 10% 배분 권장",
        "중도적": "주식 50%, 채권 40%, 대체투자 10% 배분 권장",
        "공격적": "주식 70%, 대체투자 20%, 채권 10% 배분 권장"
    }
    return recommendations.get(risk_profile, "위험 성향을 명확히 해주세요.")

def create_financial_report(analysis: str) -> str:
    """금융 분석 보고서를 작성합니다."""
    return f"[금융 분석 보고서]\\n{analysis}\\n\\n권장사항: 시장 변동성을 고려한 분산 투자 전략 수립 필요"


# 레벨 1: 기본 에이전트들
# 기존 에이전트들
portfolio_analyst = create_react_agent(
    model=model,
    tools=[calculate_returns, calculate_compound_interest],
    name="portfolio_analyst",
    prompt="당신은 포트폴리오 분석 전문가입니다. 투자 수익률, 복리를 계산하고 금융 계산을 수행합니다. 항상 한 번에 하나의 도구만 사용하세요."
)

market_researcher = create_react_agent(
    model=model,
    tools=[get_stock_info, get_economic_indicators],
    name="market_researcher",
    prompt="당신은 주식 데이터와 경제 지표에 접근할 수 있는 시장 조사 전문가입니다. 시장 인사이트와 주식 정보를 제공합니다. 계산은 수행하지 마세요."
)

# 신규 에이전트들
risk_analyst = create_react_agent(
    model=model,
    tools=[calculate_portfolio_risk],
    name="risk_analyst",
    prompt="당신은 위험 분석 전문가입니다. 포트폴리오의 위험도를 평가하고 분석합니다."
)

sector_analyst = create_react_agent(
    model=model,
    tools=[analyze_sector_performance],
    name="sector_analyst",
    prompt="당신은 섹터 분석 전문가입니다. 각 산업 섹터의 성과와 전망을 분석합니다."
)

investment_advisor = create_react_agent(
    model=model,
    tools=[generate_investment_recommendation],
    name="investment_advisor",
    prompt="당신은 투자 자문 전문가입니다. 고객의 위험 성향에 맞는 투자 전략을 제안합니다."
)

report_writer = create_react_agent(
    model=model,
    tools=[create_financial_report],
    name="report_writer",
    prompt="당신은 금융 보고서 작성 전문가입니다. 분석 결과를 종합하여 전문적인 보고서를 작성합니다."
)


# 레벨 2: 중간 관리자들 (팀 리더)
# 시장 분석팀
market_analysis_team = create_supervisor(
    [market_researcher, sector_analyst],
    model=model,
    supervisor_name="market_analysis_supervisor",
    prompt=(
        "당신은 시장 분석팀 감독자입니다. "
        "주식과 경제 지표는 market_researcher에게, "
        "섹터별 분석은 sector_analyst에게 위임하세요."
    )
).compile(name="market_analysis_team")

# 리스크 관리팀  
risk_management_team = create_supervisor(
    [portfolio_analyst, risk_analyst],
    model=model,
    supervisor_name="risk_management_supervisor",
    prompt=(
        "당신은 리스크 관리팀 감독자입니다. "
        "수익률과 복리 계산은 portfolio_analyst에게, "
        "위험도 평가는 risk_analyst에게 위임하세요."
    )
).compile(name="risk_management_team")

# 투자 자문팀
advisory_team = create_supervisor(
    [investment_advisor, report_writer],
    model=model,
    supervisor_name="advisory_supervisor",
    prompt=(
        "당신은 투자 자문팀 감독자입니다. "
        "투자 추천은 investment_advisor에게, "
        "보고서 작성은 report_writer에게 위임하세요."
    )
).compile(name="advisory_team")


# 레벨 3: 최고 관리자
chief_investment_officer = create_supervisor(
    [market_analysis_team, risk_management_team, advisory_team],
    model=model,
    supervisor_name="chief_investment_officer",
    prompt=(
        "당신은 최고 투자 책임자(CIO)입니다. "
        "시장 분석팀, 리스크 관리팀, 투자 자문팀을 총괄합니다. "
        "시장 데이터와 섹터 분석은 market_analysis_team에게, "
        "수익률과 위험 평가는 risk_management_team에게, "
        "투자 전략과 보고서는 advisory_team에게 위임하세요. "
        "모든 팀의 결과를 종합하여 통합적인 투자 의사결정을 지원합니다."
    )
).compile(name="chief_investment_officer")


# 실행 예제
if __name__ == "__main__":
    # 예제 1: 단순 쿼리 (한 팀만 필요)
    print("=== 예제 1: 단순 주식 정보 조회 ===")
    result1 = chief_investment_officer.invoke({
        "messages": [HumanMessage(content="애플 주식의 현재 정보를 알려주세요.")]
    })
    print(result1["messages"][-1].content)
    print()

    # 예제 2: 복합 쿼리 (여러 팀 협업)
    print("=== 예제 2: 포트폴리오 분석 ===")
    result2 = chief_investment_officer.invoke({
        "messages": [HumanMessage(
            content="$50,000를 투자했는데 현재 $65,000가 되었습니다. "
                   "수익률을 계산하고, 변동성 0.8, 베타 1.2일 때 위험도를 평가해주세요."
        )]
    })
    print(result2["messages"][-1].content)
    print()

    # 예제 3: 종합 투자 전략 (모든 팀 참여)
    print("=== 예제 3: 종합 투자 자문 ===")
    result3 = chief_investment_officer.invoke({
        "messages": [HumanMessage(
            content="현재 경제 상황과 기술 섹터 성과를 분석하고, "
                   "중도적 위험 성향 투자자를 위한 투자 전략 보고서를 작성해주세요."
        )]
    })
    print(result3["messages"][-1].content)