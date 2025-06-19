from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

checkpointer = InMemorySaver()
store = InMemoryStore()


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

# 전문 금융 에이전트 생성
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

# 금융팀을 위한 감독자 워크플로우 생성
finance_supervisor = create_supervisor(
    [market_researcher, portfolio_analyst],
    model=model,
    prompt=(
        "당신은 전문가 팀을 관리하는 수석 금융 자문가입니다: "
        "시장 조사원과 포트폴리오 분석가가 있습니다. "
        "주식 정보와 경제 데이터는 market_researcher를 사용하세요. "
        "투자 계산과 수익률 분석은 portfolio_analyst를 사용하세요. "
        "항상 적절한 전문가에게 위임하고 한 번에 하나의 도구만 사용하세요."
    )
)

# 워크플로우 컴파일
app = finance_supervisor.compile(checkpointer=checkpointer, store=store)

# 예제 1: 주식 정보 가져오기
print("=== 예제 1: 주식 정보 ===")
result1 = app.invoke({"messages": [HumanMessage(content="애플 주식의 현재 가격과 P/E 비율은 얼마인가요?")]},config={"configurable": {"thread_id": 100}})
print(result1["messages"][-1].content)
print()

print("=== 예제 1: 주식 정보 memory 기능 테스트===")
result1 = app.invoke({"messages": [HumanMessage(content="방금 어떤 주식 실문을 했지?")]},config={"configurable": {"thread_id": 100}})
print(result1["messages"][-1].content)
print()


# # 예제 2: 투자 수익률 계산
# print("=== 예제 2: 투자 수익률 ===")
# result2 = app.invoke({"messages": [HumanMessage(content="주식에 $10,000를 투자했는데 현재 가치가 $12,500이면 수익률은 몇 퍼센트인가요?")]})
# print(result2["messages"][-1].content)
# print()

# # 예제 3: 두 에이전트가 필요한 복잡한 쿼리
# print("=== 예제 3: 복잡한 금융 분석 ===")
# result3 = app.invoke({"messages": [HumanMessage(content="마이크로소프트 주식의 현재 가격은 얼마이고, $5,000가 연 8% 복리로 10년 후에는 얼마가 되나요?")]})
# print(result3["messages"][-1].content)