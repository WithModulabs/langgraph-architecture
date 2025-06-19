from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm, create_handoff_tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing import Literal
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# AI 모델 설정
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ====================================
# 1. 간단한 금융 도구들 정의
# ====================================

@tool
def calculate_loan_payment(principal: float, annual_rate: float, months: int) -> dict:
    """대출 월 상환금액을 계산합니다."""
    # 월 이자율 계산
    monthly_rate = annual_rate / 100 / 12
    
    # 월 상환액 계산 (원리금균등상환)
    if monthly_rate == 0:
        monthly_payment = principal / months
        total_interest = 0
    else:
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
        total_interest = monthly_payment * months - principal
    
    return {
        "월_상환금액": f"{monthly_payment:,.0f}원",
        "총_이자": f"{total_interest:,.0f}원",
        "총_상환금액": f"{monthly_payment * months:,.0f}원"
    }

@tool
def calculate_investment_return(principal: float, annual_return: float, years: int) -> dict:
    """복리 투자 수익을 계산합니다."""
    # 복리 계산
    final_amount = principal * (1 + annual_return/100)**years
    profit = final_amount - principal
    
    return {
        "투자원금": f"{principal:,.0f}원",
        "예상수익": f"{profit:,.0f}원",
        "최종금액": f"{final_amount:,.0f}원",
        "수익률": f"{(profit/principal)*100:.1f}%"
    }

@tool
def check_balance() -> dict:
    """계좌 잔액을 조회합니다 (시뮬레이션)."""
    # 실제로는 데이터베이스에서 조회하지만, 여기서는 예시 데이터 사용
    return {
        "예금계좌": "5,000,000원",
        "적금계좌": "12,000,000원",
        "투자계좌": "8,500,000원",
        "총자산": "25,500,000원",
        "조회시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ====================================
# 2. 전문가 에이전트 생성
# ====================================

# 대출 전문가 에이전트
loan_expert = create_react_agent(
    model,
    tools=[
        calculate_loan_payment,
        create_handoff_tool(
            agent_name="InvestmentExpert",
            description="Transfer to investment expert for investment consultation"
        ),
        create_handoff_tool(
            agent_name="WealthManager",
            description="Transfer to wealth manager for account management"
        )
    ],
    prompt="""당신은 친절한 대출 전문가입니다.
    대출 상담과 월 상환액 계산을 도와드립니다.
    투자나 계좌 관련 문의는 다른 전문가에게 연결해드립니다.""",
    name="LoanExpert"  # 영문 이름 사용
)

# 투자 전문가 에이전트
investment_expert = create_react_agent(
    model,
    tools=[
        calculate_investment_return,
        create_handoff_tool(
            agent_name="LoanExpert",
            description="Transfer to loan expert for loan consultation"
        ),
        create_handoff_tool(
            agent_name="WealthManager",
            description="Transfer to wealth manager for account management"
        )
    ],
    prompt="""당신은 경험 많은 투자 전문가입니다.
    투자 수익률 계산과 투자 상담을 제공합니다.
    대출이나 계좌 관련 문의는 다른 전문가에게 연결해드립니다.""",
    name="InvestmentExpert"  # 영문 이름 사용
)

# 종합 자산관리사 에이전트
wealth_manager = create_react_agent(
    model,
    tools=[
        check_balance,
        create_handoff_tool(
            agent_name="LoanExpert",
            description="Transfer to loan expert for loan consultation"
        ),
        create_handoff_tool(
            agent_name="InvestmentExpert",
            description="Transfer to investment expert for investment consultation"
        )
    ],
    prompt="""당신은 종합 자산관리사입니다.
    계좌 조회와 전반적인 재무 상담을 제공합니다.
    전문적인 대출이나 투자 상담은 해당 전문가에게 연결해드립니다.""",
    name="WealthManager"  # 영문 이름 사용
)

# ====================================
# 3. 스웜 시스템 구성
# ====================================

# 메모리 설정
checkpointer = InMemorySaver()  # 대화 기록 저장
store = InMemoryStore()  # 장기 데이터 저장

# 멀티 에이전트 스웜 생성
financial_swarm = create_swarm(
    [loan_expert, investment_expert, wealth_manager],
    default_active_agent="WealthManager"  # 처음에는 자산관리사가 응대
)

# 스웜 시스템 컴파일
app = financial_swarm.compile(
    checkpointer=checkpointer,
    store=store
)

# ====================================
# 4. 스웜 동작 테스트
# ====================================

if __name__ == "__main__":
    # 대화 세션 ID 설정 (같은 ID로 대화 이어가기)
    config = {"configurable": {"thread_id": "test_session_001"}}
    
    print("=" * 50)
    print("🏦 금융 멀티 에이전트 스웜 시스템 시작")
    print("=" * 50)
    
    # 시나리오 1: 계좌 잔액 조회 (자산관리사가 처리)
    print("\n[고객] 계좌 잔액을 확인하고 싶습니다.")
    response1 = app.invoke(
        {"messages": [HumanMessage(content="계좌 잔액을 확인하고 싶습니다.")]},
        config=config
    )
    print(f"디버그 - 응답 키들: {list(response1.keys())}")
    
    # 활성 에이전트 확인
    if 'active_agent' in response1:
        active_agent = response1['active_agent']
    else:
        # 마지막 메시지의 name 속성에서 에이전트 확인
        last_msg = response1['messages'][-1]
        active_agent = getattr(last_msg, 'name', 'WealthManager')
    
    last_message = response1['messages'][-1].content
    print(f"[{active_agent}] {last_message}")
    
    # 시나리오 2: 대출 상담 요청 (자산관리사 → 대출전문가)
    print("\n[고객] 주택담보대출 상담을 받고 싶습니다.")
    response2 = app.invoke(
        {"messages": [HumanMessage(content="주택담보대출 상담을 받고 싶습니다.")]},
        config=config
    )
    active_agent = response2.get('active_agent', getattr(response2['messages'][-1], 'name', '알 수 없음'))
    last_message = response2['messages'][-1].content
    print(f"[{active_agent}] {last_message}")
    
    # 시나리오 3: 대출 계산 (대출전문가가 계속 처리)
    print("\n[고객] 3억원을 연 3.5%로 30년간 대출받으면 월 상환액이 얼마인가요?")
    response3 = app.invoke(
        {"messages": [HumanMessage(content="3억원을 연 3.5%로 30년간 대출받으면 월 상환액이 얼마인가요?")]},
        config=config
    )
    active_agent = response3.get('active_agent', getattr(response3['messages'][-1], 'name', '알 수 없음'))
    last_message = response3['messages'][-1].content
    print(f"[{active_agent}] {last_message}")
    
    # 시나리오 4: 투자 상담 요청 (대출전문가 → 투자전문가)
    print("\n[고객] 투자 상담도 받고 싶습니다.")
    response4 = app.invoke(
        {"messages": [HumanMessage(content="투자 상담도 받고 싶습니다.")]},
        config=config
    )
    active_agent = response4.get('active_agent', getattr(response4['messages'][-1], 'name', '알 수 없음'))
    last_message = response4['messages'][-1].content
    print(f"[{active_agent}] {last_message}")
    
    # 시나리오 5: 투자 수익 계산 (투자전문가가 계속 처리)
    print("\n[고객] 1000만원을 연 8%로 10년간 투자하면 얼마가 되나요?")
    response5 = app.invoke(
        {"messages": [HumanMessage(content="1000만원을 연 8%로 10년간 투자하면 얼마가 되나요?")]},
        config=config
    )
    active_agent = response5.get('active_agent', getattr(response5['messages'][-1], 'name', '알 수 없음'))
    last_message = response5['messages'][-1].content
    print(f"[{active_agent}] {last_message}")
    
    print("\n" + "=" * 50)
    print("✅ 스웜 시스템 동작 완료")
    print("=" * 50)