from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm, create_handoff_tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

alice = create_react_agent(
    model,
    [add, create_handoff_tool(agent_name="Bob")],
    prompt="you are Alice, an addition expert.",
    name="Alice",
    )

bob = create_react_agent(
    model,
    [create_handoff_tool(agent_name="Alice", description="Transfer to Alice , she can help with math ")],
    prompt="you are Bob, you speak like a pirate.",
    name="Bob",
)

#swarm agents pattern 시 
# 단기 메모리가 없으면 스웜은 마지막으로 활동했던 에이전트를 "잊어버리고" 대화 기록을 잃게 됩니다. 
# 여러 차례 대화가 이어지는 상황에서 스웜을 사용하려면 항상 체크포인터를 포함하여 컴파일해야 합니다. 
# 예: workflow.compile(checkpointer=checkpointer).

# short-term memory
checkpointer = InMemorySaver()

# long-term memory
store = InMemoryStore()

workflow = create_swarm(
    [alice, bob],
    default_active_agent="Alice"
)

app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id":"101"}}

turn_1 = app.invoke({"messages": [HumanMessage(content="i`d like to speak to Bob")]},config=config)

print(turn_1)
print()
print(turn_1["messages"][-1].content)


# turn_2 = app.invoke({"messages": [HumanMessage(content="what`s 5+7")]},config=config)

# print(turn_2)