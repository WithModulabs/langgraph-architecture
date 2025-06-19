from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="you are a math expert. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="you are a world class researcher with access to web search. Do not any math calculations."
)


# supervisor workflow 
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "you are a team supervisor managing a research expert and a math expert."
        "For current events, use research_agent."
        "for math problems or calculations, use math_agent.`"
        "Always use one tool at a time."
    )
)

# Compile and run 
app = workflow.compile()
result = app.invoke({"messages": [HumanMessage(content="What is the headcount of Meta in 2024?")]})

print(result["messages"][-1].content)


