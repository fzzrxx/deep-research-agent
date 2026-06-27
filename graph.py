from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage
from typing import Annotated, TypedDict, Sequence
from langgraph.graph.message import add_messages
from pydantic import BaseModel

class ResearchPlan(BaseModel):
    sub_tasks: list[str]
    search_queries: list[str]

class Citation(BaseModel):
    source_url: str
    key_point: str

class ResearchReport(BaseModel):
    title: str
    summary: str
    key_findings: list[str]
    citations: list[Citation]
    follow_up_questions: list[str]


load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    report: ResearchReport | None


tools = [TavilySearch(max_results=3)]

llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)

llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}









tool_node = ToolNode(tools) 


def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "write"


planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
planner_llm = planner_llm.with_structured_output(ResearchPlan)

writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
writer_llm = writer_llm.with_structured_output(ResearchReport)

def planner_node(state: AgentState) -> dict:
    query = state["messages"][0].content  # the original user question
    plan = planner_llm.invoke(
        f"Break this research query into sub-tasks and search queries: {query}"
    )
    # turn each sub-task into a message the agent can act on
    task_summary = "\n".join(plan.search_queries)
    return {"messages": [HumanMessage(content=f"Research these queries:\n{task_summary}")]}


def writer_node(state: AgentState) -> dict:
    print("WRITER NODE REACHED")
    research_content = "\n\n".join(
        m.content for m in state["messages"]
        if hasattr(m, "content") and m.content
    )
    report = writer_llm.invoke(
        f"Based on this research, produce a structured report:\n\n{research_content}"
    )
    print(f"REPORT PRODUCED: {report}")
    return {"report": report}




builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("writer", writer_node)
builder.set_entry_point("planner")
builder.add_edge("planner", "agent")
builder.add_edge("tools", "agent")    
builder.add_conditional_edges(
    "agent",
    should_use_tool,
    {"use_tool": "tools", "write": "writer"}
)
builder.add_edge("writer", END)
graph = builder.compile()


result = graph.invoke({
    "messages": [{"role": "user", "content": "Tell me more about Langchain."}],
    "report": None
})

report = result["report"]
print(f"\nTITLE: {report.title}")
print(f"\nSUMMARY: {report.summary}")
print(f"\nKEY FINDINGS:")
for finding in report.key_findings:
    print(f"  - {finding}")
print(f"\nCITATIONS:")
for citation in report.citations:
    print(f"  - {citation.key_point} ({citation.source_url})")
print(f"\nFOLLOW-UP QUESTIONS:")
for question in report.follow_up_questions:
    print(f"  - {question}")

for message in result["messages"]:
    print(f"\n--- {message.type.upper()} ---")
    print(message.content)