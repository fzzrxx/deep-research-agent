from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage
from typing import Annotated, TypedDict, Sequence
from langgraph.graph.message import add_messages
from pydantic import BaseModel

class ResearchPlan(BaseModel):
    sub_tasks: list[str]
    search_queries: list[str]

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [TavilySearchResults(max_results=3)]

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
    return "end"


planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
planner_llm = planner_llm.with_structured_output(ResearchPlan)

def planner_node(state: AgentState) -> dict:
    query = state["messages"][0].content  # the original user question
    plan = planner_llm.invoke(
        f"Break this research query into sub-tasks and search queries: {query}"
    )
    # turn each sub-task into a message the agent can act on
    task_summary = "\n".join(plan.search_queries)
    return {"messages": [HumanMessage(content=f"Research these queries:\n{task_summary}")]}





builder = StateGraph(AgentState)


builder.add_node("planner", planner_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "agent",                          
    should_use_tool,                  
    {"use_tool": "tools", "end": END} 
)

builder.add_edge("planner", "agent")  

graph = builder.compile()


result = graph.invoke({
    "messages": [{"role": "user", "content": "Tell me more about Langchain."}]
})

for message in result["messages"]:
    print(f"\n--- {message.type.upper()} ---")
    print(message.content)