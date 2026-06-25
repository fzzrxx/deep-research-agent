from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini", temperature= 0)

tools = [TavilySearchResults(max_results=3)]

agent = create_react_agent(llm, tools)

result = agent.invoke({"messages": [{"role": "user", "content": "What is Langgraph?"}]})

print(result["messages"][-1].content)