# deep-research-agent
Agentic LLM pipeline that autonomously plans, researches, and synthesizes structured reports from the web. Built with LangGraph, LangChain, Tavily, and OpenAI.

# Deep Research Agent

An agentic LLM pipeline built with LangGraph and LangChain that autonomously plans, researches, and synthesizes structured reports from the web.

## How it works

The agent runs as a multi-node graph with four stages:

1. **Planner** — breaks the user's query into focused search tasks using structured output
2. **Researcher** — executes web searches in a reasoning loop using Tavily
3. **Critic loop** — the agent decides whether it has enough information or needs to search again
4. **Writer** — synthesizes all findings into a structured report with citations and follow-up questions

Built on [LangGraph](https://github.com/langchain-ai/langgraph) — each stage is a node in a `StateGraph`, connected by conditional edges that control the research loop.

## Output format

Every run produces a structured `ResearchReport` object validated by Pydantic:

```
title
summary
key_findings     (list)
citations        (url + key point)
follow_up_questions (list)
```

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent graph orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — LLM interface and tool integration
- [OpenAI GPT-4o-mini](https://platform.openai.com) — reasoning and structured output
- [Tavily](https://tavily.com) — real-time web search
- [Pydantic](https://docs.pydantic.dev) — output schema validation

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/fzzrxx/deep-research-agent.git
cd deep-research-agent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API keys**
```bash
cp .env.example .env
```
Then open `.env` and fill in your `OPENAI_API_KEY` and `TAVILY_API_KEY`.

- OpenAI key: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Tavily key: [app.tavily.com](https://app.tavily.com) (free tier: 1,000 searches/month)

**4. Run**
```bash
python graph.py
```

## Project structure

```
deep-research-agent/
├── graph.py          # Full agent graph — nodes, edges, state, structured output
├── .env.example      # API key template
├── requirements.txt  # Python dependencies
└── README.md
```

## What I learned building this

- How to design a `StateGraph` in LangGraph with conditional edges and loops
- The difference between a fixed chain and a dynamic agent graph
- How `bind_tools` signals tool availability to the LLM
- How `with_structured_output` enforces a Pydantic schema on LLM responses
- The two-LLM pattern — separating the reasoning LLM from the writing LLM
