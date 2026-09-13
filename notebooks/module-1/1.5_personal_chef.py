from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}
agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[web_search],
    checkpointer=InMemorySaver(),
    system_prompt=system_prompt
)
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [HumanMessage(content="I have chicken, rice, and broccoli. What can I make with these ingredients?")]},
    config)
print(response['messages'][-1].content)


from pprint import pprint

pprint(response)