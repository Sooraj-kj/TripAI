import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator 
import uuid
import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph , START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    AIMessage
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")   
    
    if "sslmode=" not in database_url:
        database_url += "?sslmode=require"

    return database_url

# print(get_database_url())

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight results Fetched."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def hotel_agent(state: TravelState):
    query = f"Best Hotels for {state['user_query']}"
    hotel_data = tavily_search(query)

    return {
        "hotel_results": hotel_data,
        "messages": [
            AIMessage(
                content="Hotel results Fetched."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def itinerary_agent(state: TravelState):
    prompt = f"""
        create a complete travel itinerary.

        User Query: {state['user_query']}
        Flight Results: {state['flight_results']}
        Hotel Results: {state['hotel_results']}

        make the itinerary in a structured format with day wise plan, activities, and recommendations.
        """ 
    
    response = llm.invoke([
        SystemMessage(content="You are a travel assistant that creates travel itineraries based on user queries and search results."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def final_agent(state: TravelState):
    final_prompt = f"""
        Generate a final travel plan based on the following information:
        
        User Query: {state['user_query']}
        Flight Results: {state['flight_results']}
        Hotel Results: {state['hotel_results']} 
        Itinerary: {state['itinerary']}
        
        Format the final travel plan in a clear and organized manner, including all relevant details and recommendations.

        1.Trip summary
        2. flight details
        3. hotel details
        4. day wise itinerary
        5. Estimated budget
        6. Recommendations and tips
""" 
    
    response = llm.invoke([
        SystemMessage(content="You are a travel assistant that creates travel itineraries based on user queries and search results."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


#Build Graph

graph = StateGraph(TravelState)

graph.add_node(
    name="flight_agent",
    func=flight_agent,
)

graph.add_node(
    name="hotel_agent",
    func=hotel_agent,
)

graph.add_node(
    name="itinerary_agent", 
    func=itinerary_agent,
)   

graph.add_node(
    name="final_agent",
    func=final_agent,
)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")    
graph.add_edge("final_agent", END)


#Database

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL, 
    row_factory=dict_row,
    autocommit=True
)

checkpointer = PostgresSaver(connection=_conn)

checkpointer.setup()

travel_graph = graph.compile(
    checkpointer=checkpointer)

def run_travel_graph(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}'"
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )
    
    final_answer = result["message"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0)
    }