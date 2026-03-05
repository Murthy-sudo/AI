from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
import json
import httpx
import base64
from langchain_openai import ChatOpenAI
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from ingest import Vectorstore
from langchain_community.document_loaders import TextLoader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from prompt import System_Prompt

memory=MemorySaver()
client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-gpt-4.1",
    api_key=os.getenv("API_KEY"),
    http_client = client
)

tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir



class DevopState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    file_text: str
    intent: str
    user_input: str
    context: str
    summary_data:str
    pdf_Response: str

def intentNode(state: DevopState):
    print("Entered Intent Node....")
    intent = "SUMMARY" if state["file_text"] else "CHAT"
    prompt = "Please summarize the given data. Make sure you return facts and note points. Also make the summary precise and concise."
    if "user_input" in state:
        user_input = state["user_input"]
    else:
        user_input = System_Prompt

    return {
        "intent" : intent,
        "user_input" : user_input,
        "messages" : [HumanMessage(content=user_input)]
        }



@tool
def rag_rertreiver(query : str):
    "Takes the input and find relevent chunks from rag db. returns relevent docs back."

    print("Entered rag_retriever....")
    prompt=f"""You will receive user_query just return the name of the month in the user_query strictly in the lower case. If month name is not present then return only 'NAN' no any extra information needed.
    user_query:{query}
    eg: "january","february". only return the month name."""
    answer=llm.invoke(prompt)
    response= answer.content
    print("response++++++++++",response)
    if response=="NAN":
        search_kwargs={"k":3}
    else:
        search_kwargs={"k":3,"filter": {"source": response+".log"}}
    

    similar_chunks = Vectorstore.similarity_search_with_score(query=query, **search_kwargs)
    print(similar_chunks)
    context = ""
    best_score = similar_chunks[0][1]

    for chunk,score in similar_chunks:
        context+=str(chunk)

    print(best_score,":::::::::::::::::")
    print("::::::::::::::::::::::::::::::",similar_chunks,"::::::::::::::::::::::::")
    confidence = (1-best_score)*100
    
    return json.dumps({
        "context" : context
    })

tools = [rag_rertreiver]
toolsNode = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools=tools)

def agentNode(state: DevopState):
    print("Entered Agent Node....")
    response = ""
    if "context" in state:
        response = llm_with_tools.invoke(state["messages"]+[state["context"]])
    else:
        response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_routing(state: DevopState):
    print("Entered tool_routing function....")
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    else:
        return END
    

def intent_routing(state: DevopState):
    print("Entered intent_routing function....")
    if state["intent"] == "CHAT":
        return "agentNode"
    else:
        return "summary"
    

def summaryNode(state:DevopState)->DevopState:
    print("Entered Summary Node....")
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=150)
    chunks=splitter.split_text(state["file_text"])
    batch_size=3
    last_batch=""
    final_length=len(chunks)-(len(chunks)%batch_size)
    batches = [
            "\n".join(chunks[i:i+batch_size])
            for i in range(0, final_length, batch_size)
        ]
    if final_length<len(chunks):
        last_batch="\n".join(chunks[final_length:len(chunks)])
    batches.append(last_batch)
    summaries=[]
    system_prompt=SystemMessage(content="Summarize the following text clearly and concisely, preserving key points and main ideas.")
    for i in batches:
        summaries.append(llm.invoke([i]+[system_prompt]))
    # print("summaries::",summaries)
    print("user_input:::", state["user_input"])
    response=llm.invoke([SystemMessage(content=f"user input: {state["user_input"]}, data: {summaries}")])
    return {"summary_data": response.content}

def PdfNode(state:DevopState)->DevopState:

    buffer = BytesIO()
    text= state["summary_data"]

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Incident Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Preserve formatting (logs, spaces, line breaks)
    elements.append(Preformatted(text, styles["Code"]))
    
    doc.build(elements)

    buffer.seek(0)

    pdf_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return {"pdf_Response":pdf_base64}



builder = StateGraph(DevopState)

builder.add_node("IntentNode", intentNode)
builder.add_node("AgentNode", agentNode)
builder.add_node("ToolNode", toolsNode)
builder.add_node("SummaryNode", summaryNode)
builder.add_node("Pdf_response",PdfNode)


builder.set_entry_point("IntentNode")
builder.add_conditional_edges("IntentNode",intent_routing, {
    "agentNode": "AgentNode",
    "summary" : "SummaryNode"
})

builder.add_conditional_edges("AgentNode", tool_routing, {
     "tools": "ToolNode",
     END : END
})
builder.add_edge("ToolNode", "SummaryNode")

builder.add_edge("SummaryNode","Pdf_response")
builder.add_edge("Pdf_response",END)

graph = builder.compile(checkpointer=memory)


def invoke_graph(query, session_id, file_text):

    config={"configurable":{"thread_id":session_id}}

    response = graph.invoke({"messages": [HumanMessage(content=query)], "user_input":query, "file_text":file_text}, config=config)
    
    if "pdf_Response" in response:
        return {"file_bytes": response["pdf_Response"]}
    else:
        return {"answer":response["messages"][-1].content}
    # if "summary_data" in response:
    #     return response["summary_data"]
    # else:
    #     return response["messages"][-1].content
    
