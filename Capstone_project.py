from IPython.display import Image, display
from typing import TypedDict, Annotated, List, Literal
import operator
from langgraph.graph import START,END,StateGraph
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
import uuid
import ollama
from pydantic import BaseModel
from langchain_ollama import ChatOllama
#
#Define State Schemas

#We would need two state schemas. 1. For Email Classification 2. Email Agent State

class Emailclassifier(BaseModel):
    urgency : Literal["Low","Medium","High"]
    topic : Literal["Account","Billing", "Bug", "Feature", "Technical Issue"]
    summary : str

class EmailState(TypedDict):
    email_content : str
    sender_id : str

    classification : Emailclassifier | None

    bug_ticket : str | None

    feature_id : str | None

    search_results : list[str] | None 

    draft_response : str | None
    edited_response : str | None
#
#Define Nodes. 
#We would need multiple nodes. 1. Read email. 2. Classify Email. 3. Bug report 
#4. New Feature. 5.Search. 6. Humanintervention 7. Draft Response 8. Send Response

def read_email(state : EmailState)->EmailState:
    """
    There will be logic to read the email from the inbox and parse it
    """
    pass

def classify_email(state : EmailState)->Command [Literal["bug_report","new_feature","search_results","human_intervention"]]:
    """
    The email content will be structured using Pydantic model in Ollama 
    and the same will be used to classify 
    """
    prompt = [{"role": "system","content":"You are a helpful email assistant. You can classify email based on the user provided content"},
              {"role": "user","content":f"""Analyze this customer email and classify it:

    Email: {state['email_content']}
    From: {state['sender_id']}

    Provide classification, including topic,urgency based on the allowed Literals"""}]
    
    response = ollama.chat(
        model='gemma2:2b', # Use a capable model
        messages=prompt,
        format=Emailclassifier.model_json_schema()
        ) # Force JSON structure

    classification = Emailclassifier.model_validate_json(response.message.content)

    print("The email can be classified as ", classification)
    if classification.urgency == "High":
        next_node = "human_intervention"
    else:
        match classification.topic:
            case "Bug":
                next_node = "bug_report"
            case "Feature":
                next_node = "new_feature"
            case "Technical Issue":
                next_node = "search_results"
            case _:
                next_node = "human_intervention"

    return Command(
        goto = [next_node],
        update = {'classification': classification}
    )

def bug_report(state: EmailState) -> EmailState:
    """Create a bug ticket"""

    # Create ticket in your bug reporting system
    bug_ticket = f"BUG_{uuid.uuid4()}"

    return {"bug_ticket": bug_ticket}

def new_feature(state: EmailState) -> EmailState:
    """Create a new feature"""

    # Create a feature ticket
    feature_id = f"FTR_{uuid.uuid4()}"

    return {"feature_id": feature_id}

def search_results(state: EmailState)->EmailState:
    """Implement the search logic here. In this case we are returning sample search results"""

    search_rslts = [
        "Search results 1",
        "Search results 2"
    ]
    return {'search_results':search_rslts}

def human_intervention(state: EmailState)->Command[Literal["draft_response","send_reply"]]:
    """Implement the search logic here. In this case we are returning sample search results"""
    
    mail_classification = state.get('classification',{})
    draft_response = ""
    human_decision=interrupt({
        "email":state['sender_id'],
        "email content" : state['email_content'],
        "topic" : mail_classification.topic,
        "urgency" : mail_classification.urgency
    })
    print("hum decision is:",human_decision.get('Approval'))
    if human_decision.get('Approval') == "Y":
        next_node = "draft_response"
        
    else:
        print("Entered False")
        edited_resp = human_decision.get('Edited_response')
        draft_response = edited_resp
        next_node = "send_reply"

    return Command(
        goto=[next_node],
        update={'draft_response':draft_response}
    )
    
def draft_response(state:EmailState)->EmailState:

    classification = state.get('classification',{})
    
    prompt = [{"role": "system","content":"You are a helpful email assistant. You can draft a courteous email based on the user provided content"},
              {"role": "user","content":f"""Draft a response to this customer email:
    {state['email_content']}

    Email intent: {classification.topic}
    Urgency level: {classification.urgency}
    Bug Ticket ID: {state.get('bug_ticket',"")}
    Feature ID: {state.get('feature_id',"")}
    search_results : {state.get('search_results',"")}

    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use the provided documentation when relevant. Based on the topic identified, provide bug ticket, feature ticket or search result. The response should be tailored to the topic itself.
    - Be brief"""}]

    llm = ChatOllama(model='gemma2:2b')
    response = llm.invoke(prompt)
    #response = ollama.chat(
    #    model='gemma2:2b',
    #    messages=prompt
    #)
    return {'draft_response':response.content}

def send_reply(state: EmailState) -> EmailState:
    """Send the email response"""
    # Integrate with a email service
    print(f"Sending reply: {state['draft_response']}")
    return {}

#
#Build Graph.
builder = StateGraph(EmailState)

builder.add_node("read_email",read_email)
builder.add_node("classify_email",classify_email)
builder.add_node("bug_report",bug_report)
builder.add_node("new_feature",new_feature)
builder.add_node("search_results",search_results)
builder.add_node("human_intervention",human_intervention)
builder.add_node("draft_response",draft_response)
builder.add_node("send_reply",send_reply)

builder.add_edge(START,"read_email")
builder.add_edge("read_email","classify_email")
builder.add_edge("bug_report","draft_response")
builder.add_edge("new_feature","draft_response")
builder.add_edge("search_results","draft_response")
builder.add_edge("draft_response","send_reply")
builder.add_edge("send_reply",END)

memory = InMemorySaver()
app = builder.compile(checkpointer = memory)

display(Image(app.get_graph().draw_mermaid_png()))
#
# TESTING........
#Testing
# Test with urgent billing issue
initial_state = {
    "email_content": "I received the notification twice. ",
    "sender_id": "customer@example.com"
}

# Run with a thread_id for persistence
config = {"configurable": {"thread_id": "customer_123"}}
result = app.invoke(initial_state, config)

if '__interrupt__' in result:
    app_dec = input("Do you Approve? Y for Yes| N for No")
    edit_resp = ""
    if app_dec == "N":
        edit_resp = input("Enter your edited response")
    human_decision = Command(
        resume={
            "Approval": app_dec,
            "Edited_response": edit_resp
        }
    )
    result=app.invoke(human_decision,config)
    print("Sent email successfully")