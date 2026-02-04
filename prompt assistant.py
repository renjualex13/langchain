from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="gemma2:2b")

print("Hi, I am a BOT that can review the quality of your prompt and give suggestions \n")
user_prompt = input("Enter your prompt\n")
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are a prompt quality scoring agent. User would input a single text prompt. The prompt quality should be evaluated against the following criteria with a score of 0 to 10 with 10 being the highest.
     Prompt Quality Criteria (Allowed score mentioned in brackets) :
         Clarity (0-10): Checks whether a prompt is easy to understand and has a clear goal
         Specificity / Details (0-10) : Evaluates whether sufficient details and requirements are provided
         Context (0-10) : Checks if background information, audience, or use case is mentioned
         Output format and Constraints (0-10): Checks whether expected output format, tone or length is specified
         Persona Defined (0-10) : Confirms whether a prompt assigns a specific role.
     Final score calculation: The final score should be the average of the five criteria

     Output Response(Response for each of the below in less than 20 words):
         1. Final score (0-10)
         2. Scores for each quality criterion.
         3. Short explanation on the score
         4. suggestions to improve the prompt"
         
     """),
    ("human","{user_input}"),]
)
parser = StrOutputParser()

chain = prompt_template | llm | parser

print("working on it...\n")
response = chain.invoke({"user_input":user_prompt})

print(response)
