'''
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

code_prompt = ChatPromptTemplate.from_template("You are a coding assistant. Question: {q}")
gen_prompt  = ChatPromptTemplate.from_template("You are a general helper. Question: {q}")

def is_code(x): 
    return "python" in x["q"].lower() or "bug" in x["q"].lower()

router = RunnableBranch(
    (is_code, code_prompt),
    gen_prompt  # fallback
)

chain = router | llm | StrOutputParser()

print(chain.invoke({"q": "Why is my Python loop slow?"}))
'''

'''
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Model (use any chat model ID you have access to)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Prompt with a slot for conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# Simple per-session history store
store = {}
def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

session_cfg = {"configurable": {"session_id": "demo"}}

print(chat.invoke({"input": "Hello, my name is Alex."}, config=session_cfg))
print(chat.invoke({"input": "What’s my name?"}, config=session_cfg))
'''
