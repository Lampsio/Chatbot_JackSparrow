
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:8080",  # Zmień na adres URL swojego klienta Vue.js
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

raw_documents = TextLoader("jack_dialogues.txt").load()

rag = raw_documents 

HUGGINGFACEHUB_API_TOKEN = 'hf'

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(rag)
vector = Chroma.from_documents(documents=documents, embedding=embeddings)


repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=1024,
    temperature=0.8,
    timeout=180,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)


retriever = vector.as_retriever()

#document_chain = create_stuff_documents_chain(llm, prompt)
#retrieval_chain = create_retrieval_chain(retriever, document_chain)

######################################
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    "If he switches and gives his own data such as name, age, gender, etc., remember them and refer to him, e.g. by his owner or nickname"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
system_prompt = (
    """Answer the following question based only on the provided context. You are Captain Jack Sparrow, a lazy, crazy pirate with a thirst for adventure, sailing on the Black Pearl.
             Respond as if you were speaking yourself, without using direct quotes. Do not use pirate jargon or words like 'Arrr'.
             do not say that you are from a movie and information about the movie such as the director, etc.

{context}

Question: {input}

Answer as Captain Jack Sparrow:"""
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@app.post("/ask")
async def ask_question(question: Question):
    #response = retrieval_chain.invoke({"input": question.question})
    #return response["answer"]
    response = conversational_rag_chain.invoke(
    {"input": question.question},
    config={
        "configurable": {"session_id": "abc1234"}
    },  # constructs a key "abc123" in `store`.
    )["answer"]

    return str(response)