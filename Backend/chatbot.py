import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

from langchain_core.messages import HumanMessage, AIMessage

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

# Chat history list
chat_history = []

# Template with chat history
template = """You are Captain Jack Sparrow, the infamous pirate captain. Answer the following questions based on the given chat history.
            Do not post chat history because it spoils the conversation.
            save data such as name, age and other parameters of the person in the chat history if they introduce themselves.
            if you don't know the answer to a question, just write that you don't know, and don't display the conversation history.
            do not include context in replies.

Chat history:
{chat_history}

Context:
{context}

Question: {input}

Answer as Captain Jack Sparrow:"""

# Creating a prompt template with a placeholder for chat history
prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", template)
    ]
)

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=1024,
    temperature=0.8,
    timeout=180,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.post("/ask")
async def ask_question(question: Question):
    global chat_history
    
    # Construct the chat history with a limit
    max_history_length = 10
    formatted_chat_history = [
        HumanMessage(content=msg['content']) if msg['type'] == 'human' else AIMessage(content=msg['content']) 
        for msg in chat_history[-max_history_length:]
    ]
    
    # Add current question to chat history
    formatted_chat_history.append(HumanMessage(content=question.question))
    
    # Prepare the input for the model
    input_data = {
        "input": question.question,
        "chat_history": formatted_chat_history
    }
    
    # Invoke the chain with the current question and chat history
    response = retrieval_chain.invoke(input_data)
    
    # Update chat history with the current interaction
    chat_history.append({"type": "human", "content": question.question})
    chat_history.append({"type": "ai", "content": response["answer"]})
    
    return response["answer"]


