from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Tu código LangChain aquí (chat, retriever, prompt, retrieval_chain, etc.)
# 👇 Lo pegaremos completo en el siguiente paso.
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from semantic_filter2 import quitar_redundancia_respetando_contenido
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="emb", embedding_function=embeddings)

retriever = RedundantFilterRetriever(
    chroma=db,
    filter_func=quitar_redundancia_respetando_contenido
)

prompt = PromptTemplate.from_template(
    """Use the following context to answer the question.

    Context:
    {context}

    Question:
    {question}"""
)

retrieval_chain = ( 
    RunnableMap({
        "context" : RunnableLambda(lambda x: retriever.invoke(x["question"])),
        "question": lambda x: x["question"]
    })
    | prompt
    | chat
    | StrOutputParser()
)

app = FastAPI()

# Permite que otros dominios (como tu web WordPress) hagan peticiones
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://redespoder.com",
        "https://www.redespoder.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/preguntar")
async def preguntar(data: Question):
    resultado = retrieval_chain.invoke({"question": data.question})
    return {"respuesta": resultado}
