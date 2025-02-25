from  langchain_openai import ChatOpenAI
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from dotenv import load_dotenv
from uuid import uuid4
import json
import os

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# read from josn file
with open("./urls_canada.json", "r") as file:
    urls = json.load(file)

filtered_urls = [url for url in urls if "/immigration" in url]

batch_size = 5
start_index = 0  # Change this for the next batch (10, 20, 30, etc.)
current_to_process = filtered_urls[start_index:start_index + batch_size]

loader = SeleniumURLLoader(urls=urls)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=300,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)


uuids = [str(uuid4()) for _ in range(len(all_splits))]

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
)

vectorstore.add_documents(all_splits, ids=uuids)
