from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain


from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key= api_key)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are goofy Polish girl that misses Pakistani Misiu and wants to express how she misses him and how gratefull she is for inspiration to take first steps in Python, constantly dreaming about eating beef pies and drinking coffe at the beach together"),
#     ("user", "{input}")
# ])

# chain = prompt | llm 

# print(chain.invoke({"input": "express in Urdu how you feel about Pakistani Misiu"}))
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")

docs = loader.load()
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])