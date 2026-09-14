from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"}
    ),
    Document(
        page_content="Reliability is not just for nuclear power stations and air traffic control software — more mundane applications are also expected to work reliably.",
        metadata={"source": "System_Design_Book"}
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"}
    ),
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vectorStore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result = vectorStore.similarity_search("What is Relational Model?",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)   

retriever = vectorStore.as_retriever()

docs = retriever.invoke("Explain deep learning")