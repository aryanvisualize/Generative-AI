from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "Hello this is Aryan Rastogi",
    "Its really nice to meet you",
    "I love to code and learn new things"
]

vector = embedding.embed_documents(texts)
print(vector)