from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv;

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    dimensions=64
)

result = embeddings.embed_query("What is the meaning of life?")

print(result)