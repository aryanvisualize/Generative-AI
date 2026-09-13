from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",temperature=0.2, max_tokens=50)

response = model.invoke("What is Machine learning, explain?")

print(response.content)