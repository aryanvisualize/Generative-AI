import os
import shutil
import streamlit as st

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from create_database import create_vectorstore


load_dotenv()


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Book RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.chat-user {
    background-color: #e8f0fe;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 10px 0;
}

.chat-ai {
    background-color: #f5f5f5;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 10px 0;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "book_name" not in st.session_state:
    st.session_state.book_name = None


# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

@st.cache_resource
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# LLM
# --------------------------------------------------

@st.cache_resource
def get_llm():

    return ChatMistralAI(
        model="ministral-3b-latest"
    )


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📚 Book RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a book and ask questions about its contents.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📖 Upload Book")

    uploaded_file = st.file_uploader(
        "Upload a PDF book",
        type=["pdf"]
    )

    if uploaded_file:

        st.success(f"Selected: {uploaded_file.name}")

        if st.button(
            "⚙️ Process Book",
            use_container_width=True
        ):

            with st.spinner(
                "Processing book... This may take a while."
            ):

                # Create temporary directory
                os.makedirs("uploaded_books", exist_ok=True)

                pdf_path = os.path.join(
                    "uploaded_books",
                    uploaded_file.name
                )

                # Save uploaded PDF
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Remove old database
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db")

                # Create vector database
                vectorstore, pages, chunks = create_vectorstore(
                    pdf_path
                )

                # Save in session
                st.session_state.vectorstore = vectorstore
                st.session_state.book_name = uploaded_file.name

                # Clear previous conversation
                st.session_state.messages = []

                st.success("Book processed successfully!")

                st.info(
                    f"📄 Pages: {pages}\n\n"
                    f"🧩 Chunks: {chunks}"
                )


    st.divider()

    if st.session_state.book_name:

        st.subheader("Current Book")

        st.write(
            f"📕 {st.session_state.book_name}"
        )

        if st.button(
            "🗑️ Clear Book",
            use_container_width=True
        ):

            st.session_state.vectorstore = None
            st.session_state.book_name = None
            st.session_state.messages = []

            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")

            st.rerun()


# --------------------------------------------------
# Main Chat Area
# --------------------------------------------------

if st.session_state.vectorstore is None:

    st.info(
        "👈 Upload a PDF book from the sidebar "
        "and click **Process Book** to start."
    )

else:

    st.success(
        f"Currently reading: **{st.session_state.book_name}**"
    )

    # Create retriever
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # Load LLM
    llm = get_llm()

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful AI Assistant.

                Answer the user's question using ONLY
                the provided context from the uploaded book.

                Do not use outside knowledge.

                If the answer is not present in the context,
                say exactly:

                "I could not find the answer in the document."
                """
            ),

            (
                "human",
                """
                Context:
                {context}

                Question:
                {question}
                """
            )
        ]
    )

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    # Chat input
    query = st.chat_input(
        "Ask something about the book..."
    )


    if query:

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):

            st.markdown(query)


        # AI response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                # Retrieve relevant documents
                docs = retriever.invoke(query)

                # Build context
                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc in docs
                    ]
                )

                # Create prompt
                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": query
                    }
                )

                # Get response
                response = llm.invoke(
                    final_prompt
                )

                answer = response.content

                st.markdown(answer)

        # Save AI response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )