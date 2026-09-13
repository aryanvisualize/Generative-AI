import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

st.set_page_config(page_title="AI Agent", page_icon="🤖")
st.title("🤖 AI Agent")

model = ChatMistralAI(
    model="ministral-3b-latest",
    temperature=0.9
)

if "mode" not in st.session_state:
    st.session_state.mode = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ended" not in st.session_state:
    st.session_state.ended = False

# ---- Mode selection (equivalent to the console's choose-your-mode prompt) ----
if st.session_state.mode is None:
    st.subheader("Choose your AI mode")
    choice = st.radio(
        "Tell your choice:",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Angry mode", 2: "Funny mode", 3: "Sad mode"}[x],
        index=None,
    )

    if st.button("Confirm") and choice is not None:
        if choice == 1:
            mode = "You are an angry AI agent, You respond aggresively and impatiently."
        elif choice == 2:
            mode = "You are a funny AI agent, you respond with humor and jokes."
        elif choice == 3:
            mode = "You are a sad AI agent, you respond with sorrow and sadness."

        st.session_state.mode = mode
        st.session_state.messages = [SystemMessage(content=mode)]
        st.rerun()

# ---- Chat (equivalent to the console's while-loop) ----
else:
    st.caption("------------------Welcome, type 0 to exit the application-----------------------")

    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    if st.session_state.ended:
        st.info("Session ended (you typed 0). Refresh the page to start again.")
    else:
        prompt = st.chat_input("You: ")

        if prompt is not None:
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            if prompt == "0":
                st.session_state.ended = True
                st.rerun()
            else:
                response = model.invoke(st.session_state.messages)
                st.session_state.messages.append(AIMessage(content=response.content))
                with st.chat_message("assistant"):
                    st.markdown(response.content)