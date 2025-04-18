import streamlit as st
import textwrap

from snowflake.snowpark.context import get_active_session
session = get_active_session()

# -- Page Title --
st.title("Chat with Cortex Search RAG")

# ------------------------------------------------------------------
# 1. RAG + TruLens Setup
# ------------------------------------------------------------------
from snowflake.core import Root
from typing import List
from snowflake.snowpark.session import Session
import numpy as np

from snowflake.cortex import complete
     
# -- A simple CortexSearchRetriever that queries your search service --
class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 2):
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve

    def retrieve(self, query: str) -> List[str]:
        root = Root(self._snowpark_session)
        # Adjust DB/SCHEMA/SERVICE to match your environment
        search_service = (
            root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
            .schemas["PUBLIC"]
            .cortex_search_services["FOMC_SEARCH_SERVICE"]
        )
        resp = search_service.search(
            query=query,
            columns=["chunk"],
            limit=self._limit_to_retrieve
        )
        if resp.results:
            return [r["chunk"] for r in resp.results]
        else:
            return []

# ------------------------------------------------------------------
# RAG class
# ------------------------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session, limit_to_retrieve=5)

    def retrieve_context(self, query: str) -> List[str]:
        """
        Retrieve relevant text from vector store (instrumented).
        """
        return self.retriever.retrieve(query)

    def build_messages_with_context(self, conversation_messages, context_chunks):
        """
        Takes the entire conversation (in Chat-style messages) and appends
        a new system-level instruction message containing the retrieved context.
        Then returns the updated list of messages.
        """
        # Create a copy so we don't mutate the original list in session state
        updated_messages = list(conversation_messages)

        # Add a system message that includes the context
        context_message_content = (
            f"You have retrieved the following context (do not hallucinate beyond it):\n"
            f"{context_chunks}\n\n"
            "Based on the conversation so far and the context above, please answer the last user question "
            "in a comprehensive, correct, and helpful way. If you don't have the information, just say so."
        )
        updated_messages.append({"role": "system", "content": context_message_content})
        return updated_messages

    def generate_completion_stream(self, messages):
        """
        Stream the response from 'claude-3-5-sonnet', using the entire conversation messages.
        """
        stream = complete(
            "claude-3-5-sonnet",
            messages,
            stream=True
        )
        return stream

# Instantiate the RAG
rag = RAG()

# ------------------------------------------------------------------
# 2. Streamlit Chat Logic
# ------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # For UI chat display (user + assistant)

if st.button("Clear Conversation"):
    st.session_state.messages.clear()

def display_messages():
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            st.chat_message("user").write(content)
        elif role == "assistant":
            st.chat_message("assistant", avatar="🤖").write(content)
        elif role == "system":
            # If you want to display system messages in the UI as well:
            st.chat_message("assistant", avatar="🔧").write(f"[SYSTEM] {content}")

# Render existing messages
display_messages()

def answer_question_using_rag(query: str):
    """
    1) Append the user question to st.session_state.messages.
    2) Retrieve context chunks.
    3) Build a new message array that includes the entire conversation + a system msg w/ context.
    4) Stream the LLM response.
    """
    # -- STEP 1: The user question is already appended in main(), so we have the full conversation in st.session_state.messages.

    # -- STEP 2: Retrieve context
    with st.spinner("Retrieving context..."):
        context_chunks = rag.retrieve_context(query)

    # Show context above LLM answer
    st.write("**Relevant Context Found:**")
    with st.expander("See retrieved context"):
        for chunk in context_chunks:
            wrapped_chunk = textwrap.fill(chunk, width=60)  # Wrap at 60 characters (for example)
            st.info(f"```\n{wrapped_chunk}\n```")

    # -- STEP 3: Build new message array including the entire conversation + a system message w/ context
    updated_messages = rag.build_messages_with_context(st.session_state.messages, context_chunks)

    # -- STEP 4: Stream the final LLM response
    with st.spinner("Generating response..."):
        stream = rag.generate_completion_stream(updated_messages)
    return stream

def main():
    user_input = st.chat_input("Ask your question about FOMC or economic data...")

    if user_input:
        # Step 1: Append user message to st.session_state for the conversation history
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Step 2: Call the RAG pipeline to get a streaming generator
        stream = answer_question_using_rag(user_input)

        # Step 3: Display the final LLM streaming response
        final_text = st.chat_message("assistant", avatar="🤖").write_stream(stream)

        # Step 4: Store final assistant message so it remains in the conversation
        st.session_state.messages.append({"role": "assistant", "content": final_text})

if __name__ == "__main__":
    main()
