import streamlit as st
import json
import _snowflake
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete

# -- Snowflake session --
session = get_active_session()

# -- Page Title --
st.title("Chat with your Structured Data :balloon:")

API_ENDPOINT = "/api/v2/cortex/analyst/message"
API_TIMEOUT = 50000  # in milliseconds

# ------------------------------------------------------------------
# Initialize st.session_state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []  # For our UI chat display
if "analyst_conversation" not in st.session_state:
    st.session_state.analyst_conversation = []  # For sending to Cortex Analyst

def get_sql_from_cortex_analyst():
    """
    Sends the entire analyst_conversation in the request to preserve context.
    """
    request_body = {
        "messages": st.session_state.analyst_conversation,  # full conversation so far
        "semantic_model_file": '@"CORTEX_ANALYST_DEMO"."REVENUE_TIMESERIES"."RAW_DATA"/revenue_timeseries.yaml',
    }
    
    resp = _snowflake.send_snow_api_request(
        "POST",
        API_ENDPOINT,
        {},
        {},
        request_body,
        None,
        API_TIMEOUT,
    )
    return json.loads(resp["content"])


def parse_analyst_messages(analyst_output):
    """
    Pull text/sql items from a valid "analyst" output 
    or return "error" if we detect an error shape.
    """
    # Valid shape => {"message": {"content": [{"type":"text","text":"..."},...]}}
    if isinstance(analyst_output, dict) and "message" in analyst_output:
        contents = analyst_output["message"].get("content", [])
        texts = []
        statements = []
        for item in contents:
            if item.get("type") == "text":
                texts.append(item.get("text", "").strip())
            elif item.get("type") == "sql":
                statements.append(item.get("statement", "").strip())
        return {"status": "ok", "texts": texts, "statements": statements}

    # Error shape => typically a list with error info
    if isinstance(analyst_output, list) and analyst_output:
        raw_content = analyst_output[0].get("content", "").strip()
        return {"status": "error", "error_message": raw_content}

    return {"status": "error", "error_message": "Could not parse analyst output"}


def answer_question_using_analyst(query: str):
    """
    1) Append the user question to st.session_state.analyst_conversation
    2) Call Cortex Analyst with the full conversation
    3) Parse the result & append a new "analyst" message
    4) Execute the returned SQL (if any) & generate final answer
    5) Return final LLM messages
    """

    # -- STEP 1: Add user query to the conversation
    st.session_state.analyst_conversation.append({
        "role": "user",
        "content": [{"type": "text", "text": query}]
    })

    # -- STEP 2: Call Cortex Analyst
    with st.spinner("Calling Cortex Analyst to generate SQL..."):
        analyst_output = get_sql_from_cortex_analyst()
        parsed = parse_analyst_messages(analyst_output)

    if parsed["status"] == "error":
        st.error("Error from Analyst:")
        st.error(parsed["error_message"])
        return [{"role": "assistant", "content": f"**Error**: {parsed['error_message']}"}]

    # Convert texts/statements into a single "analyst" message
    # so we can maintain the conversation with the Analyst
    analyst_message_content = []
    for txt in parsed["texts"]:
        analyst_message_content.append({"type": "text", "text": txt})
    for stmt in parsed["statements"]:
        analyst_message_content.append({"type": "sql", "statement": stmt})

    # Append that new "analyst" message to the conversation
    st.session_state.analyst_conversation.append({
        "role": "analyst",
        "content": analyst_message_content
    })

    if parsed["statements"]:
        # Display the analyst's text items as status messages
        for txt in parsed["texts"]:
            st.info(txt)

    # If no SQL statements, stop here
    else:
        for txt in parsed["texts"]:
            st.write(txt)
        return None

    # Let's pick the first statement
    sql_statement = parsed["statements"][0]
    with st.expander("Generated SQL Statement"):
        st.code(sql_statement, language="sql")

    # -- STEP 3: Execute SQL
    try:
        with st.spinner("Executing SQL..."):
            sql_output = session.sql(sql_statement.replace(";", ""))
    except Exception as e:
        st.error(f"Error executing SQL: {e}")
        return [{"role": "assistant", "content": f"**Error**: Failed to execute SQL. {e}"}]

    # Show results
    with st.expander("See SQL Results"):
        try:
            st.write(sql_output)
        except Exception as e:
            st.error(f"Error displaying SQL results: {e}")
            return [{"role": "assistant", "content": f"**Error**: Couldn't display SQL results. {e}"}]

    # -- STEP 4: Generate final answer from SQL results
    try:
        with st.spinner("Generating final answer from SQL results..."):
            df = sql_output.to_pandas()
            markdown_sql_output = df.to_markdown(index=False)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that uses SQL output to answer questions."
                },
                {
                    "role": "user",
                    "content": (
                        f"The user asked: {query}\n\n"
                        f"The SQL results are:\n{markdown_sql_output}\n\n"
                        "Please answer the question concisely, without extra details."
                    )
                }
            ]
            options = {"guardrails": True}
            response = complete("claude-3-5-sonnet", messages, options=options, stream = True)
    except Exception as e:
        st.error(f"Error generating final answer: {e}")
        return [{"role": "assistant", "content": f"**Error**: Unable to generate final answer. {e}"}]

    return response


def display_messages():
    """
    Renders the entire UI chat from st.session_state.messages
    """
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            st.chat_message("user").write(content)
        elif role == "assistant":
            st.chat_message("assistant", avatar="🤖").write(content)

if st.button("Clear Conversation"):
    st.session_state.messages.clear()
    st.session_state.analyst_conversation.clear()

# Show existing conversation
display_messages()

# Prompt user
user_input = st.chat_input("What is the highest revenue in each sales region?")

if user_input:
    # Show user's message right away
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate streaming tokens from the LLM
    ai_answer_generator = answer_question_using_analyst(user_input)  # must be a generator

    final_text = st.chat_message("assistant", avatar = "🤖").write_stream(ai_answer_generator)

    # Store the completed text so it remains on subsequent runs
    st.session_state.messages.append({"role": "assistant", "content": final_text})