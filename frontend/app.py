import streamlit as st
import requests
import pandas as pd

# Page config
st.set_page_config(
    page_title="Bhavana's Tokenizer",
    page_icon="🔤",
    layout="wide"
)

# API Base URL
API_URL = "http://localhost:8000"  # Change to your Render URL after deployment

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #10a37f;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .token-box {
        padding: 5px 10px;
        margin: 3px;
        border-radius: 5px;
        display: inline-block;
        background-color: #e8f4f8;
        border: 1px solid #10a37f;
    }
    .stats-box {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔤 BHAVANA\'s Tokenizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ChatGPT-Style Text Tokenization Tool</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Tokenizer", "📚 Vocabulary", "📊 Stats"])

# ==================== TAB 1: TOKENIZER ====================
with tab1:
    st.header("Tokenize Your Text")
    
    # Model selection
    model = st.selectbox(
        "Select Model:",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
    
    # Text input
    user_text = st.text_area(
        "Enter text to tokenize:",
        placeholder="Type or paste your text here...",
        height=150
    )
    
    # Tokenize button
    if st.button("🚀 Tokenize", type="primary"):
        if user_text.strip():
            try:
                # Call API
                response = requests.post(
                    f"{API_URL}/tokenize",
                    json={"text": user_text, "model": model}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Tokens", data["count"])
                    with col2:
                        st.metric("Characters", len(user_text))
                    with col3:
                        st.metric("Model", data["model"])
                    
                    st.divider()
                    
                    # Display tokens
                    st.subheader("Tokenized Output")
                    tokens_html = ""
                    for token in data["token_strings"]:
                        # Escape special characters for display
                        display_token = token.replace(" ", "␣").replace("\n", "↵")
                        tokens_html += f'<span class="token-box">{display_token}</span>'
                    
                    st.markdown(tokens_html, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Token details table
                    st.subheader("Token Details")
                    df = pd.DataFrame(data["token_details"])
                    st.dataframe(df, use_container_width=True)
                    
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to API: {str(e)}")
        else:
            st.warning("Please enter some text to tokenize!")

# ==================== TAB 2: VOCABULARY ====================
with tab2:
    st.header("📚 Token Vocabulary Explorer")
    st.write("Browse the complete vocabulary of ~100,000 tokens used by ChatGPT models")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input(
            "🔍 Search vocabulary:",
            placeholder="Search for a token..."
        )
    
    with col2:
        limit = st.number_input(
            "Results per page:",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )
    
    start = st.number_input(
        "Start from token ID:",
        min_value=0,
        max_value=100000,
        value=0,
        step=100
    )
    
    if st.button("🔎 Load Vocabulary", type="primary"):
        try:
            response = requests.post(
                f"{API_URL}/vocabulary",
                json={
                    "start": start,
                    "limit": limit,
                    "search": search_term
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Vocabulary", f"{data['total_size']:,}")
                with col2:
                    st.metric("Showing", f"{data['count']}")
                with col3:
                    st.metric("Start ID", data['showing_start'])
                with col4:
                    st.metric("End ID", data['showing_end'])
                
                st.divider()
                
                # Display vocabulary
                if data['vocabulary']:
                    df = pd.DataFrame(data['vocabulary'])
                    
                    # Style the dataframe
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=600,
                        column_config={
                            "token_id": st.column_config.NumberColumn(
                                "Token ID",
                                help="Unique identifier for this token"
                            ),
                            "token_string": st.column_config.TextColumn(
                                "Token String",
                                help="The actual text representation"
                            ),
                            "length": st.column_config.NumberColumn(
                                "Length",
                                help="Character length of the token"
                            )
                        }
                    )
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"vocabulary_{start}_{start+limit}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No tokens found matching your search criteria.")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to API: {str(e)}")

# ==================== TAB 3: STATS ====================
with tab3:
    st.header("📊 Vocabulary Statistics")
    
    if st.button("📈 Load Stats", type="primary"):
        try:
            response = requests.get(f"{API_URL}/stats")
            
            if response.status_code == 200:
                stats = response.json()
                
                st.markdown(f"""
                <div class="stats-box">
                    <h3>Encoding Information</h3>
                    <p><strong>Encoding Name:</strong> {stats['encoding_name']}</p>
                    <p><strong>Total Tokens:</strong> {stats['total_tokens']:,}</p>
                    <p><strong>Description:</strong> {stats['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("Supported Models")
                for model in stats['supported_models']:
                    st.markdown(f"✅ {model}")
                
                # Visual representation
                st.divider()
                st.subheader("Token Distribution")
                st.info(f"This tokenizer uses the **{stats['encoding_name']}** encoding with **{stats['total_tokens']:,}** unique tokens in its vocabulary.")
                
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to API: {str(e)}")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Built by Bhavana..</p>
        <p>Powered by OpenAI's tiktoken library</p>
    </div>
""", unsafe_allow_html=True)