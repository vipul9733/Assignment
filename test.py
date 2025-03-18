import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
from collections import Counter
import string

def is_valid_url(url):
    """Check if the URL is valid by attempting to parse it."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def fetch_url_content(url):
    """Fetch content from a URL and extract text."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        return f"Error fetching content: {str(e)}"

def simple_preprocess(text):
    """Simple preprocessing without NLTK."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split into words
    words = text.split()
    # Simple stopwords list
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'from'}
    # Remove stopwords
    words = [word for word in words if word not in stopwords]
    return words

def extract_sentences(text):
    """Extract sentences from text using regex."""
    # Simple sentence splitting by punctuation followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def improved_search(content_dict, query):
    """
    Improved search function that doesn't rely on NLTK.
    """
    # Preprocess query
    query_words = simple_preprocess(query)
    query_keywords = set(query_words)
    
    results = []
    
    for url, content in content_dict.items():
        # Split content into sentences and paragraphs
        sentences = extract_sentences(content)
        paragraphs = [p for p in re.split(r'\n\s*\n', content) if p.strip()]
        
        # Score each sentence based on keyword matches
        sentence_scores = []
        for sentence in sentences:
            sentence_words = simple_preprocess(sentence)
            # Count matches between query keywords and sentence
            matches = sum(1 for word in sentence_words if word in query_keywords)
            if matches > 0:
                score = matches / max(len(query_keywords), 1)  # Normalize score
                sentence_scores.append((sentence, score))
        
        # Sort sentences by score and take top results
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:5]
        
        # Find paragraphs containing top sentences for context
        for sentence, score in top_sentences:
            if score > 0.2:  # Threshold for relevance
                # Find the paragraph containing this sentence
                for paragraph in paragraphs:
                    if sentence.lower() in paragraph.lower():
                        results.append({
                            "url": url,
                            "text": paragraph.strip(),
                            "score": score,
                            "highlight": sentence
                        })
                        break
    
    # Sort results by score
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    return results

# App title
st.set_page_config(page_title="Web Content Q&A Tool", layout="wide")
st.title("Web Content Q&A Tool")

# Initialize session state
if 'content_dict' not in st.session_state:
    st.session_state.content_dict = {}
if 'urls' not in st.session_state:
    st.session_state.urls = []

# Create tabs
tab1, tab2 = st.tabs(["URL Input", "Ask Questions"])

# URL Input Tab
with tab1:
    st.header("Input URLs")
    
    # URL input field
    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input("Enter a URL:", key="url_input")
    with col2:
        add_button = st.button("Add URL")
    
    if add_button and url_input:
        if is_valid_url(url_input):
            if url_input not in st.session_state.urls:
                st.session_state.urls.append(url_input)
                st.success(f"Added: {url_input}")
            else:
                st.warning("URL already added.")
        else:
            st.error("Please enter a valid URL.")
    
    # Display added URLs
    if st.session_state.urls:
        st.subheader("Added URLs:")
        for i, url in enumerate(st.session_state.urls):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{i+1}. {url}")
            with col2:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.urls.pop(i)
                    if url in st.session_state.content_dict:
                        del st.session_state.content_dict[url]
                    st.experimental_rerun()
    
    # Fetch content button
    if st.session_state.urls:
        if st.button("Fetch Content from URLs"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, url in enumerate(st.session_state.urls):
                status_text.text(f"Fetching content from {url}...")
                content = fetch_url_content(url)
                st.session_state.content_dict[url] = content
                progress_bar.progress((i + 1) / len(st.session_state.urls))
            
            status_text.text("Content fetched successfully!")
            st.success(f"Fetched content from {len(st.session_state.urls)} URLs.")

# Ask Questions Tab
with tab2:
    st.header("Ask Questions")
    
    if not st.session_state.content_dict:
        st.info("Please add URLs and fetch content first in the URL Input tab.")
    else:
        st.write(f"You can now ask questions about the content from {len(st.session_state.content_dict)} URLs.")
        
        question = st.text_input("Enter your question:", key="question_input")
        search_button = st.button("Search")
        
        if search_button and question:
            with st.spinner("Searching..."):
                st.subheader("Results:")
                results = improved_search(st.session_state.content_dict, question)
                
                if results:
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} from {result['url']} (Relevance: {result['score']:.2f})"):
                            # Display full paragraph
                            st.markdown(result['text'])
                            
                            # Highlight the most relevant sentence
                            st.markdown("**Most relevant sentence:**")
                            st.info(result['highlight'])
                            
                            st.caption(f"Source: {result['url']}")
                            
                    st.success(f"Found {len(results)} relevant results.")
                else:
                    st.warning("No relevant information found in the content.")
                    st.info("Try rephrasing your question or adding more URLs with relevant content.")
        
        # Option to view raw content
        if st.checkbox("Show raw content"):
            for url, content in st.session_state.content_dict.items():
                with st.expander(f"Content from {url}"):
                    # Show a preview of the content with a character limit
                    preview_length = 5000
                    display_text = content[:preview_length] + ("..." if len(content) > preview_length else "")
                    st.text_area("Raw text:", value=display_text, height=300)
                    st.text(f"Total characters: {len(content)}")

# Add sample URLs for quick testing
if st.sidebar.checkbox("Show demo options"):
    st.sidebar.subheader("Demo Options")
    if st.sidebar.button("Add Sample URLs"):
        sample_urls = [
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Artificial_intelligence"
        ]
        for url in sample_urls:
            if url not in st.session_state.urls:
                st.session_state.urls.append(url)
        st.sidebar.success("Sample URLs added")
        st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("Web Content Q&A Tool - Search across multiple web pages")