import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from graph_builder import build_graph
from models import BlogState
from utils import get_last_critique_structured, parse_critique_entry
from dotenv import load_dotenv
from langchain_tavily import TavilySearch


load_dotenv()

import os
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS"))

try:
    if TAVILY_API_KEY:
        tavily_search_tool = TavilySearch(max_results = 5)
        st.session_state.tavily_search = True
    else:
        st.session_state.tavily_search = False
        st.warning("Tavily API Key not found. Blog search will be skipped.")
except Exception as e:
    st.error(f"Failed to initialize Tavily Search Tool: {e}")
    st.session_state.tavily_search = False
    
def search_for_blog(user_prompt, search_tool: TavilySearch):
    search_query = f"{user_prompt} blog post"
    
    results = search_tool.invoke({"query": search_query})
    # print(results)
    # print(results['results'])
    
    if not results or len(results) == 0:
        return "> *No highly relevant blogs found for this topic."
    
    formatted_output = "### 📚 Found Existing Blogs:\n"
    
    try:
        for i, res in enumerate(results["results"]):
            title = res.get('title', 'No Title')
            url = res.get('url', '#')
            snippet = res.get('content', 'No description available.')
            
            formatted_output += f"{i+1}. **[{title}]({url})**\n"
            formatted_output += f"  - *Snippet:* {snippet[:120]}...\n\n"
            
        return formatted_output
    
    except Exception as e:
        return f"> *Error during contextual search: {e}*"

st.set_page_config(page_title="Self-Correcting Blog Agent")
st.title("Self-Correcting Blog Agent")

def init_session_state():
    if "blog_graph" not in st.session_state:
        st.session_state.blog_graph = build_graph()
        
        st.session_state.run_status = "IDLE"
        st.session_state.current_state = {}
        st.session_state.final_content = ""
        st.session_state.critique_log = []
        
init_session_state()

def start_blog_generation(user_prompt):
    """Callback function"""
    st.subheader("Contextual Search Results")
    if st.session_state.tavily_search:
        with st.spinner("Searching the web for existing blogs..."):
            search_output = search_for_blog(user_prompt=user_prompt, search_tool=tavily_search_tool)
            st.markdown(search_output)
    
    st.divider()
    
    st.session_state.run_status = "RUNNING"
    st.session_state.final_content = ""
    st.session_state.critique_log = []
    
    callback_container = st.empty()
    
    st_callback = StreamlitCallbackHandler(callback_container, expand_new_thoughts=True, collapse_completed_thoughts=False)
    
    initial_state: BlogState = {
        "user_prompt": user_prompt,
        "blog_content": "",
        "critique_history": [],
        "revision_count": 1,
        "critique_decision": "REVISE",
        "quality_score": 0.0
        
    }
    
    try:
        final_state = st.session_state.blog_graph.invoke(
            initial_state,
            config = {"callbacks": [st_callback]}
        )
        
        st.session_state.current_state = final_state
        st.session_state.final_content = final_state.get("blog_content", "")
        st.session_state.critique_log = final_state.get("critique_history", [])
        
    except Exception as e:
        st.error(f"An error occurred during graph execution: {e}")
        
    finally:
        st.session_state.run_status = "FINISHED"
        
with st.form("blog_form", clear_on_submit=False):
    st.markdown("Enter the topic and specific requirements for your blog post.")
    
    user_input = st.text_area(
        "Blog Topic/Outline",
        placeholder="e.g., The future of AI in content marketing, focusing on personalization and SEO.",
        height = 150,
        key="user_prompt_input"
    )
    
    submitted = st.form_submit_button(
        "Generate and Critique",
        type="primary",
        disabled=st.session_state.run_status == "RUNNING"
    )
    
if submitted and user_input:
    start_blog_generation(user_input)


if st.session_state.run_status == "RUNNING":
    st.info("Agent is working on the draft, See the live updates below...")
    
    st.divider()
    
if st.session_state.final_content:
    final_state = st.session_state.current_state
    
    st.success("Generation Complete!")
    
    final_score = final_state.get("quality_score", 0.0)
    
    if final_state["critique_decision"] == "PASS" or final_score >= 85.0:
        st.write(f"The draft passed the quality check after **{final_state['revision_count']}** revision(s) with a final score of **{final_score:.1f}/100**.")
    elif final_state["revision_count"] >= MAX_REVISIONS:
        st.warning(f"Draft finalized after reaching the maximum limit of {MAX_REVISIONS} revisions. Final Score: **{final_score:.1f}/100**.")
        
    st.subheader("Final Blog Post")
    st.markdown(st.session_state.final_content)
    
    st.divider()
    
    with st.expander(f"Agent Reflection and Critique History ({len(st.session_state.critique_log)} Iterations)"):
        for i, critique in enumerate(st.session_state.critique_log):
            critique_data = parse_critique_entry(critique)
            
            score = critique_data.get("quality_score", "N/A")
            
            status = "PASSED" if critique_data.get("decision") == "PASS" else "REVISED"
            if i + 1 == len(st.session_state.critique_log) and status == "REVISED":
                status = "STOPPED (MAX REVISIONS)"
                
            st.markdown(f"**Iteration {i+1}: Status: {status} | Score: {score}/100**")
            st.markdown(f"**Summary** {critique_data.get('critique_summary')}")
            feedback_list = critique_data.get("specific_feedback")
            if feedback_list:
                st.markdown("**Specific Feedback**")
                for fb in feedback_list:
                    st.markdown(f" *{fb}*")
                    
            st.markdown("---")