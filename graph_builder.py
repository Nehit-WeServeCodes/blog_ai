from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage, HumanMessage

import json

from models import BlogState, CritiqueDecision
from utils import get_last_critique_structured, parse_critique_entry

load_dotenv()

import os
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS"))

# MAX_REVISIONS = 25

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

structured_llm = llm.with_structured_output(CritiqueDecision)

# GENERATOR NODE
def generate_draft(state: BlogState) -> BlogState:
    print("--- GENERATOR: Generating Initial Draft ---")
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert blog writer skilled at writing clear, engaging, and informative content.",
        ),
        (
            "human",
            "Write a detailed, engaging blog post based on the following topic and keep in a {tone} tone:\n\n"
            "{user_prompt}\n\n"
            "Make sure the writing is well structured with title of the blog, introduction, body, and conclusion"
            "Keep the tone conversational yet professional.",
        ),
    ])
    
    chain = prompt_template | llm
    response = chain.invoke({"user_prompt": state['user_prompt'], "tone": "inspirational"})
    
    return {
        "blog_content": response.content,
        "revision_count": 1,
        "critique_history": [AIMessage(content="Initial draft generated.")],
        "critique_decision": "REVISE",
        "quality_score": 0.0
    }
    

# CRITICIZER NODE
def evaluate_draft(state: BlogState) -> BlogState:
    attempt = state.get("revision_count", 1)
    print(f"--- CRITICIZER: Evaluating Draft (Attempt {attempt}) ---")
    
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a professional editor who evaluates blogs based on quality, clarity and engagement. "
            "Be objective and provide constructive criticism. **You must assign a Quality Score from 0.0 to 100.0.**"
            "Use the provided schema to return your decision.",
        ),
        (
            "human",
            "Here is the current blog draft that needs critique:\n\n"
            "{blog_content}\n\n"
            "Evaluate the above blog and decide whether it should be 'REVISE' or 'ACCEPT'."
            "Also, based on the requirements and the quality, provide a **Quality Score** (0.0 to 100.0)."
            "Also summarize your critique and list 2-3 specific feedback points for improvement."
        ),
    ])
    
    chain = prompt_template | structured_llm
    
    critique_result: CritiqueDecision = chain.invoke({
        "blog_content": state["blog_content"]
    })
    
    critique_update = critique_result.model_dump()
    critique_update["revision_count"] = state["revision_count"]
    
    # print(critique_update)
    
    return {
        "critique_decision": critique_result.decision,
        "critique_history": [AIMessage(content=json.dumps(critique_update))],
        "revision_count": state["revision_count"],
        "quality_score": critique_result.quality_score,
    }
    
def update_revision(state: BlogState) -> BlogState:
    return {
        "revision_count": len(state["critique_history"])
    }
    

# ENHANCER NODE
def revise_draft(state: BlogState) -> BlogState:
    print("--- ENHANCER: Revising Draft ---")
    latest_critique_dict = get_last_critique_structured(state["critique_history"])
    # print(latest_critique_dict)
    latest_critique = latest_critique_dict.get("specific_feedback", [])
    feedback_text = "\n- " + "\n- ".join(latest_critique)
    
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert blog editor. Your task is to revise the given blog draft "
            "by applying the critique feedback while preserving the original tone, structure, and coherence.",
        ),
        (
            "human",
            "Here is the critique feedback you must address:\n"
            f"{feedback_text}\n\n"
            "And here is the current blog draft that needs revision:\n\n"
            f"{state['blog_content']}\n\n"
            "Now, produce an improved version of this blog that incorporates the feedback above. "
            "Ensure the result remains fluent and engaging.",
        ),
    ])
    
    chain = prompt_template | llm
    response = chain.invoke({})
    
    return {
        "blog_content": response.content,
        # "revision_count": state["revision_count"] + 1
    }


def should_continue(state: BlogState):
    """Determines the next step based on the Criticizer's decision and loop count."""
    
    current_score = state.get("quality_score", 0.0)
    TARGET_SCORE = 85.0
    
    if current_score >= TARGET_SCORE:
        print(f"--- ROUTER: Current Score ({current_score}) - Target Score ({TARGET_SCORE}) Met. Finalizing Draft ---")
    
    if state["critique_decision"] == "PASS":
        print("--- ROUTER: Draft PASSED. ENDING GRAPH ---")
        return END
    
    if state["revision_count"] >= MAX_REVISIONS:
        print(f"--- ROUTER: Max Revisions ({MAX_REVISIONS}) Reached. FORCING END ---")
        return "forced_end"
    
    print("--- ROUTER: Draft needs REVISION. Looping to Enhancer ---")
    return "revise_draft"


def build_graph():
    """Compiles the Graph"""
    workflow = StateGraph(BlogState)
    
    workflow.add_node("generator", action=generate_draft)
    workflow.add_node("evaluate_draft", action=evaluate_draft)
    workflow.add_node("revise_draft", action=revise_draft)
    workflow.add_node("update_revision", action=update_revision)
    
    workflow.add_edge(START, "generator")
    workflow.add_edge("generator", "evaluate_draft")
    workflow.add_edge("evaluate_draft", "update_revision")
    workflow.add_conditional_edges(
        "update_revision",
        should_continue,
        {
            "revise_draft": "revise_draft",
            "forced_end": END,
            END: END,
        }
    )
    
    workflow.add_edge("revise_draft", "evaluate_draft")
    
    return workflow.compile()