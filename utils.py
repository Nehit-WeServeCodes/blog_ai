from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
import json, ast

from models import BlogState, CritiqueDecision

def parse_critique_entry(entry):
    """Safely convert AIMessage or string to dict."""
    if isinstance(entry, dict):
        return entry
    elif isinstance(entry, AIMessage):
        text = entry.content
    else:
        text = str(entry)
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return {"raw": text}

def get_last_critique_structured(history):
    """Return the most recent critique entry as a dict."""
    for item in reversed(history):
        if isinstance(item, dict):
            return item
        elif isinstance(item, AIMessage):
            parsed = parse_critique_entry(item)
            if isinstance(parsed, dict):
                return parsed
    return {}
