from typing import TypedDict, Annotated, Sequence, Literal, List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class BlogState(TypedDict):
    """Represents the shared state of the LangGraph workflow"""
    user_prompt: str
    blog_content: str
    critique_history: Annotated[Sequence[BaseMessage], add_messages]
    critique_decision: Literal["REVISE", "PASS"]
    revision_count: int
    quality_score: float
    
    
class CritiqueDecision(BaseModel):
    """The structured output schema the Criticizer must return."""
    decision: Literal["REVISE", "PASS"] = Field(
        ...,
        description="The binary decision: 'REVISE' if the blog needs improvement, 'PASS' if it is finalized."
    )
    
    quality_score: float = Field(
        ...,
        description="A quantitative quality score for the blog draft, ranging from 0.0 (Poor) to 100.0 (Excellent). Use this score to track iterative improvement."
    )
        
    critique_summary: str = Field(
        ...,
        description="A brief, high-level summary of the evaluation results."
    )
    
    specific_feedback: List[str] = Field(
        ...,
        description="Actionable, itemized feedback points for the Enhancer to address."
    )