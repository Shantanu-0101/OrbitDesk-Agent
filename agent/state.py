"""
agent/state.py
--------------
Shared typed state that flows through every node in the LangGraph graph.
All nodes read from and write to this state object.
"""

from typing import TypedDict, Optional, List


class SourcePassage(TypedDict):
    """A single retrieved source passage."""
    source_id: str      # e.g. "KB-003" or "CASE-1041"
    passage: str        # Relevant excerpt from the document


class AgentState(TypedDict):
    """
    The central state object passed between all LangGraph nodes.

    Flow:
      question → triage → retrieval → generation → verification → (revision?) → output
    """

    # Input 
    question: str       # The raw user question

    # Triage Node Output 
    classification: Optional[str]         # answerable | requires_clarification |
                                          # requires_escalation | out_of_scope | safe_failure
    triage_reason: Optional[str]          # Why this classification was chosen

    # Retrieval Node Output 
    retrieved_sources: Optional[List[SourcePassage]]   # Top-K retrieved passages
    retrieval_scores: Optional[List[float]]            # Cosine similarity scores

    # Generation Node Output 
    generated_answer: Optional[str]        # Raw answer from the local LLM
    clarification_question: Optional[str]  # Set if classification == requires_clarification

    # Verification Node Output 
    verification_passed: Optional[bool]    # Did the answer pass all checks?
    verification_issues: Optional[List[str]]  # List of issues found (if any)
    revision_count: int                    # How many times we've tried to revise (max 1)

    # Final Output 
    final_answer: Optional[str]            # The polished final answer (human-readable)
    confidence: Optional[float]            # 0.0 – 1.0
    requires_human: Optional[bool]         # Should a human agent take over?
    warnings: Optional[List[str]]          # Any warnings to surface
    execution_log: List[str]              # Which nodes ran (for traceability)
