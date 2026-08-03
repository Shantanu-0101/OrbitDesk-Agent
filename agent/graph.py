"""
agent/graph.py
--------------
Wires the 4 nodes into a LangGraph workflow.
Implements the conditional routing and retry logic.
"""

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import triage_node, retrieval_node, generation_node, verification_node


def route_triage(state: AgentState):
    """
    Decides where to go after the Triage node.
    """
    classification = state.get("classification")
    
    if classification == "answerable":
        return "retrieval"
    else:
        # If it's vague, out of scope, or needs escalation, we skip retrieval
        # and go straight to the end. The final output will handle the message.
        return END


def route_verification(state: AgentState):
    """
    Decides what to do after Verification.
    If it fails, we revise EXACTLY ONCE to prevent infinite loops.
    """
    passed = state.get("verification_passed", False)
    revisions = state.get("revision_count", 0)
    
    if passed:
        return END
        
    if not passed and revisions < 1:
        print("  [Router] Verification failed. Attempting revision (1/1)...")
        # Update revision count so we don't loop forever
        state["revision_count"] = revisions + 1
        return "generation"
        
    print("  [Router] Verification failed twice. Outputting safe failure.")
    return END


def build_graph():
    """
    Constructs and compiles the LangGraph.
    """
    # 1. Initialize the graph with our shared state
    workflow = StateGraph(AgentState)
    
    # 2. Add our 4 nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("verification", verification_node)
    
    # 3. Define the entry point
    workflow.set_entry_point("triage")
    
    # 4. Add conditional edges (the routing logic)
    workflow.add_conditional_edges(
        "triage",
        route_triage,
        {
            "retrieval": "retrieval",
            END: END
        }
    )
    
    # 5. Add standard edges
    workflow.add_edge("retrieval", "generation")
    workflow.add_edge("generation", "verification")
    
    # 6. Add conditional edge for the retry loop
    workflow.add_conditional_edges(
        "verification",
        route_verification,
        {
            "generation": "generation",
            END: END
        }
    )
    
    # Compile it into a runnable application
    app = workflow.compile()
    return app


# ── Quick test 
if __name__ == "__main__":
    app = build_graph()
    print("Graph built successfully!")
    
    # can print the graph structure if needed
    print(app.get_graph().draw_ascii())