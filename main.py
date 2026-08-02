"""
main.py
-------
The main entry point for the OrbitDesk Support Agent.
It loads the 5 sample questions and runs them through the LangGraph.
"""

import json
import argparse
from colorama import Fore, Style, init

from agent.graph import build_graph
from agent.retriever import get_retriever
from agent.models import get_llm
from agent.state import AgentState

# Initialize colorama for colored terminal output
init(autoreset=True)

def format_final_output(state: AgentState) -> dict:
    """
    Takes the messy LangGraph state and formats it into the exact
    JSON schema requested by the assignment.
    """
    classification = state.get("classification", "safe_failure")
    
    # Base output structure
    output = {
        "classification": classification,
        "answer": "",
        "sources": [],
        "confidence": 0.0,
        "requires_human": False,
        "reason": state.get("triage_reason", ""),
        "clarification_question": None,
        "warnings": state.get("warnings", []),
        "execution_log": state.get("execution_log", [])
    }
    
    # Route: Answerable
    if classification == "answerable":
        if state.get("verification_passed", False):
            output["answer"] = state.get("generated_answer", "")
            output["sources"] = state.get("retrieved_sources", [])
            output["confidence"] = state.get("confidence", 0.85)
        else:
            output["classification"] = "safe_failure"
            output["answer"] = "I'm sorry, I couldn't confidently verify the answer based on the available documentation."
            output["confidence"] = 0.0
            output["requires_human"] = True
            output["reason"] = "Verification failed even after revision."
            
    # Route: Clarification
    elif classification == "requires_clarification":
        output["answer"] = state.get("clarification_question", "Could you provide more details?")
        output["clarification_question"] = output["answer"]
        output["confidence"] = 0.9
        
    # Route: Escalation
    elif classification == "requires_escalation":
        output["answer"] = "I need to escalate this issue to our engineering team."
        output["requires_human"] = True
        output["confidence"] = 1.0
        
    # Route: Out of scope
    elif classification == "out_of_scope":
        output["answer"] = "I am a technical support assistant. I cannot assist with billing, legal, or account changes."
        output["confidence"] = 1.0
        
    return output


def print_colored_output(question: str, output: dict):
    """Prints the result nicely for the video demo."""
    print(f"\n{Fore.CYAN}==================================================")
    print(f"{Fore.CYAN}QUESTION: {Fore.WHITE}{question}")
    print(f"{Fore.CYAN}==================================================")
    
    # Determine color based on classification
    cls_color = Fore.GREEN
    if output["classification"] == "requires_clarification":
        cls_color = Fore.YELLOW
    elif output["classification"] in ["requires_escalation", "out_of_scope", "safe_failure"]:
        cls_color = Fore.RED
        
    print(f"{Fore.MAGENTA}Path Taken: {Fore.WHITE}{' → '.join(output['execution_log'])}")
    print(f"{Fore.MAGENTA}Classification: {cls_color}{output['classification'].upper()}")
    print(f"{Fore.MAGENTA}Reason: {Fore.WHITE}{output['reason']}")
    
    print(f"\n{Fore.GREEN}🤖 Answer:")
    print(f"{Fore.WHITE}{output['answer']}")
    
    if output["sources"]:
        print(f"\n{Fore.YELLOW}📚 Sources Cited:")
        for src in output["sources"]:
            print(f"   - {src['source_id']}")
            
    print(f"{Fore.CYAN}==================================================\n")


def process_question(app, question: str):
    """Runs a single question through the graph."""
    initial_state = {"question": question, "revision_count": 0, "execution_log": []}
    
    print(f"\n{Fore.YELLOW}Processing...{Style.RESET_ALL}")
    
    # Run the graph!
    final_state = app.invoke(initial_state)
    
    # Format to required JSON schema
    output = format_final_output(final_state)
    
    # Print readable version
    print_colored_output(question, output)
    
    return output


def main():
    parser = argparse.ArgumentParser(description="OrbitDesk Support Agent")
    parser.add_argument("--question", type=str, help="Ask a custom question")
    args = parser.parse_args()

    print(f"{Fore.GREEN}Starting OrbitDesk Local Agent...{Style.RESET_ALL}")
    
    # 1. Warm up the models (This is where the downloads happen if missing)
    get_retriever()
    get_llm()
    
    # 2. Build the LangGraph
    app = build_graph()
    
    # 3. Handle custom question
    if args.question:
        process_question(app, args.question)
        return

    # 4. Handle the 5 required test cases
    try:
        with open("../sample_questions.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            questions = data.get("questions", [])
    except FileNotFoundError:
        print(f"{Fore.RED}Could not find sample_questions.json!{Style.RESET_ALL}")
        return

    all_outputs = []
    
    for q in questions:
        q_text = q.get("question", "")
        out = process_question(app, q_text)
        all_outputs.append({
            "question_id": q.get("question_id"),
            "question": q_text,
            "response": out
        })
        
    # Save the output to a JSON file (great for the submission repo)
    with open("outputs/final_results.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2)
        
    print(f"{Fore.GREEN}All done! Results saved to outputs/final_results.json{Style.RESET_ALL}")


if __name__ == "__main__":
    main()