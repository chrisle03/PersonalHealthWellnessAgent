import os
import sys
import pandas as pd
import json
from datetime import datetime

# Add the current directory to sys.path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the agent components
try:
    from src.language_model import LLM
    from src.knowledge_base import TOOLS
    from src.agent_system import ReActAgent
except ImportError:
    from language_model import LLM
    from knowledge_base import TOOLS
    from agent_system import ReActAgent

def evaluate_agent():
    #  Define Test Questions
    test_questions = [
        "How do I make a chocolate cake?",
        "What is a good workout for a beginner?",
        "How much time does it take to cook apple pie?",
        "Can you find a recipe with chicken and rice?"
    ]

    # Setup Logging
    results_dir = os.path.join("results", "logs")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"evaluation_{timestamp}.json")

    #  Initialize Agent
    print("Initializing Agent for Evaluation...")
    agent = ReActAgent(llm=LLM, tools=TOOLS)

    evaluation_results = []

    #  Run Evaluation Loop
    for i, query in enumerate(test_questions):
        print(f"\n[{i+1}/{len(test_questions)}] Processing: {query}")
        try:
            result = agent.run(query)
          
            eval_entry = {
                "question": query,
                "final_answer": result["final_answer"],
                "steps_taken": len(result["steps"]),
                "trajectory": result["steps"], 
                "status": "success"
            }
        except Exception as e:
            print(f"Error processing query: {e}")
            eval_entry = {
                "question": query,
                "error": str(e),
                "status": "failed"
            }
        
        evaluation_results.append(eval_entry)

    # Save Results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation complete. Results saved to: {output_file}")

if __name__ == "__main__":
    evaluate_agent()
