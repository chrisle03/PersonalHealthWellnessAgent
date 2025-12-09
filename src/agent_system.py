# Step 4: Integrating Components into an Agent System

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Tuple, Optional, Any
import json, math, re, textwrap, random, os, sys
import math
from collections import Counter, defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from prompting_techniques import make_prompt, parse_action

@dataclass
class Step:
    thought: str
    action: str
    observation: str

@dataclass
class AgentConfig:
    max_steps: int = 6
    allow_tools: Tuple[str, ...] = ("search",)
    verbose: bool = True

class ReActAgent:
    def __init__(self, llm: Callable[[str], str], tools: Dict[str, Dict[str, Any]], config: AgentConfig | None=None):
        self.llm = llm
        self.tools = tools
        self.config = config or AgentConfig()
        self.trajectory: List[Step] = []

    def run(self, user_query: str) -> Dict[str, Any]:
        self.trajectory.clear()

        step_idx = 0
        final_answer = None
        
        for step_idx in range(self.config.max_steps):
            if self.config.verbose:
                print(f"--- Step {step_idx + 1} ---")
            
            # 1. Format prompt
            prompt = make_prompt(user_query, self.trajectory)
            
            # 2. Get LLM response
            out = self.llm(prompt)

            # Expect two lines: Thought:..., Action:...
            t_match = re.search(r"Thought:\s*(.*)", out)
            a_match = re.search(r"Action:\s*(.*)", out)
            thought = t_match.group(1).strip() if t_match else "(no thought)"
            action_line = a_match.group(1).strip() if a_match else "finish[answer=\"(no action)\"]"
            
            # Ensure action_line has "Action: " prefix for parsing
            if not action_line.startswith("Action:"):
                action_line = "Action: " + action_line

            # 3. Parse action
            parsed = parse_action(action_line)

            if not parsed:
                observation = "Invalid action format. Stopping."
                self.trajectory.append(Step(thought, action_line, observation))
                break
            
            name, args = parsed

            # Check if this is a finish action
            if name == "finish":
                observation = "done"
                self.trajectory.append(Step(thought, action_line, observation))
                # Extract the answer from args
                final_answer = args.get("answer", "No answer provided")
                break

            if name not in self.config.allow_tools or name not in self.tools:
                observation = f"Action '{name}' not allowed or not found."
                self.trajectory.append(Step(thought, action_line, observation))
                break

            # 4. Execute the action
            try:
                obs_payload = self.tools[name]["fn"](**args)
                observation = json.dumps(obs_payload, ensure_ascii=False)
            except Exception as e:
                observation = f"Tool error: {e}"

            self.trajectory.append(Step(thought, action_line, observation))

        # If we didn't find a finish action, try to extract from trajectory
        if final_answer is None:
            for step in reversed(self.trajectory):
                if "finish[" in step.action or "finish [" in step.action:
                    # Try multiple patterns
                    patterns = [
                        r'answer="([^"]*)"',
                        r"answer='([^']*)'",
                        r'answer=([^,\]]+)'
                    ]
                    for pattern in patterns:
                        m = re.search(pattern, step.action)
                        if m:
                            final_answer = m.group(1)
                            break
                    if final_answer:
                        break
            
        print("DEBUG — trajectory:", self.trajectory)
        print("DEBUG — final_answer:", final_answer)
        
        return {
            "question": user_query,
            "final_answer": final_answer,
            "steps": [asdict(step) for step in self.trajectory]
        }