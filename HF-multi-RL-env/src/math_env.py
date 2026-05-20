r"""
MATH ENVIRONMENT (1): Very simple single-turn (model gets ONE shot. No tools, no state, no loop) math reasoning environment

for info an env contains:
1. Tasks       - what problems to solve
2. Prompt      - how to present a task to the model
3. Harness     - how the model generates (shared/llm.py)
4. Reward      - how we score the result
5. Termination - when the episode ends (single-turn = after 1 reply)

run: uv run tally-RL-env/HF-multi-RL-env/src/math_env.py

output: 
    Task Q-> Question:str
    gold = value:num | predicted = value:num | reward = value:num
"""

import re
from llm import chat

# COMPONENT 1: TASKS / DATASET  
# A.T. Paper, Write few tasks; Each task = an input (question) + the ground-truth answer (for scoring).

TASKS = [
    {"id": "Q1", "question": "Natalia sold 48 clips in April and half as many in May. How many clips did she sell in total?", "answer": 72},
    {"id": "Q2", "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?", "answer": 3},
    {"id": "Q3", "question": "Weng earns $12 per hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn in dollars?", "answer": 10},
    {"id": "Q4", "question": "Betty has half the money she needs for a $100 wallet. Her parents give her $15 and her grandparents twice as much as her parents. How much more does she need?", "answer": 5},
    {"id": "Q5", "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write in a year (52 weeks)?", "answer": 624},
]

# COMPONENT 2: PROMPT TEMPLATE 
# Instruct the model to put its final answer in \boxed{...} 

def build_prompt(task):
    system = (
        "You are a careful math solver. Think step by step. "
        "Put ONLY your final numeric answer inside \\boxed{}, like \\boxed{42}."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": task["question"]},
    ]

# COMPONENT 4: REWARD / RUBRIC  
# A.T. paper: a deterministic check, 1.0 if the parsed answer matches gold, else 0.0 

def extract_answer(text):
    """Pull the final integer out of the model's reply."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    candidate = boxed[-1] if boxed else text
    nums = re.findall(r"-?\d+", candidate.replace(",", ""))
    return int(nums[-1]) if nums else None


# Coneptual RLVR 
def reward_fn(completion, gold):
    pred = extract_answer(completion)
    return 1.0 if pred is not None and pred == gold else 0.0


# COMPONENT 5:  THE ROLLOUT (ONE trajectory)
# TERMINATION condition - done=True after one reply.
def rollout(task):
    messages = build_prompt(task)          # component 2
    completion = chat(messages)            # component 3 (harness)
    reward = reward_fn(completion, task["answer"])  # component 4
    return {
        "task_id": task["id"],
        "question": task["question"],
        "gold": task["answer"],
        "completion": completion,
        "predicted": extract_answer(completion),
        "reward": reward,
        "done": True,                      # component 5: single-turn
    }

if __name__ == "__main__":
    results = []
    for task in TASKS:
        print(f"\n{'-'*10}\nTASK {task['id']}-> {task['question']}")
        traj = rollout(task)
        print(f"  gold = {traj['gold']}  |  predicted = {traj['predicted']}  |  reward = {traj['reward']}")
        results.append(traj)

    total = sum(r["reward"] for r in results)
    print(f"\n{'='*10}")
    print(f"SUMMARY: {total:.0f} / {len(results)} solved!  (mean reward {total/len(results):.2f})")