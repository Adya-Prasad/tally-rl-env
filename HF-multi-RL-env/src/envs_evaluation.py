# src/phase3_compare.py
# =====================================================================
# PHASE 3 — Compare the two environments across the paper's components,
# then see what a framework (Verifiers) would and wouldn't change.
#
# Key realization this phase delivers:
#   Both environments expose the SAME shape to a collector:
#       reset -> (loop: act -> observe -> reward -> done?) -> final reward
#   Math is that loop with exactly ONE iteration. Wordle is the loop with
#   up to six. They are not two different things. They are the same thing
#   at different trip counts. THAT is the paper's "same skeleton" thesis.
# =====================================================================

import math_env
import wordle_env


# %% -------------------------------------------------------------------
# A UNIFORM TRAJECTORY VIEW
# Both envs already return a dict from rollout(). Here we normalize them
# into one common record so a single "collector" can read either. This is
# precisely the job a framework's rollout API does: give every env the
# same interface so the trainer doesn't care which env it's pulling from.
# ----------------------------------------------------------------------
def normalize_math(res):
    return {
        "env": "math (single-turn)",
        "task_id": res["task_id"],
        "turns": 1,
        "reward": res["reward"],
        "solved": res["reward"] == 1.0,
    }


def normalize_wordle(res):
    return {
        "env": "wordle (multi-turn)",
        "task_id": res["task_id"],
        "turns": res["turns_used"],
        "reward": res["reward"],
        "solved": res["solved"],
    }


# %% -------------------------------------------------------------------
# THE COLLECTOR — one loop that drives BOTH environments.
# Notice it doesn't know or care that one is single-turn and one is
# multi-turn. It just calls rollout() and reads the reward. That
# indifference is the entire value of a standard environment interface.
# ----------------------------------------------------------------------
def collect(env_module, normalize_fn, label):
    print(f"\n{'#'*64}\n# COLLECTING FROM: {label}\n{'#'*64}")
    records = []
    for task in env_module.TASKS:
        res = env_module.rollout(task)  # quiet; we summarize
        rec = normalize_fn(res)
        records.append(rec)
        flag = "✓" if rec["solved"] else "·"
        print(f"  {flag} {rec['task_id']}: {rec['turns']} turn(s), reward {rec['reward']}")
    mean = sum(r["reward"] for r in records) / len(records)
    print(f"  mean reward: {mean:.2f}")
    return records


# %% -------------------------------------------------------------------
# RUN BOTH, THEN COMPARE
# ----------------------------------------------------------------------
if __name__ == "__main__":
    math_records = collect(math_env, normalize_math, "Environment A — math")
    wordle_records = collect(wordle_env, normalize_wordle, "Environment B — Wordle")

    print(f"\n{'='*64}\nSIDE-BY-SIDE: the same 12 components, two fillings\n{'='*64}")
    rows = [
        ("Component",        "Env A (math)",          "Env B (Wordle)"),
        ("Tasks",            "question + answer",      "secret word"),
        ("Prompt",           "static question",        "history + constraints"),
        ("Harness",          "chat() x1",              "chat() in a loop"),
        ("Tool / Action",    "none",                   "score_guess each turn"),
        ("Observation",      "none",                   "feedback + constraints"),
        ("State",            "none",                   "history across turns"),
        ("Termination",      "after 1 reply",          "GGGGG or 6 turns"),
        ("Reward",           "0/1 exact-match",        "shaped, graded"),
        ("Episode control",  "trainer (1 step)",       "trainer (the loop)"),
    ]
    w = [max(len(r[c]) for r in rows) for c in range(3)]
    for r in rows:
        print(f"  {r[0]:<{w[0]}}  |  {r[1]:<{w[1]}}  |  {r[2]:<{w[2]}}")

    print(f"\n{'='*64}")
    print("KEY INSIGHT: the collector above ran BOTH envs through the same")
    print("rollout() -> reward interface. Single-turn is the loop with one")
    print("iteration; multi-turn is the loop with many. Same skeleton.")