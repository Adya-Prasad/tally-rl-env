r"""
MULTI-TURN WORDE ENVIRONMENT (2):

This is Env A's skeleton with THREE slots filled in that were empty
before:
  - TOOL          : guess(word) -> feedback        (the model's action)
  - OBSERVATION   : the feedback string the model reads back
  - STATE         : the secret word + guess history, persisted across turns

Plus the rollout becomes a LOOP instead of a single call.
Everything else (tasks, prompt, harness, reward) you already met in Env A.
"""
from llm import chat


# THE GAME MECHANICS  ("Execution backend" — here, pure Python, no model)
# Test this in isolation FIRST. If the scoring is wrong, no model or
# reward function can save you. This is the paper's "validate the
# mechanics before the rollout" rule.

def score_guess(secret, guess):
    """
    Standard Wordle feedback, per letter:
      G = right letter, right position (green)
      Y = right letter, wrong position (yellow)
      B = letter not in the word (black/gray)
    Returns a 5-char string like 'GBYBB'.
    """
    secret, guess = secret.upper(), guess.upper()
    result = ["B"] * 5
    secret_chars = list(secret)

    # First pass: greens (exact matches), and consume those letters
    for i in range(5):
        if guess[i] == secret[i]:
            result[i] = "G"
            secret_chars[i] = None  # used up, can't re-match as yellow

    # Second pass: yellows (right letter, wrong spot) among what's left
    for i in range(5):
        if result[i] == "G":
            continue
        if guess[i] in secret_chars:
            result[i] = "Y"
            secret_chars[secret_chars.index(guess[i])] = None  # consume

    return "".join(result)


# Quick mechanics self-test — run this file directly to see it.
# (We test the BACKEND before wiring in a model. Always.)
def _self_test():
    assert score_guess("CRANE", "CRANE") == "GGGGG", "exact match failed"
    assert score_guess("CRANE", "NACRE") == "YYYYG", "anagram-ish failed"
    assert score_guess("CRANE", "ZZZZZ") == "BBBBB", "no-match failed"
    assert score_guess("LEVEL", "EAGLE") == "YBBYY", "double-letter failed"
    print("[ok] score_guess mechanics pass")


# %% -------------------------------------------------------------------
# COMPONENT 1: TASKS  — each task is just a secret word to guess.
# (Compare to Env A where a task was question+answer. Same idea,
#  different shape. The paper's translation table in action.)
# ----------------------------------------------------------------------
TASKS = [
    {"id": "w1", "secret": "CRANE"},
    {"id": "w2", "secret": "MONEY"},
    {"id": "w3", "secret": "PLANT"},
    {"id": "w4", "secret": "BRAVE"},
    {"id": "w5", "secret": "LIGHT"},
]

MAX_TURNS = 6  # COMPONENT 5: termination — Wordle gives 6 guesses.


# %% -------------------------------------------------------------------
# COMPONENT 2: PROMPT  — how we present the game + history to the model.
# Note this prompt INCLUDES the running history. That history IS the
# "observation" being fed back each turn. The model is stateless; WE
# carry the state and re-show it every turn.
def summarize_constraints(history):
    """Turn raw feedback into explicit constraints the model can reason over.
    This is OBSERVATION ENGINEERING: same information, made legible."""
    greens = ["_"] * 5            # confirmed position
    musts = set()                 # letters that must appear (yellow somewhere)
    nopes = set()                 # letters confirmed absent (black everywhere)
    for guess, fb in history:
        for i, (ltr, mark) in enumerate(zip(guess, fb)):
            if mark == "G":
                greens[i] = ltr
                musts.discard(ltr)
            elif mark == "Y":
                musts.add(ltr)
    # a letter is "absent" only if it never showed G or Y anywhere
    seen_good = set(greens) | musts
    for guess, fb in history:
        for ltr, mark in zip(guess, fb):
            if mark == "B" and ltr not in seen_good:
                nopes.add(ltr)
    return greens, musts, nopes


def build_messages(history):
    system = (
        "You are an expert Wordle solver. Guess the secret 5-letter English word.\n"
        "Feedback per letter: G=correct spot, Y=in word wrong spot, B=not in word.\n"
        "STRATEGY: every new guess MUST satisfy all known constraints, MUST be a "
        "real English word, and MUST be different from every previous guess.\n"
        "Reply with ONLY: GUESS: XXXXX"
    )
    if history:
        greens, musts, nopes = summarize_constraints(history)
        tried = ", ".join(g for g, _ in history)
        lines = [f"Guess {i+1}: {g} -> {fb}" for i, (g, fb) in enumerate(history)]
        user = (
            "Game so far:\n" + "\n".join(lines) + "\n\n"
            f"Confirmed positions: {' '.join(greens)}\n"
            f"Letters that ARE in the word (wrong spot): {sorted(musts) or 'none yet'}\n"
            f"Letters NOT in the word: {sorted(nopes) or 'none yet'}\n"
            f"Already tried (do NOT repeat): {tried}\n\n"
            "Give a NEW valid guess satisfying all constraints."
        )
    else:
        user = "No guesses yet. Make a strong opening guess."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def parse_guess(reply):
    """Pull a 5-letter guess out of the model's reply. Returns UPPER or None."""
    import re
    m = re.search(r"GUESS:\s*([A-Za-z]{5})", reply)
    if m:
        return m.group(1).upper()
    # fallback: first standalone 5-letter token
    m = re.search(r"\b([A-Za-z]{5})\b", reply)
    return m.group(1).upper() if m else None


# %% -------------------------------------------------------------------
# COMPONENT 4: REWARD — graded, not just 0/1.
# Solving fast is better than solving slow. This is a SHAPED reward:
# it gives the (hypothetical) trainer a richer signal than pass/fail.
# The paper's "dense vs sparse" axis — this is on the denser side.
def compute_reward(solved, turns_used, history):
    """Shaped/dense reward. Rewards GOOD PLAY, not just winning, so the
    signal isn't all-or-nothing (the paper's dense-vs-sparse axis)."""
    if solved:
        return round(1.0 - 0.1 * (turns_used - 1), 2)   # solving still best
    # partial credit: progress + valid exploration, even on a loss
    guesses = [g for g, _ in history]
    unique_ratio = len(set(guesses)) / len(guesses) if guesses else 0  # penalize repeats
    best_greens = max((fb.count("G") for _, fb in history), default=0) / 5
    return round(0.3 * unique_ratio + 0.3 * best_greens, 2)

# %% -------------------------------------------------------------------
# THE ROLLOUT — now a LOOP. This is the heart of multi-turn.
# Compare line-by-line to Env A's rollout(): same five components,
# but wrapped in a while-loop with STATE (history) carried across turns.
# ----------------------------------------------------------------------
def rollout(task, verbose=True):
    secret = task["secret"]
    history = []            # COMPONENT 3 (state): persists across turns
    solved = False
    trajectory = []         # full record of the episode, for reading later

    for turn in range(1, MAX_TURNS + 1):
        # 1. model reads observation (history), produces an action
        messages = build_messages(history)
        reply = chat(messages, max_tokens=50)
        guess = parse_guess(reply)

        # guard against a malformed guess (the paper's "observation the
        # model can't produce cleanly" pitfall — handle it, don't crash)
        if guess is None or len(guess) != 5:
            if verbose:
                print(f"  turn {turn}: unparseable reply {reply!r} -> skipping turn")
            trajectory.append({"turn": turn, "raw": reply, "guess": None, "feedback": None})
            continue

        # 2. the TOOL runs: score the guess against the secret
        feedback = score_guess(secret, guess)

        # 3. update STATE; the feedback is the OBSERVATION fed back next turn
        history.append((guess, feedback))
        trajectory.append({"turn": turn, "guess": guess, "feedback": feedback})

        if verbose:
            print(f"  turn {turn}: {guess} -> {feedback}")

        # 4. TERMINATION check
        if feedback == "GGGGG":
            solved = True
            break

    reward = compute_reward(solved, len(history), history)
    return {
        "task_id": task["id"],
        "secret": secret,
        "solved": solved,
        "turns_used": len(history),
        "reward": reward,
        "trajectory": trajectory,   # the full multi-turn record
    }

if __name__ == "__main__":
    _self_test()   # validate mechanics BEFORE any model call

    results = []
    for task in TASKS:
        print(f"\n{'='*60}\nTASK {task['id']}: secret = {task['secret']}")
        res = rollout(task)
        status = "SOLVED" if res["solved"] else "failed"
        print(f"  -> {status} in {res['turns_used']} turns | reward = {res['reward']}")
        results.append(res)

    solved_count = sum(r["solved"] for r in results)
    mean_reward = sum(r["reward"] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {solved_count}/{len(results)} solved | mean reward {mean_reward:.2f}")


# FINDINGS

# This is exactly the paper's pitfall list, two entries at once: "observation format the model can't parse / act on well" and the deeper one — your reward gives no signal for the behavior you actually care about. You reward solving. You give zero signal for "made a valid new guess that respects the feedback." So from the model's point of view, repeating BASED and guessing a fresh word are worth the same: nothing, until it happens to win. That's a sparse-reward dead zone — the thing the paper warns kills training signal.
# This is the lesson. You did not find a bad model. You found an under-specified environment. And you found it by reading trajectories, exactly as the paper insists — "the biggest mistakes in RL env design are caught by reading 5 trajectories, not by 1000 training steps." You just lived that sentence.
# Before (raw observation, sparse reward): 0/5 solved, mean reward 0.00, model stuck repeating BASED four times in a row. After (sharpened observation, shaped reward): 4/5 solved, mean reward 0.69, and not a single repeated guess anywhere in the output. Same model, same secret words, same loop. The only thing that changed was the environment — the observation got more legible and the reward got denser.