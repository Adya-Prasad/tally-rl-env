"""
A RL environment for memory-management agents,
inspired by Memory-R1 (arxiv 2508.19828). An agent maintains a memory bank
across a sequence of dialogue turns and is rewarded based on whether a
downstream question can be answered correctly from the final memory state.

DESIGN INFLUENCES
─────────────────
- Memory-R1: Action space {ADD, UPDATE, DELETE, NOOP}, outcome-driven reward
  computed by checking if the final memory bank supports correct QA.
- NVIDIA NeMo Gym patterns: Backend-as-separate-class (state holder),
  semantic-action dispatch in step(), state-matching verification.
- Gymnasium API: Standard reset/step contract, observation_space and
  action_space declarations.

WHAT THIS ENV COMPOSES
─────────────────────
1. Multi-turn stateful env — state evolves across many steps
2. Semantic action space with structured arguments (matches LLM-agent reality)
3. Outcome-driven sparse reward (only at terminal state)
4. Backend pattern — state lives in MemoryStore, env wraps it thinly
5. Task generation — env produces episodes with varying difficulty
6. Smoke testing — a hand-coded expert verifies the env is solvable
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# DATA MODEL

@dataclass
class MemoryEntry:
    """A single fact stored in the memory bank."""
    entry_id: int
    content: str  # e.g., "User adopted a dog named Buddy"


@dataclass
class DialogueTurn:
    """One turn of dialogue the agent must process."""
    speaker: str   # "user" or "system"
    utterance: str # what was said
    extracted_fact: Optional[str] = None  # pre-extracted; in real Memory-R1
                                          # this is done by an extraction LLM


@dataclass
class Episode:
    """One complete training episode."""
    turns: list[DialogueTurn]
    final_question: str
    correct_answer: str
    # Ground-truth final memory state — what an optimal agent would maintain.
    # Used only for reward verification, never shown to the agent.
    target_memory_facts: list[str] = field(default_factory=list)


# BACKEND — holds all state, separate from the env wrapper (NeMo-Gym "Resources Server" pattern)

class MemoryBackend(ABC):
    """Abstract memory store; implementations can vary (in-memory, vector DB)."""

    @abstractmethod
    def add(self, content: str) -> int: ...

    @abstractmethod
    def update(self, entry_id: int, new_content: str) -> bool: ...

    @abstractmethod
    def delete(self, entry_id: int) -> bool: ...

    @abstractmethod
    def get_all(self) -> list[MemoryEntry]: ...

    @abstractmethod
    def reset(self) -> None: ...


class InMemoryBackend(MemoryBackend):
    """Simple Python-list backed memory store. Future: VectorDBBackend etc."""

    def __init__(self):
        self._entries: list[MemoryEntry] = []
        self._next_id: int = 0

    def add(self, content: str) -> int:
        entry = MemoryEntry(entry_id=self._next_id, content=content)
        self._entries.append(entry)
        self._next_id += 1
        return entry.entry_id

    def update(self, entry_id: int, new_content: str) -> bool:
        for entry in self._entries:
            if entry.entry_id == entry_id:
                entry.content = new_content
                return True
        return False

    def delete(self, entry_id: int) -> bool:
        for i, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                self._entries.pop(i)
                return True
        return False

    def get_all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def reset(self) -> None:
        self._entries = []
        self._next_id = 0

# ACTION SPACE — Memory-R1's four action types
class ActionType(IntEnum):
    ADD = 0
    UPDATE = 1
    DELETE = 2
    NOOP = 3

# THE ENV — thin gym.Env wrapper around the backend
class MemoryOpsEnv(gym.Env):
    """RL env where an agent manages a memory bank across dialogue turns.

    Each step, the agent sees:
      - The current dialogue turn (incl. extracted fact)
      - Its current memory bank state
    It then issues a memory operation. After all turns are processed,
    the env scores the final memory state against a target.
    """

    metadata = {"render_modes": ["text"]}

    def __init__(self, episode: Episode, backend: Optional[MemoryBackend] = None):
        super().__init__()
        self.episode = episode
        self.backend = backend or InMemoryBackend()

        # Observation: current turn text + extracted fact + memory snapshot.
        # In a real LLM-agent training setup, this would be tokenized to text.
        self.observation_space = spaces.Dict({
            "turn_index": spaces.Discrete(len(episode.turns) + 1),
            "current_utterance": spaces.Text(max_length=512),
            "extracted_fact": spaces.Text(max_length=256),
            "memory_snapshot": spaces.Text(max_length=2048),
        })

        # Action: (action_type, target_entry_id, new_content).
        # For ADD: only new_content matters.
        # For UPDATE: both target_entry_id and new_content.
        # For DELETE: only target_entry_id.
        # For NOOP: nothing matters.
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(len(ActionType)),
            "target_entry_id": spaces.Discrete(1000),  # cap on memory size
            "new_content": spaces.Text(max_length=256),
        })

        self._turn_index: int = 0

    # Standard gym.Env API 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.backend.reset()
        self._turn_index = 0
        return self._get_obs(), self._get_info()

    def step(self, action: dict):
        # Apply the action against the backend
        self._apply_action(action)
        self._turn_index += 1

        # Are we done with all turns?
        done = self._turn_index >= len(self.episode.turns)

        # Compute reward — outcome-driven, only at terminal step
        if done:
            reward = self._compute_terminal_reward()
            terminated = True
        else:
            reward = 0.0
            terminated = False

        # Truncation never triggers here — episode is bounded by turns.
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # Action dispatch

    def _apply_action(self, action: dict) -> None:
        """Dispatch the semantic action to the appropriate backend method.

        This is the same pattern your Tally env will use — dispatch in step()
        to backend methods based on action_type.
        """
        action_type = ActionType(action["action_type"])
        target_id = int(action["target_entry_id"])
        new_content = action["new_content"]

        if action_type == ActionType.ADD:
            if new_content:  # ignore empty content
                self.backend.add(new_content)
        elif action_type == ActionType.UPDATE:
            self.backend.update(target_id, new_content)
        elif action_type == ActionType.DELETE:
            self.backend.delete(target_id)
        elif action_type == ActionType.NOOP:
            pass
        # Note: invalid actions silently no-op. A stricter env could return
        # a small negative reward here to discourage invalid actions.

    # Reward: state-matching against target memory facts

    def _compute_terminal_reward(self) -> float:
        """Outcome-driven reward.

        Score = fraction of target facts present in final memory,
        with a binary success bonus if ALL target facts are present
        AND there are no spurious facts.

        This implements the Memory-R1 insight: reward the AGENT's
        memory state, not its individual operations.
        """
        memory_contents = {e.content for e in self.backend.get_all()}
        target_facts = set(self.episode.target_memory_facts)

        if not target_facts:
            return 0.0

        # Coverage: fraction of target facts present
        covered = target_facts & memory_contents
        coverage_score = len(covered) / len(target_facts)

        # Cleanliness: penalize spurious facts (those not in target)
        spurious = memory_contents - target_facts
        cleanliness_penalty = len(spurious) * 0.1

        # Binary success bonus: exact match (Memory-R1 style)
        perfect_match = memory_contents == target_facts
        success_bonus = 0.5 if perfect_match else 0.0

        reward = coverage_score - cleanliness_penalty + success_bonus
        return float(np.clip(reward, -1.0, 1.5))

    # Observation construction

    def _get_obs(self) -> dict:
        if self._turn_index < len(self.episode.turns):
            turn = self.episode.turns[self._turn_index]
            utterance = turn.utterance
            extracted = turn.extracted_fact or ""
        else:
            utterance = "(episode complete)"
            extracted = ""

        memory_lines = [
            f"  [{e.entry_id}] {e.content}" for e in self.backend.get_all()
        ]
        memory_snapshot = "\n".join(memory_lines) if memory_lines else "  (empty)"

        return {
            "turn_index": self._turn_index,
            "current_utterance": utterance,
            "extracted_fact": extracted,
            "memory_snapshot": memory_snapshot,
        }

    def _get_info(self) -> dict:
        return {
            "turn_index": self._turn_index,
            "n_memory_entries": len(self.backend.get_all()),
            "episode_length": len(self.episode.turns),
        }

# A SAMPLE EPISODE — the Memory-R1 "Buddy/Scout" case study

def make_buddy_scout_episode() -> Episode:
    """The canonical Memory-R1 example: handling 'I adopted another dog'.

    Naive memory managers DELETE 'Buddy' then ADD 'Scout' (losing Buddy).
    A correct manager UPDATEs the existing entry to include both dogs.
    """
    return Episode(
        turns=[
            DialogueTurn(
                speaker="user",
                utterance="I adopted a dog named Buddy last week!",
                extracted_fact="User adopted a dog named Buddy",
            ),
            DialogueTurn(
                speaker="user",
                utterance="Just adopted another dog. Naming him Scout!",
                extracted_fact="User adopted a second dog named Scout",
            ),
        ],
        final_question="How many dogs does the user have, and what are their names?",
        correct_answer="Two dogs: Buddy and Scout",
        target_memory_facts=[
            "User adopted two dogs: Buddy and Scout",
        ],
    )

# REWARD PROFILING — multiple "agents" demonstrate env calibration

def run_random_agent(env: MemoryOpsEnv) -> float:
    """Random agent — baseline. Should usually score poorly."""
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = {
            "action_type": int(env.np_random.integers(0, len(ActionType))),
            "target_entry_id": int(env.np_random.integers(0, 10)),
            "new_content": "some random fact",
        }
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return total_reward


def run_noop_agent(env: MemoryOpsEnv) -> float:
    """NOOP agent — never modifies memory. Should score zero (no coverage)."""
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = {
            "action_type": int(ActionType.NOOP),
            "target_entry_id": 0,
            "new_content": "",
        }
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return total_reward


def run_expert_agent(env: MemoryOpsEnv) -> float:
    """Hand-coded expert for the Buddy/Scout episode.

    Strategy:
      Turn 1: ADD a fact about Buddy
      Turn 2: UPDATE the existing entry to mention both dogs
              (rather than ADD a new one or DELETE+ADD)
    """
    obs, info = env.reset()
    total_reward = 0.0

    # Turn 1: ADD initial fact (consolidated form so target matches at end)
    action = {
        "action_type": int(ActionType.ADD),
        "target_entry_id": 0,
        "new_content": "User adopted two dogs: Buddy and Scout",  # we'll update next turn
    }
    # Hmm — at turn 1 we don't yet know about Scout. The realistic expert
    # ADDs "User adopted a dog named Buddy" first, then UPDATEs at turn 2.
    action["new_content"] = "User adopted a dog named Buddy"
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    # Turn 2: UPDATE the existing entry rather than ADD a separate one.
    # entry_id=0 because it was the first thing ADDed.
    action = {
        "action_type": int(ActionType.UPDATE),
        "target_entry_id": 0,
        "new_content": "User adopted two dogs: Buddy and Scout",
    }
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    return total_reward

# Smoke tests / demo

def main():
    print("=" * 70)
    print("MemoryOps RL Environment — Reward Profiling")
    print("=" * 70)
    print("\nEpisode: Memory-R1 'Buddy/Scout' case study")
    print("  Turn 1: User adopts Buddy")
    print("  Turn 2: User adopts Scout (a SECOND dog, not a replacement)")
    print("  Target: memory should reflect TWO dogs, Buddy AND Scout\n")

    episode = make_buddy_scout_episode()

    # Run each "agent" 5 times to see the spread
    n_trials = 5

    print("─── Random agent (baseline) ───")
    random_scores = []
    for trial in range(n_trials):
        env = MemoryOpsEnv(episode=episode)
        env.reset(seed=trial)
        score = run_random_agent(env)
        random_scores.append(score)
        print(f"  Trial {trial + 1}: reward = {score:+.3f}")
    print(f"  Mean: {np.mean(random_scores):+.3f}  Std: {np.std(random_scores):.3f}")

    print("\n─── NOOP agent (does nothing) ───")
    env = MemoryOpsEnv(episode=episode)
    score = run_noop_agent(env)
    print(f"  reward = {score:+.3f}  (expected: 0.0, since memory stays empty)")

    print("\n─── Expert agent (hand-coded optimal) ───")
    env = MemoryOpsEnv(episode=episode)
    score = run_expert_agent(env)
    print(f"  reward = {score:+.3f}  (expected: ~1.5, perfect match + bonus)")
    print(f"  Final memory state:")
    for entry in env.backend.get_all():
        print(f"    [{entry.entry_id}] {entry.content}")

    print("\n─── Reward profiling verdict ───")
    expert_score = score
    random_mean = float(np.mean(random_scores))
    if expert_score > random_mean + 0.3:
        print(f"  ✓ Expert ({expert_score:+.3f}) clearly beats random ({random_mean:+.3f}).")
        print(f"  ✓ Reward signal is well-calibrated.")
    else:
        print(f"  ✗ Expert ({expert_score:+.3f}) doesn't clearly beat random ({random_mean:+.3f}).")
        print(f"  ✗ Reward signal needs recalibration.")


if __name__ == "__main__":
    main()