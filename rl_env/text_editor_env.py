"""
An agent operates a small text editor: it can move a cursor, insert text,
delete text, and save. The task is to transform a starting document into
a target document. Episode ends when the document matches target (success)
or after a step budget (failure).

What this env compose:
  1. STATEFUL environment (the document is the world; agent mutates it)
  2. SEMANTIC action space — actions are (action_type, args) just like Tally
  3. OUTCOME-DRIVEN reward — env verifies "did the final state match?"
  4. The Backend pattern — the env logic stays simple; a 'Document' class
     holds the actual state. This is the same pattern your Tally env will
     use, except with TallyBackend / SimulatedTallyBackend.
  5. Task specifications — env can be parameterized to different tasks
"""

from dataclasses import dataclass
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

# The backend: the part that holds state and exposes operations.

class Document:
    """In-memory text document with a cursor.
    it holds all the actual state and exposes operations as methods.
    The env class is a thin gym wrapper around it.
    """

    def __init__(self, initial_text: str = ""):
        self.text: str = initial_text
        self.cursor: int = 0          # position in [0, len(text)]
        self.saved: bool = False      # has the doc been saved?

    def move_cursor(self, position: int) -> None:
        # Clip cursor to legal range
        self.cursor = max(0, min(len(self.text), position))
        self.saved = False

    def insert(self, snippet: str) -> None:
        self.text = self.text[:self.cursor] + snippet + self.text[self.cursor:]
        self.cursor += len(snippet)
        self.saved = False

    def delete_chars(self, n: int) -> None:
        # Delete n chars starting at cursor (clip to remaining length)
        n = max(0, min(n, len(self.text) - self.cursor))
        self.text = self.text[:self.cursor] + self.text[self.cursor + n:]
        self.saved = False

    def save(self) -> None:
        self.saved = True

    def snapshot(self) -> str:
        """Render the doc as a string for observations."""
        cursor_marker = "|"  # visual cursor
        return self.text[:self.cursor] + cursor_marker + self.text[self.cursor:]


# The Task: defines goal and verification

@dataclass
class TextEditorTask:
    description: str
    initial_text: str
    target_text: str
    requires_save: bool = True  # task only complete after save

    def is_complete(self, doc: Document) -> bool:
        text_matches = doc.text == self.target_text
        if self.requires_save:
            return text_matches and doc.saved
        return text_matches


# The Env: thin gym wrapper around Document + Task

class TextEditorEnv(gym.Env):
    """Multi-step text-editing env. Action = (action_type, args)."""

    metadata = {"render_modes": ["text"]}

    # Semantic action types - exactly the pattern Tally use
    ACTION_MOVE_CURSOR = 0
    ACTION_INSERT = 1
    ACTION_DELETE = 2
    ACTION_SAVE = 3
    N_ACTION_TYPES = 4

    def __init__(self, task: TextEditorTask, max_steps: int = 30):
        super().__init__()
        self.task = task
        self.max_steps = max_steps

        # Observation: current doc snapshot + task description + step counter
        self.observation_space = spaces.Dict({
            "doc_snapshot": spaces.Text(max_length=1024),
            "target_text": spaces.Text(max_length=1024),
            "saved": spaces.Discrete(2),  # 0 or 1
            "steps_remaining": spaces.Discrete(max_steps + 1),
        })

        # Action: a Dict combining action type with type-specific args.
        # For real LLM-agent training, an action is typically a JSON string
        # the agent generates — but the env decodes it into this structure.
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(self.N_ACTION_TYPES),
            "int_arg": spaces.Discrete(1024),  # cursor pos or delete count
            "text_arg": spaces.Text(max_length=256),  # text to insert
        })

        self.doc: Optional[Document] = None
        self.step_count: int = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.doc = Document(initial_text=self.task.initial_text)
        self.step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action: dict):
        self.step_count += 1
        action_type = action["action_type"]

        # Dispatch — this is the central pattern.
        # In Tally, this exact dispatch table will map to backend methods
        # (CreateVoucher → backend.create_voucher, AddEntry → ..., etc.).
        if action_type == self.ACTION_MOVE_CURSOR:
            self.doc.move_cursor(action["int_arg"])
        elif action_type == self.ACTION_INSERT:
            self.doc.insert(action["text_arg"])
        elif action_type == self.ACTION_DELETE:
            self.doc.delete_chars(action["int_arg"])
        elif action_type == self.ACTION_SAVE:
            self.doc.save()
        # else: invalid action_type — silently no-op. A stricter env could
        # return a small negative reward here to discourage invalid actions.

        # OUTCOME-DRIVEN REWARD — same pattern as Memory-R1.
        # We don't reward per-step closeness to target. We give +1 only
        # when the task is complete. This avoids reward hacking where the
        # agent finds ways to "be closer" without actually solving the task.
        terminated = self.task.is_complete(self.doc)
        truncated = (not terminated) and self.step_count >= self.max_steps
        reward = 1.0 if terminated else 0.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        return {
            "doc_snapshot": self.doc.snapshot(),
            "target_text": self.task.target_text,
            "saved": 1 if self.doc.saved else 0,
            "steps_remaining": self.max_steps - self.step_count,
        }

    def _get_info(self):
        return {
            "step_count": self.step_count,
            "current_text": self.doc.text,
            "task_description": self.task.description,
        }

    def render(self):
        print(f"Task: {self.task.description}")
        print(f"Current: {self.doc.snapshot()}")
        print(f"Target : {self.task.target_text}")
        print(f"Saved  : {self.doc.saved}\n")


# Demo: hand-coded "expert" policy that solves a specific task

def demo_expert_policy():
    """A hand-coded policy that solves a 'fix the typo' task.

    Task: initial 'hello wrld', target 'hello world'. The expert
    moves cursor to position 7, inserts 'o', then saves.

    Showing this demonstrates that the env is genuinely solvable and
    that the action vocabulary is rich enough to express the solution.
    """
    task = TextEditorTask(
        description="Fix the typo: insert 'o' after 'w' to make 'hello world'",
        initial_text="hello wrld",
        target_text="hello world",
        requires_save=True,
    )
    env = TextEditorEnv(task=task)
    obs, info = env.reset()
    env.render()

    # Step 1: move cursor between 'w' and 'r' (position 7)
    obs, reward, term, trunc, info = env.step(
        {"action_type": env.ACTION_MOVE_CURSOR, "int_arg": 7, "text_arg": ""}
    )
    env.render()

    # Step 2: insert 'o'
    obs, reward, term, trunc, info = env.step(
        {"action_type": env.ACTION_INSERT, "int_arg": 0, "text_arg": "o"}
    )
    env.render()

    # Step 3: save
    obs, reward, term, trunc, info = env.step(
        {"action_type": env.ACTION_SAVE, "int_arg": 0, "text_arg": ""}
    )
    env.render()

    print(f"Solved? {term} | Total steps: {info['step_count']} | Reward: {reward}")


if __name__ == "__main__":
    demo_expert_policy()