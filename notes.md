
**KEYPOINTS**
1. Reward model
2. RL for spatial need (Spatial RL - GUI agents)
3. RL env for Tally prime

---

## My Strategic Approach: Applied RL
_The Foundations_
1. **Markov Decision Processes (MDPs):** Formally define Tally as an MDP. What is the State ($S$)? What is the Action ($A$)? What is the Transition Probability ($P$)?
2. **Inverse Reinforcement Learning (IRL):** Since Tally has "correct" ways of doing things (accounting rules), look into how to learn a reward function by watching a human use Tally.
3. **Hierarchical RL (HRL):** Tally tasks are long. Learn how to break a big task (Audit) into sub-tasks (Navigate to Ledger -> Verify Entry -> Compare).

_The Reward Model Design_

4. Use a **Bradley-Terry model** for preference learning (ranking which agent actions look "more professional").
5. **Penalty Design:** How do you penalize "looping" (the agent clicking the same menu over and over) without killing its curiosity to explore?

_Building the "Tally Environment_

6. **Observation Space:** Use a mix of **OCR (Optical Character Recognition)** to read text and **Coordinate Mapping** to understand the "spatial" layout.
7. **State Representation:** Use a **Graph Neural Network (GNN)** to represent the Tally menu structure. This aligns perfectly with Sid’s "Graph Systems" expertise.


### 1. Reward model
Reward Models in Deep Reinforcement Learning: A Survey: https://arxiv.org/pdf/2506.15421
https://cameronrwolfe.substack.com/p/reward-models.
https://medium.com/@vi.ha.engr/building-an-rlhf-pipeline-for-llms-a-beginner-friendly-tutorial-21112bfcff9b

### 2. RL for spatial need
eGUIDE: Data Efficient GUI Grounding via Spatial Reasoning and Search: https://arxiv.org/pdf/2505.15259
UI Agents with Reinforcement Learning - Toward Digital Inhabitants: https://arxiv.org/pdf/2604.27955

GUI (GUI grounding via RL) and visual grounding agents trained with RL to interact with user interfaces.


### 3. RL env for Tally prime

• Build a training environment where an RL agent learns to operate the Tally Prime interface to complete real accounting tasks — like an LLM-based agent that learns to navigate the Tally UI, enter vouchers, run reports, manage ledgers, etc.
• This is agentic RL applied to a real-world enterprise software UI. 
• Build a Gymnasium-style environment that wraps Tally Prime's workflow space, so RL agents can be trained on accounting tasks within it
• Current LLM-RL practice (Memory-R1, WebAgent-R1, Search-R1 all use semantic action spaces of various flavors)

