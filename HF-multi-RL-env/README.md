_Environment count is becomes a dominant scaling axis for post-training. Frontiers models are gettoing trained in multiple environments_

### Three-tier taxonomy
1. **(Tier 1) Pure task libraries:** provide just problems plus verifiers - RLVE, Reasoning Gym. No transport, no tools, no state. These are not environments in the operational sense; they're datasets with grading functions.
2. **(Tier 2) Environment frameworks:** define how to build an environment but expect you to bring your own trainer — OpenEnv, ORS, NeMo Gym, RL-Factory. 
3. **(Tier 3) Environment-plus-training-bundled:** ships env definition, rollout, and training in one package — Verifiers, SkyRL Gym, GEM, plus the ones the paper excluded from comparison (Atropos, Harbor, RAGEN, rLLM).

### How an RL training system fits together
```
Every RL training system for LLMs sits on the same five-stage spine: Tasks → Harness → Reward → Rollout → Trainer.
```

* OpenEnv covers Harness + Reward only.
* ORS covers Tasks + Harness + Reward. 
* NeMo Gym extends through Rollout (including Tasks + Harness + Reward ).
* Verifiers + Prime RL covers the entire 5 spine: Tasks + Harness + Reward + Rollout + Trainer
* SkyRL Gym covers Harness + Reward + Rollout + Trainer
* GEM covers Tasks + Harness + Reward + Rollout.

### Internalization:  What makes an RL environment for LLMs - 11 components

1. Tasks/Dataset: What problems should the model solve
2. Initial State Management: How is per-episode state set up at the start of a rollout?
3. Prompt Template
4. Tool Definition
5. Observation Format
6. Execution Backend
7. State Management
8. Reward/Rubric
9. Done/Termination
10. Episode Control
11. Transport/Protocol, and 
* (implicit twelfth: ) Native Training Integration.

### The four reward patterns
1. External reward (SkyRL Gym, GEM) — environment returns text, your training script computes the reward. Maximum flexibility, natural fit for offline RL or distillation, but you write every scoring function from scratch.
2. Server-embedded reward (ORS) — every tool response carries a reward inline; the server scores as it goes. Dense per-step credit assignment, awkward when reward only makes sense at trajectory end.
3. Post-episode verify (NeMo Gym) — a separate /verify endpoint scores the full trajectory after rollout completes. Natural fit for unit-test suites or LLM-as-judge over whole conversations.
4. Environment-embedded rubric (OpenEnv, Verifiers) — a composable Rubric object computes rewards during step() with primitives like WeightedSum, Sequential, Gate, and LLMJudge.

The reason this dimension is more important than transport (HTTP vs in-process): reward shape determines what algorithm you can train. Sparse trajectory-level rewards point you toward REINFORCE-style methods and GRPO. Dense per-step rewards open up PPO with proper advantage estimation. The paper's separate note about what's inside a reward function — procedural/verifiable (RLVR pattern) versus LLM-as-judge versus dense-vs-sparse — is the conceptual bridge to the academic RL literature. In practice you almost always end up with a mix: procedural for what you can verify, LLM-judge for what you can't, composed with weights into a single scalar.

> "Each framework is the same thing wearing different clothes. The same environment ports across all six. What changes between them is how the env wires into the rest of training, not what it can do. You won't miss anything fundamental by picking one — what changes is convenience, which one is most pleasant depends on what's already in your stack."