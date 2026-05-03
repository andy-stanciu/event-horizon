# Systematic Study of World-Model Exploitation Over Long Imagination Horizons

**Alexander Chin, Andy Stanciu, Marcin Anforowicz**
University of Washington
{alexchin, andys22, manfor}@cs.washington.edu

---

## 1 Introduction

Model-based reinforcement learning agents train a policy by imagining trajectories inside a learned world model. The actor maximizes predicted reward along imagined rollouts. This works when the world model is accurate. But the world model is never perfect. Small prediction errors compound over multi-step rollouts.

This creates a failure mode called **world-model exploitation**. The actor is not adversarial by design, but optimization pressure is adversarial in effect. If any region of state space produces falsely high predicted reward, gradient descent steers the policy toward it. The model becomes a liability.

Researchers have intuitions about when this happens. DreamerV3 caps its imagination horizon at 15 steps. TD-MPC2 plans over only 3. Drama (Mamba-2) and STORM (transformer) use long-context backbones but still train actors on short horizons. The implicit claim is that longer horizons make exploitation worse. This has never been systematically tested across backbones.

We propose such a study. We evaluate five representative algorithms across four benchmark families, sweeping imagination horizon across a wide range. We measure both real-environment performance and the gap between imagined and actual return. This tells us where each algorithm's world model breaks down, and whether modern long-context backbones translate their memory advantages into policy performance or whether compounding error dominates.

---

## 2 Proposal

### Algorithms

- **PPO**: model-free baseline. No world model, so cannot exploit one.
- **DreamerV3**: latent-imagination agent with a GRU-based RSSM. Default horizon 15.
- **TD-MPC2**: learned latent model with CEM planning and TD value estimates. Default planning horizon 3.
- **Drama**: DreamerV3 with the GRU replaced by a Mamba-2 selective state-space model. Default horizon 15.
- **STORM**: transformer-based world model in the Dreamer framework. Default horizon 15.

We use official releases: `dreamerv3-torch`, `tdmpc2`, `realwenlongwang/Drama`, `weipu-zhang/STORM`, and `CleanRL` for PPO.

### Benchmarks

- **DeepMind Control (DMC) proprioceptive suite**: smooth dynamics, dense rewards. Sanity check.
- **Craftax**: procedurally generated 2D Minecraft-like environment. Tests sparse rewards and long horizons.
- **Memory Maze**: tests long-term memory in the world model.
- **MetaWorld**: robotic manipulation. Tests discontinuous contact dynamics.

### Stress-test Axis

We sweep imagination horizon \(H \in \{5, 15, 30, 60\}\) for the four model-based algorithms. All other hyperparameters stay at published defaults. PPO is included as a no-world-model control and is unaffected by \(H\).

### Methodology

Three seeds per configuration, reporting mean and standard error. Fixed environment-step budgets:

- 500k for DMC
- 1M for Craftax
- 500k for Memory Maze
- 1M for MetaWorld

We log:

1. Episode return and success rate in the real environment.
2. World-model prediction MSE as a function of rollout step — shows where dynamics become unreliable.
3. The imagination-reality gap: predicted return in imagination minus actual return in the environment. A widening gap at longer horizons is a direct signature of exploitation.

### Impacts

Practitioners currently pick RL algorithms by loose analogy to published results. The default imagination horizons are treated as fixed. Nobody has tested whether these defaults generalize or whether long-context backbones benefit from longer horizons. A systematic characterization would make algorithm selection more principled and would serve as a common reference for future papers.

### Novelty

Papers introducing model-based RL methods typically report results at a single horizon on one or two favorable benchmarks. They do not systematically characterize exploitation as a function of imagination depth. We are not aware of existing work that evaluates recent methods on a common protocol across memory-demanding, procedural, continuous-control, and contact-rich benchmarks while varying imagination horizon.

### How Learning and Probabilistic Inference Play a Role

All methods are deep-learning-based. DreamerV3, Drama, and STORM are trained with an ELBO and KL regularization. TD-MPC2 combines learned latent dynamics with model-predictive planning. PPO uses stochastic policy gradients with a learned value function.

### Success Metrics

We consider the project successful if we produce:

1. Baseline performance tables for each algorithm-benchmark pair at default horizon.
2. Performance-vs-horizon curves across all four benchmarks.
3. World-model prediction error as a function of rollout step.
4. Imagination-reality gap as a function of horizon.
5. **Stretch**: wall-clock and GPU-hour cost per algorithm, and whether long-context backbones show smaller imagination-reality gaps than the Dreamer GRU at matched horizons.

### Challenges

- **Heterogeneous codebases**:
  - Five codebases with different dependencies, configs, and logging conventions.
  - Requires instrumenting each codebase to log predicted and actual returns.
- **Hyperparameter fairness**:
  - Each algorithm is tuned for its native benchmarks. We use published defaults and document deviations.
- **Compute**:
  - ~400 GPU-hours on Hyak is tight. We prioritize DMC and Craftax before Memory Maze and MetaWorld.

### Software and Compute

- **Compute**: UW's Hyak HPC cluster, four Quadro RTX 6000s for two weeks. Colab Pro for development.
- **Codebases**: `dreamerv3-torch`, `nicklashansen/tdmpc2`, `realwenlongwang/Drama`, `weipu-zhang/STORM`, `vwxyzjn/cleanrl`.
- **Benchmarks**: `dm_control`, `Craftax`, `jurgisp/memory-maze`, `Farama-Foundation/Metaworld`.
- **Analysis**: Weights & Biases, matplotlib, seaborn, pandas.

