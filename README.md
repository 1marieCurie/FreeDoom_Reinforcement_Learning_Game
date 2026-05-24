# 🎮 FreeDoom — Deep Reinforcement Learning for Deathmatch Combat

> **Training deep RL agents to fight in a VizDoom deathmatch environment using PPO and Dueling DRQN.**

**Authors:** Ouali Yassine & Mellak Khadija  
**Institution:** Department of Intelligent Systems, ENSA Fez — University of Sidi Mohammed Ben Abdellah  
**Contact:** yassine.ouali@usmba.ac.ma · khadija.mellak@usmba.ac.ma

---

## 📋 Table of Contents

- [Overview](#overview)
- [Agents](#agents)
- [Project Structure](#project-structure)
- [Environment Design](#environment-design)
- [Reward Functions](#reward-functions)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [References](#references)
- [Demo](#demo)

---

## Overview

FreeDoom investigates two complementary deep reinforcement learning strategies for first-person combat in a [VizDoom](https://vizdoom.cs.put.edu.pl/) deathmatch scenario:

| Agent | Algorithm | Environment | Training Steps | Key Result |
|---|---|---|---|---|
| **PPO Agent** | Proximal Policy Optimization | Spatially constrained arena (600×600 units), 2 bots | 301 056 | 50% success rate (both enemies eliminated) |
| **Arnold** | Dueling Double Deep Recurrent Q-Network (Dueling DRQN) | Full deathmatch map, 4 bots, 5 rotating maps | 3 271 025 | 16.8 mean kills/episode, K/D = 10.68 |

The core research question is: *can a simpler on-policy method (PPO) with spatial constraints match the performance of a richer value-based recurrent architecture (Dueling DRQN) on this sparse, adversarial task?*

---

## Agents

### 1. PPO Agent (`doom_ppo_v1_3`)

Uses **Proximal Policy Optimization** with a CNN policy (Stable-Baselines3). A key design choice is a **spatial constraint**: the agent is bounded to a 600×600 unit arena centered on its spawn point, preventing aimless map wandering and forcing enemy encounters. The reward was iteratively refined over 4 training experiments (`Essai 1` → `Essai 4`).

### 2. Arnold (Dueling DRQN)

Uses a **Dueling Double Deep Recurrent Q-Network** on the full deathmatch map. Architecture extensions over standard DQN:
- **Dueling streams** — separates Q-value into state-value V(s) and advantage A(s,a)
- **LSTM layer** — maintains memory of recent events (enemy positions, shot trajectories)
- **Auxiliary game-feature head** — predicts binary enemy visibility, acting as a perceptual regularizer (inspired by [Lample & Chaplot, 2016](https://arxiv.org/abs/1609.05521))

---

## Project Structure

```
freedoom/
│
├── doom_env.py              # Gymnasium wrapper for VizDoom (DoomEnv class)
├── train.py                 # PPO training script (Stable-Baselines3)
├── test.py                  # PPO evaluation script (10 test episodes)
├── arnoldv2.ipynb           # Arnold (Dueling DRQN) training notebook
│
├── deathmatch_mine.cfg      # VizDoom scenario configuration (required)
├── deathmatch_mine.wad      # VizDoom map file (required)
│
├── checkpoints/             # PPO model checkpoints (auto-created during training)
├── ppo_doom_tensorboard/    # TensorBoard logs (auto-created during training)
│
└── doom_ppo_v1_3.zip        # Final saved PPO model
```

---

## Environment Design

### VizDoom Configuration

| Parameter | Value |
|---|---|
| Scenario | `deathmatch_mine.cfg` / `deathmatch_mine.wad` |
| Difficulty | Skill level 3 (intermediate) |
| Episode timeout | 4 200 game tics |
| Training resolution | 320×240 (HUD disabled) |
| Inspection resolution | 640×480 |
| Training mode | `ASYNC_PLAYER` |
| Test mode | `PLAYER` |

### Observation Space

Raw VizDoom screen buffers are preprocessed at every step:

1. Transpose `(C, H, W)` → `(H, W, C)` if needed
2. Grayscale → RGB conversion if single-channel
3. Clip to 3 channels
4. Resize to **84×84** pixels (OpenCV)
5. Cast to `uint8`

Final observation space: `Box(0, 255, shape=(84, 84, 3), dtype=uint8)`

### Action Space

8 discrete macro-actions (`Discrete(8)`), each executed for **4 consecutive game tics** (frame skip = 4):

| Index | Action | Buttons |
|---|---|---|
| 0 | Move forward | `MOVE_FORWARD` |
| 1 | Move backward | `MOVE_BACKWARD` |
| 2 | Strafe left | `MOVE_LEFT` |
| 3 | Strafe right | `MOVE_RIGHT` |
| 4 | Turn left | `TURN_LEFT` |
| 5 | Turn right | `TURN_RIGHT` |
| 6 | Attack | `ATTACK` |
| 7 | Move forward + Attack | `MOVE_FORWARD` + `ATTACK` |

### Game Variables

Eight variables are tracked at each step:

| Idx | Variable | Role |
|---|---|---|
| 0 | `KILLCOUNT` | Kill reward signal |
| 1 | `HEALTH` | Damage & low-health penalty |
| 2 | `ARMOR` | Reserved (unused) |
| 3 | `SELECTED_WEAPON` | Reserved (unused) |
| 4 | `S_WEAPON_AMMO` | Ammunition penalty |
| 5 | `POSITION_X` | Spatial constraint |
| 6 | `POSITION_Y` | Spatial constraint |
| 7 | `DAMAGECOUNT` | Damage-received penalty (cumulative delta) |

### Spatial Constraint (PPO Agent)

The episode terminates immediately with a −1.0 zone penalty if the agent exceeds 300 game units in either axis from its spawn point:

```
|x_t − x_0| > 300  OR  |y_t − y_0| > 300  →  episode terminated
```

This creates a pseudo 600×600 unit arena that prevents wandering and increases enemy encounter density during early training.

---

## Reward Functions

### PPO Agent — Per-Step Reward

```
r_t = r_kill + r_surv + r_time + r_dmg + r_ammo + r_zone + r_death
```

| Component | Condition | Value | Purpose |
|---|---|---|---|
| Kill bonus | Per new kill | **+1.0** | Primary objective |
| Survival bonus | Every step | +0.005 | Discourage passive death |
| Time penalty | Every step | −0.001 | Discourage idling |
| Damage penalty | Per damage unit Δd | −0.02 × Δd | Penalize absorbing hits |
| Ammo penalty | Per ammo unit Δa | −0.003 × Δa | Discourage random firing |
| Zone penalty | Exit 300-unit bound | −1.0 | Hard spatial constraint |
| Death penalty | Agent dies | −0.3 | Penalize terminal failure |
| Low-health penalty | Health < 25 | −0.01/step | Avoid dangerous situations |

### Arnold — Per-Step Reward

```
r_t = r_kill + r_death + r_dist + r_stale + r_health
```

| Component | Value | Purpose |
|---|---|---|
| Kill bonus | **+30** | Primary combat objective |
| Death penalty | −10 | Penalize terminal failures |
| Distance bonus | +0.005 × d | Encourage map exploration |
| Stale penalty | −0.1/step | Discourage standing still |
| Health pickup | +1 | Reward self-preservation |

> Kill–reward Pearson correlation r = **0.984**, confirming the reward is well-aligned with the primary objective.

---

## Installation

### Prerequisites

- Python 3.8+
- VizDoom (with `deathmatch_mine.cfg` and `.wad` scenario files)
- CUDA-capable GPU recommended for training

### Install Dependencies

```bash
pip install vizdoom gymnasium stable-baselines3 opencv-python numpy torch tensorboard
```

> For Arnold (Dueling DRQN), additional dependencies used in the notebook may be required.

---

## Usage

### Training the PPO Agent

```bash
python train.py
```

This will:
- Initialize `DoomEnv` in `ASYNC_PLAYER` mode
- Train a PPO agent with `CnnPolicy` for 300 000 timesteps
- Save checkpoints every 50 000 steps to `./checkpoints/`
- Log training metrics to `./ppo_doom_tensorboard/`
- Save the final model as `doom_ppo_v1_3.zip`

**Key hyperparameters (`train.py`):**

| Parameter | Value |
|---|---|
| Policy | `CnnPolicy` |
| Learning rate | 1e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| gamma | 0.99 |
| Total timesteps | 300 000 |

### Monitoring Training

```bash
tensorboard --logdir ./ppo_doom_tensorboard/
```

### Testing the PPO Agent

```bash
python test.py
```

Runs 10 evaluation episodes with `deterministic=True` and renders a visible window. Prints step-level stats every 50 steps and a summary at the end of each episode.

### Training Arnold (Dueling DRQN)

Open and run `arnoldv2.ipynb` in Jupyter. The notebook contains the full Dueling DRQN implementation, training loop, and evaluation code.

---

## Results

### PPO Agent — Essai 4 (`doom_ppo_v1_3`)

**Training (301 056 steps, ~13.85 hours):**

| Metric | Value |
|---|---|
| Final mean episode length | ~103.6 steps (+48% from start) |
| Final mean episode reward | ~0.38 (started at ~−0.2) |
| Peak episode reward | ~0.45 (at ~150k steps) |

**Test evaluation (10 episodes):**

| Episode | Reward | Kills | Steps | Outcome |
|---|---|---|---|---|
| 1 | 2.180 | 2 | 167 | ✅ Success |
| 2 | −0.324 | 0 | 84 | ❌ Failure |
| 3 | 2.180 | 2 | 167 | ✅ Success |
| 4 | −0.617 | 0 | 107 | ❌ Failure |
| 5 | −0.818 | 0 | 50 | ❌ Failure |
| 6 | 2.145 | 2 | 154 | ✅ Success |
| 7 | −0.004 | 0 | 168 | ❌ Failure |
| 8 | 2.180 | 2 | 167 | ✅ Success |
| 9 | −0.873 | 0 | 34 | ❌ Failure |
| 10 | 2.180 | 2 | 167 | ✅ Success |

**Summary: 5/10 success rate · Mean reward = 0.623 · Mean episode length = 126.5 steps**

### Arnold (Dueling DRQN) — Final Performance

| Metric | Value |
|---|---|
| Total training steps | 3 271 025 |
| Mean episode kills | **16.83** |
| Peak episode kills | **43** |
| Mean K/D ratio | **10.68** |
| % episodes with K/D > 1 | 86.8% |
| Greedy eval K/D (final) | **5.40** |
| Mean episode reward | 503.0 |
| Peak episode reward | 1 307.3 |
| Mean TD loss | 0.2384 |
| Reward–kill correlation | **r = 0.984** |

### PPO Training Experiments Summary

| Experiment | Key Change | Outcome |
|---|---|---|
| Essai 1 | 5-action baseline, no health/ammo | Proof-of-concept |
| Essai 2 (`doom_ppo_v1_1`) | Fixed `DAMAGECOUNT` as penalty (not bonus) | ~100k steps, unstable (ep_len 70–85) |
| Essai 3 (`doom_ppo_v1_2`) | Kill reward +50, survival bonus, time penalty | 1M steps / ~38h; degradation at 550k steps |
| Essai 4 (`doom_ppo_v1_3`) | Kill reward normalized to +1.0, ammo penalty, health tracking | **Final model — 50% success rate** |

---

## Limitations

- **No outgoing damage signal** — VizDoom's default config does not expose damage inflicted on enemies; only kill events provide combat feedback. Custom ACS scripting would be needed.
- **Spatial constraint approximation** — The 300-unit bound is a rough proxy; a curriculum approach expanding the zone as skill improves would be more principled.
- **Insufficient training budget** — PPO clipping fraction never fell below 0.40, confirming the policy had not converged at 300k steps. Extending to 1M–5M steps is recommended.
- **Single-seed evaluation** — All runs used one random seed; multi-seed experiments with statistical significance tests are needed for robust conclusions.
- **Limited action space** — The 8 macro-actions omit strafing-while-turning and crouching; Arnold's factored action space handles this more flexibly.

---

## References

1. J. Schulman et al., "Proximal policy optimization algorithms," *arXiv:1707.06347*, 2017.
2. G. Lample and D. S. Chaplot, "Playing FPS games with deep reinforcement learning," *arXiv:1609.05521*, 2016.
3. M. Kempka et al., "ViZDoom: A Doom-based AI research platform for visual reinforcement learning," *IEEE CIG*, 2016.
4. M. Abadi et al., "TensorFlow: A system for large-scale machine learning," *OSDI*, 2016.
5. M. Towers et al., "Gymnasium: A standard interface for reinforcement learning environments," *NeurIPS*, 2025.
6. Z. Wang et al., "Dueling network architectures for deep reinforcement learning," *ICML*, 2016.
7. M. Hausknecht and P. Stone, "Deep recurrent Q-learning for partially observable MDPs," *AAAI*, 2015.
8. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, 2015.

---

## Demo

### PPO Agent in Action

> 📸 *Place a screenshot or GIF of the PPO agent playing here*

![PPO Agent Demo](demo/ppo_demo.png)

---

### Arnold (Dueling DRQN) in Action

> 📸 *Place a screenshot or GIF of Arnold playing here*

![Arnold Demo](demo/arnold_demo.png)

---

*FreeDoom — ENSA Fez, University of Sidi Mohammed Ben Abdellah*
