# Scaling Goal-Conditioned Reinforcement Learning

**Weights & Biases:** [api.wandb.ai/links/aho13-duke-university/x91o1azt](https://api.wandb.ai/links/aho13-duke-university/x91o1azt)

![Environments](assets/envs.gif)
Our work builds on top of [JAXGCRL](https://github.com/MichalBortkiewicz/JaxGCRL), feel free to check it out!!

## Research extensions (this fork)

This repository is a research fork of [Scaling CRL](https://github.com/wang-kevin3290/scaling-crl) (Wang et al., NeurIPS 2025 Best Paper). It investigates how to combine the **depth-scaling** insights from that paper with **isotropic Gaussian representation regularization** introduced by Obando-Ceron et al. (2026) [[arXiv:2602.19373](https://arxiv.org/abs/2602.19373)] and the original SIGReg technique from the LeJEPA / Scaling-CRL work [[arXiv:2503.14858](https://arxiv.org/abs/2503.14858)].

### Motivation

Wang et al. (2025) showed that very deep residual networks (up to 1024 layers) dramatically improve performance on unsupervised goal-conditioned RL tasks. However, as networks grow deeper, **representation collapse** and **neuron dormancy** become increasingly problematic. Obando-Ceron et al. (2026) proved theoretically that isotropic Gaussian embeddings are provably advantageous under the non-stationarity inherent in RL: they induce stable tracking of time-varying targets, achieve maximal entropy under a fixed variance budget, and encourage balanced use of all representational dimensions. Their proposed regularizer, **Sketched Isotropic Gaussian Regularization (SIGReg-ISO)**, is parameter-free — it pushes the empirical characteristic function of 1-D sliced projections of an embedding batch to match that of `N(0,1)`.

**Our core hypothesis:** applying this parameter-free isotropic Gaussian regularizer to the actor trunk of a deep CRL network will reduce collapse at scale and improve policy learning stability, providing a stronger foundation for further depth scaling.

### Experiments

#### 1. ISO actor in off-policy SAC/CRL (`train.py`)

The main experiment integrates `sigreg_iso` as an **actor trunk regularizer** within the standard SAC + contrastive critic training loop.

- **`--iso_actor`**: enables the parameter-free `sigreg_iso` loss on the actor's backbone embedding before the action heads (Obando-Ceron et al., 2026).
- **`--actor_embed_reg`**: the original learned SIGReg-forward path (knot-based, parameterized) for comparison.
- Both paths scale the regularization contribution by `--actor_reg_coeff` (default `0.05`).

`sigreg_iso` hyperparameters:

| Flag | Default | Meaning |
|------|---------|---------|
| `--sigreg_iso_num_slices` | `16` | Number of random 1-D projection directions |
| `--sigreg_iso_num_t` | `8` | Points on the characteristic-function evaluation grid |
| `--sigreg_iso_t_max` | `5.0` | Grid spans `[-t_max, t_max]` |

#### 2. ISO actor in on-policy PPO (`trainISO.py`)

A PPO training loop with an `ISOActor` backbone and `sigreg_iso` applied to the policy trunk at every update step. This tests the same regularization hypothesis under an on-policy learning regime.

#### 3. Frozen-critic actor training (`train_frozen_critic.py`)

Freeze a pretrained deep critic (trained with InfoNCE or SIGReg) and train a fresh deep actor from scratch against the fixed Q-landscape. This isolates **how much of the performance gain from depth comes from the critic's representation** versus the actor, and tests whether a good deep critic can bootstrap a good actor even when the actor starts from random weights.

### Embedding analysis (`src/embedding_metrics.py`)

To understand *why* regularization helps or hurts, we track a suite of representation-quality metrics logged during training:

| Metric | What it measures |
|--------|-----------------|
| `effective_rank` | Roy & Vetterli (2007) — exp(entropy of normalized singular value distribution); proxy for representation capacity actually used |
| `numerical_rank` | Count of singular values above 1% of max; hard rank estimate |
| `isotropy_score` | min / max eigenvalue of the covariance matrix; 1.0 = perfectly isotropic |
| `participation_ratio` | `(Σλ_i)² / Σλ_i²`; another rank proxy robust to small singular values |
| `two_nn_intrinsic_dim` | Facco et al. (2017) two-nearest-neighbor estimator of the manifold dimension |
| `alignment` | Mean L2 distance between matched `(s,a)` and goal embedding pairs |
| `uniformity` | Wang & Isola (2020) log-mean-exp metric on pairwise distances; measures spread |
| `pos_neg_ratio` | Mean positive-pair distance / mean negative-pair distance; should be ≪ 1 |

### Codebase structure

Code is factored into `src/` for clean reuse across training scripts:

```
src/
  args.py              # All CLI flags (shared across train.py / trainISO.py)
  networks.py          # UnifiedEncoder, SA_encoder, G_encoder, Actor, ISOActor
  loss.py              # sigreg_forward, sigreg_iso (SIGReg-ISO), lejepa_loss, SIGRegModule
  buffer.py            # Trajectory replay buffer
  evaluator.py         # Goal-conditioned success evaluator
  env_factory.py       # Brax environment construction
  embedding_metrics.py # Representation quality diagnostics
  types.py             # TrainingState, ISOTrainingState, Transition
  utils.py             # Checkpoint save/load
```

### Results

![Preliminary results](/assets/preliminary_results.png)

**Humanoid** evaluation success rate across training steps. 
- Green (on-policy): Isotropic On-Policy (PPO + `sigreg_iso` actor, `trainISO.py`). 
- Blue (off-policy): original 1000-layer SSLRL baseline (`train.py`, no ISO regularization). 
- Red (off-policy): SAC with isotropic policy and contrastive critic (`train.py --iso_actor`).

### Running experiments

**YAML-based SLURM submission** (recommended on a cluster):

```sh
# Preview the generated sbatch script without submitting
python scripts/slurm_from_yaml.py configs/train/humanoid_iso.yaml train.py --dry-run

# Submit to SLURM
python scripts/slurm_from_yaml.py configs/train/humanoid_iso.yaml train.py
python scripts/slurm_from_yaml.py configs/trainISO/humanoid.yaml trainISO.py
```

**Example configs**:

| Config | Setup |
|--------|-------|
| `configs/train/humanoid_iso.yaml` | Humanoid, SAC + deep network, `iso_actor` (parameter-free SIGReg-ISO on actor trunk) |
| `configs/train/humanoid_offline_sigreg_actor_infonce.yaml` | Humanoid, SAC, learned SIGReg-forward on actor, InfoNCE critic (offline W&B logging) |
| `configs/train/reacher.yaml` | Reacher baseline, shallow network |
| `configs/trainISO/humanoid.yaml` | Humanoid, PPO + ISOActor + `sigreg_iso` on policy backbone |

### Citation for SIGReg-ISO

If this work is useful to you, please also cite the SIGReg-ISO paper:

```bibtex
@article{obando2026stable,
  title   = {Stable Deep Reinforcement Learning via Isotropic Gaussian Representations},
  author  = {Obando-Ceron, Johan and others},
  journal = {arXiv preprint arXiv:2602.19373},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.19373}
}
```

# Installation

```sh
uv sync
```
Then just fix the two Brax issues described below, and you'll be all set.


## Fixing two bugs in brax 0.10.1
1. There is a minor bug in brax's contact.py file. To fix it, first locate the brax contact.py file in your virtual environment: 
```
find .venv -name contact.py
```
Then open the file and replace it with the following code:
```python
from typing import Optional
from brax import math
from brax.base import Contact
from brax.base import System
from brax.base import Transform
import jax
from jax import numpy as jp
from mujoco import mjx

def get(sys: System, x: Transform) -> Optional[Contact]:
    """Calculates contacts.
    Args:
        sys: system defining the kinematic tree and other properties
        x: link transforms in world frame
    Returns:
        Contact pytree
    """
    #NOTE: THIS WAS MODIFIED SINCE AFTER MUJOCO 3.1.5, mjx.ncon IS NOT AVAILABLE
    # ncon = mjx.ncon(sys)
    # if not ncon:
    #   return None
    data = mjx.make_data(sys)
    if data.ncon == 0:
        return None
    @jax.vmap
    def local_to_global(pos1, quat1, pos2, quat2):
        pos = pos1 + math.rotate(pos2, quat1)
        mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
        return pos, mat
    x = x.concatenate(Transform.zero((1,)))
    xpos = x.pos[sys.geom_bodyid - 1]
    xquat = x.rot[sys.geom_bodyid - 1]
    geom_xpos, geom_xmat = local_to_global(
        xpos, xquat, sys.geom_pos, sys.geom_quat
    )
    # pytype: disable=wrong-arg-types
    d = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
    d = mjx.collision(sys, d)
    # pytype: enable=wrong-arg-types
    c = d.contact
    elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5
    body1 = jp.array(sys.geom_bodyid)[c.geom1] - 1
    body2 = jp.array(sys.geom_bodyid)[c.geom2] - 1
    link_idx = (body1, body2)
    return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
```
2. There is also a minor bug in brax's json.py file. To fix it, first locate the brax json.py file in your virtual environment:
```
find .venv -name json.py | grep "/brax/io/json.py"
```
Then open the file and change the if statement in line 159 to:  
```python
if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():
```


# Running experiments
Now, we are ready to run the train script. To run the code, you'll need a GPU. For Humanoid-based environments, it may require up to 80GB of GPU memory (for deep networks). Below is an example command to run the training script (an additional example can be found in the provided slurm script `job.slurm`): 

```sh
uv run train.py --env_id "humanoid" --eval_env_id "humanoid" --num_epochs 100 --total_env_steps 100000000 --critic_depth 16 --actor_depth 16 --actor_skip_connections 4 --critic_skip_connections 4 --batch_size 512 --vis_length 1000 --save_buffer 0 
```


>[!NOTE]
>If you would like the experiments to be synced to wandb, you should go to `train.py` and replace the default values of `wandb_entity` and `wandb_project_name` (line 34-35 of the `train.py` file) with your particular wandb entity and wandb project name. Alternatively, these two can also be set as hyperparameter flags when running the train script.

# Citing Scaling CRL 📜
```bibtex
@inproceedings{wang2025,
  title     = {1000 Layer Networks for Self-Supervised {RL}: Scaling Depth Can Enable New Goal-Reaching Capabilities},
  author    = {Kevin Wang and Ishaan Javali and Micha{\l} Bortkiewicz and Tomasz Trzcinski and Benjamin Eysenbach},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025},
  url       = {https://openreview.net/forum?id=s0JVsx3bx1}
}
```



<!-- 
## Troubleshooting Potential Errors

**If you encounter the following error:**
```AttributeError: module 'mujoco.mjx' has no attribute 'ncon'```  

**Fix:**
1. Locate the brax contact.py file in your conda environment: 
   ```
   find ~/.conda/envs/scaling-crl -name contact.py
   ```
2. Open the file and replace it with the following code:

    ```python
    from typing import Optional
    from brax import math
    from brax.base import Contact
    from brax.base import System
    from brax.base import Transform
    import jax
    from jax import numpy as jp
    from mujoco import mjx

    def get(sys: System, x: Transform) -> Optional[Contact]:
        """Calculates contacts.
        Args:
            sys: system defining the kinematic tree and other properties
            x: link transforms in world frame
        Returns:
            Contact pytree
        """
        #NOTE: THIS WAS MODIFIED SINCE AFTER MUJOCO 3.1.5, mjx.ncon IS NOT AVAILABLE
        # ncon = mjx.ncon(sys)
        # if not ncon:
        #   return None
        data = mjx.make_data(sys)
        if data.ncon == 0:
            return None
        @jax.vmap
        def local_to_global(pos1, quat1, pos2, quat2):
            pos = pos1 + math.rotate(pos2, quat1)
            mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
            return pos, mat
        x = x.concatenate(Transform.zero((1,)))
        xpos = x.pos[sys.geom_bodyid - 1]
        xquat = x.rot[sys.geom_bodyid - 1]
        geom_xpos, geom_xmat = local_to_global(
            xpos, xquat, sys.geom_pos, sys.geom_quat
        )
        # pytype: disable=wrong-arg-types
        d = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
        d = mjx.collision(sys, d)
        # pytype: enable=wrong-arg-types
        c = d.contact
        elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5
        body1 = jp.array(sys.geom_bodyid)[c.geom1] - 1
        body2 = jp.array(sys.geom_bodyid)[c.geom2] - 1
        link_idx = (body1, body2)
        return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
    ```
3. Save the file and rerun the training script.


**If you encounter the following error:** ```Error rendering final policy: unsupported operand type(s) for ==: 'ArrayImpl' and 'list'```  

**Fix:**
1. Locate the brax json.py file in your conda environment:
   ```
   find ~/.conda/envs/scaling-crl -name json.py | grep "/brax/io/json.py"
   ```
2. Open the file and change the if statement in line 159 to:
    ```python
    if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():
    ```
3. Save the file and rerun the training script. -->

