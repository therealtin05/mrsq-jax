# MRSQ-JAX

JAX re-implementation of **The Surprising Difficulty of Search in Model-Based Reinforcement Learning**.

Paper: https://arxiv.org/abs/2601.21306

Kaggle setup: https://www.kaggle.com/code/therealtin/public-mrsq-jax

## Results

![LAP Comparision](media/comparison.png)

## Installation

```bash
conda create -n mrsq-jax python=3.10
conda activate mrsq-jax

pip install -r requirements.txt
pip install -e .
```

## Training

Example:

```bash
python3 main.py env.env_name=humanoid-run env.backend=dmc
python3 main.py env.env_name=HalfCheetah-v4 env.backend=gymnasium mrsq.episodic=true
python3 main.py env.env_name=h1-sit_simple-v0 env.backend=humanoid-bench mrsq.episodic=true
```

## Acknowledgements

This implementation is inspired by:

* https://github.com/adaptive-intelligent-robotics/QDax
* https://github.com/ShaneFlandermeyer/tdmpc2-jax

Some code structure and implementation details follow ideas from these projects.