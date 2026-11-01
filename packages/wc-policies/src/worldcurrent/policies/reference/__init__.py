"""Reference policies: one small implementation per policy family, numpy only, no downloaded weights.

They are trained from your own runs (development splits only, sealed as artifacts with provenance) and run through
the same agents, safety shield, benchmarks and gate as any other policy. They are meant for teaching, testing the
stack end to end in seconds, and as baselines on low-dimensional state; they are not the published architectures
at scale. For those, use the LeRobot policies (ACT, Diffusion, VQ-BeT, pi0, SmolVLA, ...) or openpi servers.

| Plugin          | Family                       | Idea (reference)                                                    | Stochastic |
|-----------------|------------------------------|---------------------------------------------------------------------|------------|
| policy.knn      | retrieval / non-parametric   | nearest neighbours in observation space, average their chunks (VINN) | no         |
| policy.bc       | regression behaviour cloning | MLP from observation to an action chunk, MSE (BC with chunking)      | no         |
| policy.flow     | generative, flow matching    | velocity field from noise to the chunk, Euler integration (pi0, SmolVLA action experts) | yes |
| policy.tokens   | tokenized actions            | each action dimension binned, categorical heads, sampled (RT-1, OpenVLA-style discretization) | yes |
| policy.mpc      | model-based planning         | cross-entropy method over action sequences in a learned world model (MPC, TD-MPC without the value) | yes |

Run any of them with the "policy" agent, ACT-style with "temporal_ensemble", or steer the stochastic ones with
"steered" (seeded noise, sample-and-select). Learners of the same names build the artifacts:
`wc learn job.json` with {"learner": "policy.flow", "inputs": ["runs/demos"], "out": "artifacts/flow"}.
"""

# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {"policies": "wc.policy/v0.1", "learners": "wc.learner/v0.1"},
    "maturity": "implemented",
}
