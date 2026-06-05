# Code for the paper "Topological Out-of-Domain Generalization in Dynamical Systems Reconstruction" submitted to NeurIPS 2026

Models used in the paper are defined under `src/models`, training with teacher forcing and regularizations is defined under `src/training`. To train a single model, check out `src/experiment_scripts/single_experiment.py` (run `python3 src/experiment_scripts/single_experiment.py --help` to see all options), for a full comparison of different models and datasets see `src/experiment_scripts/full_run.py` (run with `--help` to see options again).
