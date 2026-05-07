import os
import random
import numpy as np
from code.train import run_training
from code.evaluate import run_evaluation

seed = 42


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    set_seeds(seed)
    os.makedirs("artifacts/models", exist_ok=True)

    run_training(total_timesteps=300_000, seed=seed)
    run_evaluation()


if __name__ == "__main__":
    main()