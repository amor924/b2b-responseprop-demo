from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Reproducibility
    random_state: int = 42

    # Data sizes
    n_train: int = 50000
    n_future: int = 20000  # for drift simulation

    # Split
    test_size: float = 0.25

    # Modeling
    C: float = 1.0  # inverse regularization strength
    max_iter: int = 200

    # Targeting
    n_deciles: int = 10

CFG = Config()
