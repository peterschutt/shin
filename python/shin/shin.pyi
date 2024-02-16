from typing import Sequence

def optimise(
    inverse_odds: list[float],
    sum_inverse_odds: float,
    n: int,
    max_iterations: int = ...,
    convergence_threshold: float = ...,
) -> tuple[float, float, float]: ...
def calculate_implied_odds(
    probabilities: Sequence[float], overround: float
) -> list[float]: ...
