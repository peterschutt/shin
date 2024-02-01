from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

from .shin import optimise as _optimise_rust

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

T = TypeVar("T")
OutputT = TypeVar("OutputT", bound="list[float] | dict[Any, float]")


def _optimise(
    inverse_odds: list[float],
    sum_inverse_odds: float,
    n: int,
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-12,
) -> tuple[float, float, float]:
    delta = float("Inf")
    z = 0.0
    iterations = 0
    while delta > convergence_threshold and iterations < max_iterations:
        z0 = z
        z = (
            sum(
                sqrt(z**2 + 4 * (1 - z) * io**2 / sum_inverse_odds)
                for io in inverse_odds
            )
            - 2
        ) / (n - 2)
        delta = abs(z - z0)
        iterations += 1
    return z, delta, iterations


@dataclass
class ShinOptimisationDetails(Generic[OutputT]):
    implied_probabilities: OutputT
    iterations: float
    delta: float
    z: float

    @overload
    def __getitem__(self, key: Literal["implied_probabilities"]) -> OutputT:
        ...

    @overload
    def __getitem__(self, key: Literal["iterations"]) -> float:
        ...

    @overload
    def __getitem__(self, key: Literal["delta"]) -> float:
        ...

    @overload
    def __getitem__(self, key: Literal["z"]) -> float:
        ...

    def __getitem__(
        self, key: Literal["implied_probabilities", "iterations", "delta", "z"]
    ) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)


# sequence input, full output False
@overload
def calculate_implied_probabilities(
    odds: Sequence[float],
    *,
    max_iterations: int = ...,
    convergence_threshold: float = ...,
    full_output: Literal[False] = False,
    force_python_optimiser: bool = ...,
) -> list[float]:
    ...


# mapping, full output False
@overload
def calculate_implied_probabilities(
    odds: Mapping[T, float],
    *,
    max_iterations: int = ...,
    convergence_threshold: float = ...,
    full_output: Literal[False] = False,
    force_python_optimiser: bool = ...,
) -> dict[T, float]:
    ...


# sequence, full output True
@overload
def calculate_implied_probabilities(
    odds: Sequence[float],
    *,
    max_iterations: int = ...,
    convergence_threshold: float = ...,
    full_output: Literal[True],
    force_python_optimiser: bool = ...,
) -> ShinOptimisationDetails[list[float]]:
    ...


# mapping, full output True
@overload
def calculate_implied_probabilities(
    odds: Mapping[T, float],
    *,
    max_iterations: int = ...,
    convergence_threshold: float = ...,
    full_output: Literal[True],
    force_python_optimiser: bool = ...,
) -> ShinOptimisationDetails[dict[T, float]]:
    ...


def calculate_implied_probabilities(
    odds: Sequence[float] | Mapping[T, float],
    *,
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-12,
    full_output: bool = False,
    force_python_optimiser: bool = False,
) -> (
    ShinOptimisationDetails[list[float]]
    | ShinOptimisationDetails[dict[T, float]]
    | list[float]
    | dict[T, float]
):
    odds_seq = odds.values() if isinstance(odds, Mapping) else odds

    if len(odds_seq) < 2:
        raise ValueError("len(odds) must be >= 2")

    if any(o < 1 for o in odds_seq):
        raise ValueError("All odds must be >= 1")

    optimise = _optimise if force_python_optimiser else _optimise_rust

    n = len(odds_seq)
    inverse_odds = [1.0 / o for o in odds_seq]
    sum_inverse_odds = sum(inverse_odds)

    if n == 2:
        diff_inverse_odds = inverse_odds[0] - inverse_odds[1]
        z = ((sum_inverse_odds - 1) * (diff_inverse_odds**2 - sum_inverse_odds)) / (
            sum_inverse_odds * (diff_inverse_odds**2 - 1)
        )
        delta = 0.0
        iterations = 0.0
    else:
        z, delta, iterations = optimise(
            inverse_odds=inverse_odds,
            sum_inverse_odds=sum_inverse_odds,
            n=n,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )

    p_gen = (
        (sqrt(z**2 + 4 * (1 - z) * io**2 / sum_inverse_odds) - z) / (2 * (1 - z))
        for io in inverse_odds
    )
    if isinstance(odds, Mapping):
        d = {k: v for k, v in zip(odds, p_gen)}
        if full_output:
            return ShinOptimisationDetails(
                implied_probabilities=d,
                iterations=iterations,
                delta=delta,
                z=z,
            )
        return d

    l = list(p_gen)
    if full_output:
        return ShinOptimisationDetails(
            implied_probabilities=l,
            iterations=iterations,
            delta=delta,
            z=z,
        )
    return l


# sequence input
@overload
def calculate_implied_odds(
    probabilities: Sequence[float],
    *,
    margin: float = ...,
) -> list[float]:
    ...


# mapping input
@overload
def calculate_implied_odds(
    probabilities: Mapping[T, float],
    *,
    margin: float = ...,
) -> dict[T, float]:
    ...


def calculate_implied_odds(
    probabilities: Sequence[float] | Mapping[T, float], *, margin: float = 0.0
) -> list[float] | Mapping[T, float]:
    prob_seq = (
        list(probabilities.values())
        if isinstance(probabilities, Mapping)
        else probabilities
    )

    if not margin:
        z = 0.0
    else:

        def solver_func(z_in: float) -> float:
            return sum(_inverse_odds_for_z(z_in, prob_seq)) - (1 + margin)

        z = _solve_for_root(solver_func)

    price_gen = (1 / p for p in _inverse_odds_for_z(z, prob_seq))
    if isinstance(probabilities, Mapping):
        return {k: p for k, p in zip(probabilities, price_gen)}
    return list(price_gen)


def _inverse_odds_for_z(z: float, prob_seq: Sequence[float]) -> list[float]:
    """Calculate the inverse odds for a given z.

    Args:
        z: The value to solve for.
        prob_seq: The probabilities should sum to 0.
    """
    y = [sqrt((z * p) + ((1 - z) * p**2)) for p in prob_seq]
    sum_y = sum(y)
    return [yy * sum_y for yy in y]


def _solve_for_root(
    func: Callable[[float], float],
    a: float = 0,
    b: float = 3,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> float:
    """Solve for the root of a function using the bisection method.

    Args:
        func: The function to find the root of.
        a: The lower bound of the interval to search for the root in.
        b: The upper bound of the interval to search for the root in.
        tolerance: The tolerance of the root approximation.
        max_iterations: The maximum number of iterations to perform.

    Returns:
        The root approximation.

    Raises:
        ValueError: If the function does not change sign over the interval.
    """
    if func(a) * func(b) >= 0:
        raise ValueError(
            "Function does not change sign over the interval. Choose different a and b."
        )

    c = a
    for _ in range(max_iterations):
        c = (a + b) / 2
        if func(c) == 0 or (b - a) / 2 < tolerance:
            return c
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
    return c  # Return the last midpoint as the root approximation
