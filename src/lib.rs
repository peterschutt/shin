use argmin::{
    core::{CostFunction, Error, Executor, IterState},
    solver::brent::BrentOpt,
};
use pyo3::prelude::*;

#[pyfunction]
fn optimise(
    inverse_odds: Vec<f64>,
    sum_inverse_odds: f64,
    n: u64,
    max_iterations: u64,
    convergence_threshold: f64,
) -> PyResult<(f64, f64, u64)> {
    let mut delta = f64::INFINITY;
    let mut z: f64 = 0.0;
    let mut iterations = 0;
    let denominator: f64 = (n - 2) as f64;
    while delta > convergence_threshold && iterations < max_iterations {
        let z0 = z;
        z = (inverse_odds
            .iter()
            .map(|io| (z.powi(2) + 4.0 * (1.0 - z) * io.powi(2) / sum_inverse_odds).sqrt())
            .sum::<f64>()
            - 2.0)
            / denominator;
        delta = (z - z0).abs();
        iterations += 1;
    }
    iterations += 1;
    Ok((z, delta, iterations))
}

fn inverse_odds_for_z(z: f64, prob_seq: &[f64]) -> Vec<f64> {
    let y: Vec<f64> = prob_seq
        .iter()
        .map(|&p| f64::sqrt((z * p) + ((1.0 - z) * p * p)))
        .collect();
    let sum_y: f64 = y.iter().sum();
    y.into_iter().map(|yy| yy * sum_y).collect()
}

struct ZSolve<'a> {
    probabilities: &'a [f64],
    overround: f64,
}

impl<'a> CostFunction for ZSolve<'a> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, z: &Self::Param) -> Result<Self::Output, Error> {
        let sum_inverse_odds = inverse_odds_for_z(*z, &self.probabilities)
            .iter()
            .sum::<f64>();
        Ok((sum_inverse_odds - (1.0 + self.overround)).abs())
    }
}

fn solve_for_z(prob_seq: &[f64], overround: f64, max_iters: u64, eps: f64, t: f64) -> IterState<f64, (), (), (), f64> {
    let cost = ZSolve {
        probabilities: prob_seq,
        overround,
    };
    let solver = BrentOpt::new(0., 1.).set_tolerance(eps, t);

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(max_iters).target_cost(0.0))
        .run()
        .unwrap();

    res.state
}

#[pyfunction]
fn calculate_implied_odds(probabilities: Vec<f64>, overround: f64, max_iters: u64, eps: f64, t: f64) -> Vec<f64> {

    let inverse_odds: Vec<f64>;

    if overround == 0.0 {
        inverse_odds = probabilities;
    } else {
        let solver_state = solve_for_z(&probabilities, overround, max_iters, eps, t);
        inverse_odds = inverse_odds_for_z(solver_state.best_param.unwrap(), &probabilities);
    }

    inverse_odds
    .into_iter()
    .map(|io| 1.0 / io)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_solve_for_z() {
        let prob_seq = vec![0.1, 0.2, 0.3, 0.4];
        let overround = 0.1;
        let expected = 0.033725;
        let max_iters = 100;
        let eps = f64::EPSILON.sqrt();
        let t = 1e-8;
        assert_relative_eq!(
            solve_for_z(&prob_seq, overround, max_iters, eps, t).best_param.unwrap(),
            expected,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_inverse_odds_for_z() {
        let z = 0.5;
        let prob_seq = vec![0.1, 0.2, 0.3, 0.4];
        let expected = vec![0.363898, 0.537513, 0.685198, 0.821066];
        let result = inverse_odds_for_z(z, &prob_seq);
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-6);
        }
    }

    fn call_calculate_implied_odds(probabilities: Vec<f64>, overround: f64) -> Vec<f64> {
        calculate_implied_odds(probabilities, overround, 100, f64::EPSILON.sqrt(), 1e-8)
    }

    #[test]
    fn test_calculate_implied_odds_no_overround() {
        let odds = [3., 3., 3.];
        let probabilities: Vec<f64> = odds.iter().map(|&o| 1.0 / o).collect();
        let overround = 0.0;
        let expected = vec![3., 3., 3.];
        let result = call_calculate_implied_odds(probabilities, overround);
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_calculate_implied_odds() {
        let odds = [3., 3., 3.];
        let probabilities: Vec<f64> = odds.iter().map(|&o| 1.0 / o).collect();
        let overround = 0.1;
        let expected = vec![2.72727, 2.72727, 2.72727];
        let result = call_calculate_implied_odds(probabilities, overround);
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-6);
        }
        assert_relative_eq!(result.iter().map(|o| 1. / o).sum::<f64>() - 1., overround, epsilon = 1e-6);
    }
}

/// Fast calculations for Shin's method
#[pymodule]
#[pyo3(name = "shin")]
fn shin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimise, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_implied_odds, m)?)?;
    Ok(())
}
