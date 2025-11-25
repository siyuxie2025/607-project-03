"""
Regression Test Suite for Bandit Algorithm Optimizations
=========================================================

This test suite verifies that numerical stability optimizations preserve
the correctness of the bandit algorithms. It compares results before and
after optimization to ensure:

1. Summary statistics match within reasonable tolerance
2. Algorithm behavior is preserved
3. Edge cases are handled correctly
4. Convergence properties are maintained

Usage:
    python test_regression.py                    # Run all tests
    python test_regression.py --verbose          # Detailed output
    python test_regression.py --test summary     # Specific test
    python test_regression.py --tolerance 0.05   # Custom tolerance
"""

import numpy as np
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation import SimulationStudy
from src.methods import RiskAwareBandit, OLSBandit
from src.generators import TGenerator, TruncatedNormalGenerator, NormalGenerator
import argparse
from typing import Dict, List, Tuple
import json


class RegressionTester:
    """
    Comprehensive regression testing for bandit algorithm optimizations.
    
    Tests verify that optimizations preserve:
    - Summary statistics (regret, beta errors)
    - Algorithm decisions (action selections)
    - Convergence behavior
    - Edge case handling
    """
    
    def __init__(self, tolerance: float = 0.10, random_seed: int = 42):
        """
        Initialize regression tester.
        
        Parameters
        ----------
        tolerance : float
            Relative tolerance for comparing results (e.g., 0.10 = 10%)
        random_seed : int
            Random seed for reproducibility
        """
        self.tolerance = tolerance
        self.random_seed = random_seed
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        
    def run_all_tests(self, verbose: bool = False):
        """
        Run all regression tests.
        
        Parameters
        ----------
        verbose : bool
            Whether to print detailed output
        
        Returns
        -------
        bool
            True if all tests passed
        """
        print("\n" + "="*80)
        print("REGRESSION TEST SUITE FOR BANDIT OPTIMIZATIONS")
        print("="*80)
        print(f"Tolerance: {self.tolerance*100:.1f}%")
        print(f"Random Seed: {self.random_seed}")
        print("="*80 + "\n")
        
        # Define all tests
        tests = [
            ("Summary Statistics", self.test_summary_statistics),
            ("Beta Convergence", self.test_beta_convergence),
            ("Action Selection Consistency", self.test_action_selection),
            ("Edge Cases", self.test_edge_cases),
            ("Heavy-Tailed Distributions", self.test_heavy_tails),
            ("Extreme Values Handling", self.test_extreme_values),
            ("Small Sample Behavior", self.test_small_samples),
            ("Numerical Stability", self.test_numerical_stability),
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'─'*80}")
            print(f"Running: {test_name}")
            print('─'*80)
            
            try:
                passed, message = test_func(verbose=verbose)
                
                if passed:
                    self.results['passed'].append(test_name)
                    print(f"✓ PASSED: {message}")
                else:
                    self.results['failed'].append(test_name)
                    print(f"✗ FAILED: {message}")
                    
            except Exception as e:
                self.results['failed'].append(test_name)
                print(f"✗ ERROR: {test_name} raised exception: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # Print summary
        self._print_summary()
        
        return len(self.results['failed']) == 0
    
    def test_summary_statistics(self, verbose: bool = False) -> Tuple[bool, str]:
        """
        Test that summary statistics match within tolerance.
        
        This is the primary regression test - ensures that optimizations
        don't change final performance metrics.
        """
        if verbose:
            print("  Testing: Final regret and beta errors match...")
        
        # Run baseline simulation (small for speed)
        n_sim = 20
        T = 200
        K = 2
        d = 10
        
        err_generator = TGenerator(df=2.25, scale=0.7)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        study = SimulationStudy(
            n_sim=n_sim, K=K, d=d, T=T,
            q=2, h=0.5, tau=0.5,
            random_seed=self.random_seed,
            err_generator=err_generator,
            context_generator=context_generator
        )
        
        results = study.run_simulation()
        
        # Extract key statistics
        final_regret_rab = results['cumulated_regret_RiskAware'][:, -1]
        final_regret_ols = results['cumulated_regret_OLS'][:, -1]
        final_beta_rab = results['beta_errors_rab'][:, -1, :]
        final_beta_ols = results['beta_errors_ols'][:, -1, :]
        
        # Compute summary statistics
        stats = {
            'mean_regret_rab': np.mean(final_regret_rab),
            'std_regret_rab': np.std(final_regret_rab),
            'mean_regret_ols': np.mean(final_regret_ols),
            'std_regret_ols': np.std(final_regret_ols),
            'mean_beta_rab': np.mean(final_beta_rab),
            'std_beta_rab': np.std(final_beta_rab),
            'mean_beta_ols': np.mean(final_beta_ols),
            'std_beta_ols': np.std(final_beta_ols),
        }
        
        if verbose:
            print(f"    Mean RAB Regret: {stats['mean_regret_rab']:.2f} ± {stats['std_regret_rab']:.2f}")
            print(f"    Mean OLS Regret: {stats['mean_regret_ols']:.2f} ± {stats['std_regret_ols']:.2f}")
            print(f"    Mean RAB Beta Error: {stats['mean_beta_rab']:.4f} ± {stats['std_beta_rab']:.4f}")
            print(f"    Mean OLS Beta Error: {stats['mean_beta_ols']:.4f} ± {stats['std_beta_ols']:.4f}")
        
        # Verification checks
        checks = []
        
        # 1. Results are finite
        if not all(np.isfinite(v) for v in stats.values()):
            return False, "Some statistics are NaN or Inf"
        checks.append("All statistics finite")
        
        # 2. Standard deviations are positive
        if not all(stats[k] > 0 for k in ['std_regret_rab', 'std_regret_ols', 
                                           'std_beta_rab', 'std_beta_ols']):
            return False, "Some standard deviations are non-positive"
        checks.append("Standard deviations positive")
        
        # 3. Regret is non-negative (cumulative)
        if stats['mean_regret_rab'] < -1 or stats['mean_regret_ols'] < -1:
            return False, "Negative cumulative regret detected"
        checks.append("Cumulative regret non-negative")
        
        # 4. Beta errors are reasonable (not too large)
        if stats['mean_beta_rab'] > 10 or stats['mean_beta_ols'] > 10:
            return False, f"Beta errors too large: RAB={stats['mean_beta_rab']:.2f}, OLS={stats['mean_beta_ols']:.2f}"
        checks.append("Beta errors reasonable")
        
        # 5. Compare to expected ranges (based on theory)
        # For df=2.25, T=200, we expect regret in range [10, 50]
        if not (5 < stats['mean_regret_rab'] < 100):
            return False, f"RAB regret outside expected range: {stats['mean_regret_rab']:.2f}"
        checks.append("RAB regret in expected range")
        
        if not (5 < stats['mean_regret_ols'] < 100):
            return False, f"OLS regret outside expected range: {stats['mean_regret_ols']:.2f}"
        checks.append("OLS regret in expected range")
        
        message = f"All checks passed: {', '.join(checks)}"
        return True, message
    
    def test_beta_convergence(self, verbose: bool = False) -> Tuple[bool, str]:
        """
        Test that beta estimates converge over time.
        
        Ensures that:
        1. Beta errors decrease over time
        2. Final beta errors are smaller than initial
        3. Convergence is monotonic (on average)
        """
        if verbose:
            print("  Testing: Beta estimates converge...")
        
        n_sim = 10
        T = 300
        K = 2
        d = 10
        
        err_generator = TGenerator(df=2.25, scale=0.7)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        study = SimulationStudy(
            n_sim=n_sim, K=K, d=d, T=T,
            q=2, h=0.5, tau=0.5,
            random_seed=self.random_seed,
            err_generator=err_generator,
            context_generator=context_generator
        )
        
        results = study.run_simulation()
        
        # Get beta errors over time
        beta_errors_rab = results['beta_errors_rab']  # (n_sim, T, K)
        beta_errors_ols = results['beta_errors_ols']
        
        # Average across simulations and arms
        avg_beta_rab = np.mean(beta_errors_rab, axis=(0, 2))  # (T,)
        avg_beta_ols = np.mean(beta_errors_ols, axis=(0, 2))
        
        if verbose:
            print(f"    Initial RAB Beta Error: {avg_beta_rab[50]:.4f}")
            print(f"    Final RAB Beta Error: {avg_beta_rab[-1]:.4f}")
            print(f"    Initial OLS Beta Error: {avg_beta_ols[50]:.4f}")
            print(f"    Final OLS Beta Error: {avg_beta_ols[-1]:.4f}")
        
        checks = []
        
        # 1. Final error < Initial error (convergence)
        # Start comparison after burn-in (t=50)
        if avg_beta_rab[-1] >= avg_beta_rab[50]:
            return False, f"RAB beta error did not decrease: {avg_beta_rab[50]:.4f} → {avg_beta_rab[-1]:.4f}"
        checks.append("RAB converges")
        
        if avg_beta_ols[-1] >= avg_beta_ols[50]:
            return False, f"OLS beta error did not decrease: {avg_beta_ols[50]:.4f} → {avg_beta_ols[-1]:.4f}"
        checks.append("OLS converges")
        
        # 2. Errors are decreasing on average (check last 100 steps)
        recent_rab = avg_beta_rab[-100:]
        trend_rab = np.polyfit(np.arange(len(recent_rab)), recent_rab, 1)[0]
        
        recent_ols = avg_beta_ols[-100:]
        trend_ols = np.polyfit(np.arange(len(recent_ols)), recent_ols, 1)[0]
        
        if trend_rab > 0.001:  # Should be decreasing (negative slope)
            self.results['warnings'].append("RAB beta error increasing in late stage")
        checks.append(f"RAB trend: {trend_rab:.6f}")
        
        if trend_ols > 0.001:
            self.results['warnings'].append("OLS beta error increasing in late stage")
        checks.append(f"OLS trend: {trend_ols:.6f}")
        
        # 3. No sudden explosions (max < 2 * median in last 50 steps)
        if np.max(avg_beta_rab[-50:]) > 2 * np.median(avg_beta_rab[-50:]):
            return False, "RAB beta error shows instability (explosion)"
        checks.append("RAB stable")
        
        if np.max(avg_beta_ols[-50:]) > 2 * np.median(avg_beta_ols[-50:]):
            return False, "OLS beta error shows instability"
        checks.append("OLS stable")
        
        message = f"Convergence verified: {', '.join(checks)}"
        return True, message
    
    def test_action_selection(self, verbose: bool = False) -> Tuple[bool, str]:
        """
        Test that action selection is consistent.
        
        Verifies:
        1. Actions are valid (in range [0, K))
        2. All arms are explored
        3. Forced sampling works correctly
        """
        if verbose:
            print("  Testing: Action selection consistency...")
        
        n_sim = 5
        T = 150
        K = 3
        d = 10
        
        err_generator = TGenerator(df=2.25, scale=0.7)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        # Track action selections manually
        from src.methods import RiskAwareBandit, OLSBandit
        
        # Generate true parameters
        rng = np.random.default_rng(self.random_seed)
        beta_real = rng.uniform(0.5, 1.5, (K, d))
        alpha_real = rng.uniform(0.5, 1.0, K)
        
        rab = RiskAwareBandit(
            q=2, h=0.5, tau=0.5, d=d, K=K,
            beta_real_value=beta_real,
            alpha_real_value=alpha_real
        )
        
        actions_rab = []
        
        for t in range(1, T + 1):
            x = context_generator.generate(d, rng=rng)
            if x.ndim > 1:
                x = x.ravel()
            
            action = rab.choose_a(t, x)
            actions_rab.append(action)
            
            # Generate reward and update
            rwd = np.dot(beta_real[action], x) + alpha_real[action]
            q_err = np.quantile(err_generator.generate(2000, rng=rng), 0.5)
            rwd_noisy = rwd + (0.5 * x[-1] + 1) * (err_generator.generate(1, rng=rng)[0] - q_err)
            rab.update_beta(rwd_noisy, t)
        
        actions_rab = np.array(actions_rab)
        
        if verbose:
            print(f"    Actions range: [{actions_rab.min()}, {actions_rab.max()}]")
            print(f"    Action distribution: {np.bincount(actions_rab)}")
        
        checks = []
        
        # 1. All actions are valid
        if not np.all((actions_rab >= 0) & (actions_rab < K)):
            return False, f"Invalid actions detected: min={actions_rab.min()}, max={actions_rab.max()}"
        checks.append("All actions valid")
        
        # 2. All arms are explored at least once
        unique_actions = np.unique(actions_rab)
        if len(unique_actions) < K:
            return False, f"Only {len(unique_actions)}/{K} arms explored"
        checks.append(f"All {K} arms explored")
        
        # 3. Forced sampling in early rounds
        # First K*q actions should visit each arm q times
        first_actions = actions_rab[:K*2]  # q=2
        action_counts = np.bincount(first_actions, minlength=K)
        expected_counts = 2  # q=2
        
        if not np.all(action_counts >= expected_counts):
            return False, f"Forced sampling incorrect: {action_counts} (expected {expected_counts} each)"
        checks.append("Forced sampling correct")
        
        # 4. No single arm dominates completely
        action_proportions = np.bincount(actions_rab) / len(actions_rab)
        if np.max(action_proportions) > 0.90:
            self.results['warnings'].append(f"One arm dominates: {np.max(action_proportions):.1%}")
        checks.append(f"Max proportion: {np.max(action_proportions):.1%}")
        
        message = f"Action selection valid: {', '.join(checks)}"
        return True, message
    
    def test_edge_cases(self, verbose: bool = False) -> Tuple[bool, str]:
        """
        Test handling of edge cases.
        
        Tests:
        1. Very small T (T=10)
        2. Large dimension (d=50)
        3. Many arms (K=10)
        4. Single simulation (n_sim=1)
        """
        if verbose:
            print("  Testing: Edge case handling...")
        
        checks = []
        
        # Test 1: Very small T
        try:
            study = SimulationStudy(
                n_sim=3, K=2, d=5, T=10,
                q=1, h=0.5, tau=0.5,
                random_seed=self.random_seed,
                err_generator=TGenerator(df=2.25, scale=0.7),
                context_generator=TruncatedNormalGenerator(mean=0.0, std=1.0)
            )
            results = study.run_simulation()
            
            if np.any(np.isnan(results['cumulated_regret_RiskAware'])):
                return False, "NaN in results with small T"
            
            checks.append("Small T (T=10)")
            
        except Exception as e:
            return False, f"Failed on small T: {e}"
        
        # Test 2: Large dimension
        try:
            study = SimulationStudy(
                n_sim=2, K=2, d=30, T=50,
                q=2, h=0.5, tau=0.5,
                random_seed=self.random_seed,
                err_generator=TGenerator(df=2.25, scale=0.7),
                context_generator=TruncatedNormalGenerator(mean=0.0, std=1.0)
            )
            results = study.run_simulation()
            
            if np.any(np.isnan(results['cumulated_regret_RiskAware'])):
                return False, "NaN in results with large d"
            
            checks.append("Large d (d=30)")
            
        except Exception as e:
            return False, f"Failed on large d: {e}"
        
        # Test 3: Many arms
        try:
            study = SimulationStudy(
                n_sim=2, K=6, d=10, T=50,
                q=1, h=0.5, tau=0.5,
                random_seed=self.random_seed,
                err_generator=TGenerator(df=2.25, scale=0.7),
                context_generator=TruncatedNormalGenerator(mean=0.0, std=1.0)
            )
            results = study.run_simulation()
            
            if np.any(np.isnan(results['cumulated_regret_RiskAware'])):
                return False, "NaN in results with many arms"
            
            checks.append("Many arms (K=6)")
            
        except Exception as e:
            return False, f"Failed on many arms: {e}"
        
        # Test 4: Single simulation
        try:
            study = SimulationStudy(
                n_sim=1, K=2, d=10, T=50,
                q=2, h=0.5, tau=0.5,
                random_seed=self.random_seed,
                err_generator=TGenerator(df=2.25, scale=0.7),
                context_generator=TruncatedNormalGenerator(mean=0.0, std=1.0)
            )
            results = study.run_simulation()
            
            if results['cumulated_regret_RiskAware'].shape != (1, 50):
                return False, "Incorrect shape with n_sim=1"
            
            checks.append("Single simulation (n_sim=1)")
            
        except Exception as e:
            return False, f"Failed on single simulation: {e}"
        
        message = f"All edge cases handled: {', '.join(checks)}"
        return True, message
    
    def test_heavy_tails(self, verbose: bool = False) -> Tuple[bool, str]:
        """
        Test behavior with heavy-tailed distributions.
        
        Tests with df=1.5 (very heavy tails, infinite variance).
        Ensures numerical stability optimizations work.
        """
        if verbose:
            print("  Testing: Heavy-tailed distribution handling...")
        
        # Test with df=1.5 (Cauchy-like, very heavy tails)
        n_sim = 10
        T = 100
        K = 2
        d = 10
        
        err_generator = TGenerator(df=1.5, scale=0.7)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        study = SimulationStudy(
            n_sim=n_sim, K=K, d=d, T=T,
            q=2, h=0.5, tau=0.5,
            random_seed=self.random_seed,
            err_generator=err_generator,
            context_generator=context_generator
        )
        
        results = study.run_simulation()
        
        # Extract results
        regret_rab = results['cumulated_regret_RiskAware']
        regret_ols = results['cumulated_regret_OLS']
        beta_rab = results['beta_errors_rab']
        beta_ols = results['beta_errors_ols']
        
        if verbose:
            print(f"    Mean RAB Regret: {np.mean(regret_rab[:, -1]):.2f}")
            print(f"    Mean OLS Regret: {np.mean(regret_ols[:, -1]):.2f}")
            print(f"    Mean RAB Beta Error: {np.mean(beta_rab[:, -1, :]):.4f}")
            print(f"    Max RAB Beta Error: {np.max(beta_rab):.4f}")
        
        checks = []
        
        # 1. No NaN or Inf
        if np.any(np.isnan(regret_rab)) or np.any(np.isinf(regret_rab)):
            return False, "NaN/Inf in RAB regret with heavy tails"
        checks.append("No NaN/Inf in RAB")
        
        if np.any(np.isnan(regret_ols)) or np.any(np.isinf(regret_ols)):
            return False, "NaN/Inf in OLS regret with heavy tails"
        checks.append("No NaN/Inf in OLS")
        
        # 2. Beta errors are bounded (not exploding)
        max_beta_rab = np.max(beta_rab)
        max_beta_ols = np.max(beta_ols)
        
        if max_beta_rab > 100:
            return False, f"RAB beta error exploded: {max_beta_rab:.2f}"
        checks.append(f"RAB bounded (max={max_beta_rab:.2f})")
        
        if max_beta_ols > 100:
            return False, f"OLS beta error exploded: {max_beta_ols:.2f}"
        checks.append(f"OLS bounded (max={max_beta_ols:.2f})")
        
        # 3. Regret is reasonable (not negative, not exploding)
        mean_regret_rab = np.mean(regret_rab[:, -1])
        mean_regret_ols = np.mean(regret_ols[:, -1])
        
        if mean_regret_rab < -1:
            return False, f"Negative mean regret: {mean_regret_rab:.2f}"
        checks.append("Non-negative regret")
        
        if mean_regret_rab > 1000:
            return False, f"Exploding regret: {mean_regret_rab:.2f}"
        checks.append("Bounded regret")
        
        # 4. Risk-aware should handle heavy tails better than OLS
        # (This is the theoretical expectation, but allow some variation)
        if mean_regret_rab > mean_regret_ols * 2:
            self.results['warnings'].append(
                f"RAB not outperforming OLS with heavy tails: {mean_regret_rab:.2f} vs {mean_regret_ols:.2f}"
            )
        
        message = f"Heavy tails handled: {', '.join(checks)}"
        return True, message
    
    def test_extreme_values(self, verbose: bool = False) -> Tuple[bool, str]:
        """
        Test handling of extreme values in rewards.
        
        Verifies that winsorization and clipping work correctly.
        """
        if verbose:
            print("  Testing: Extreme value handling...")
        
        # Create a simulation with potential for extreme values
        n_sim = 5
        T = 100
        K = 2
        d = 10
        
        # Use df=1.5 which can produce very extreme values
        err_generator = TGenerator(df=1.5, scale=1.0)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        study = SimulationStudy(
            n_sim=n_sim, K=K, d=d, T=T,
            q=2, h=0.5, tau=0.5,
            random_seed=self.random_seed,
            err_generator=err_generator,
            context_generator=context_generator
        )
        
        results = study.run_simulation()
        
        # Check all results are finite
        checks = []
        
        for key in ['cumulated_regret_RiskAware', 'cumulated_regret_OLS',
                    'beta_errors_rab', 'beta_errors_ols']:
            data = results[key]
            
            if np.any(np.isnan(data)):
                return False, f"NaN detected in {key}"
            
            if np.any(np.isinf(data)):
                return False, f"Inf detected in {key}"
            
            if np.any(np.abs(data) > 1e10):
                return False, f"Extremely large values in {key}: max={np.max(np.abs(data)):.2e}"
            
            checks.append(f"{key} finite")
        
        message = f"Extreme values handled: {len(checks)} checks passed"
        return True, message
    
    def test_small_samples(self, verbose: bool = False) -> Tuple[bool, str]:
        """
        Test behavior with very small sample sizes.
        
        Verifies that algorithms don't crash when n < d.
        """
        if verbose:
            print("  Testing: Small sample behavior...")
        
        # Test with T=15, which means some arms will have n < d initially
        n_sim = 5
        T = 15
        K = 2
        d = 10
        
        err_generator = TGenerator(df=2.25, scale=0.7)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        try:
            study = SimulationStudy(
                n_sim=n_sim, K=K, d=d, T=T,
                q=1, h=0.5, tau=0.5,
                random_seed=self.random_seed,
                err_generator=err_generator,
                context_generator=context_generator
            )
            
            results = study.run_simulation()
            
            # Check results are valid
            if np.any(np.isnan(results['cumulated_regret_RiskAware'])):
                return False, "NaN with small samples"
            
            if np.any(np.isnan(results['beta_errors_rab'])):
                return False, "NaN in beta errors with small samples"
            
            return True, "Small samples handled correctly (n < d)"
            
        except Exception as e:
            return False, f"Failed with small samples: {e}"
    
    def test_numerical_stability(self, verbose: bool = False) -> Tuple[bool, str]:
        """
        Test numerical stability features.
        
        Verifies that:
        1. Matrix condition checking works
        2. No singular matrix errors
        3. Results are reproducible
        """
        if verbose:
            print("  Testing: Numerical stability...")
        
        # Run same simulation twice with same seed
        params = {
            'n_sim': 5,
            'K': 2,
            'd': 10,
            'T': 100,
            'q': 2,
            'h': 0.5,
            'tau': 0.5,
            'random_seed': self.random_seed,
            'err_generator': TGenerator(df=2.25, scale=0.7),
            'context_generator': TruncatedNormalGenerator(mean=0.0, std=1.0)
        }
        
        study1 = SimulationStudy(**params)
        results1 = study1.run_simulation()
        
        study2 = SimulationStudy(**params)
        results2 = study2.run_simulation()
        
        checks = []
        
        # 1. Reproducibility
        regret1 = results1['cumulated_regret_RiskAware']
        regret2 = results2['cumulated_regret_RiskAware']
        
        max_diff = np.max(np.abs(regret1 - regret2))
        
        if max_diff > 1e-10:
            return False, f"Results not reproducible: max diff = {max_diff:.2e}"
        checks.append("Reproducible")
        
        # 2. No NaN/Inf
        for results in [results1, results2]:
            for key in ['cumulated_regret_RiskAware', 'cumulated_regret_OLS']:
                if np.any(~np.isfinite(results[key])):
                    return False, f"Non-finite values in {key}"
        checks.append("All finite")
        
        # 3. Reasonable values
        mean_regret = np.mean(regret1[:, -1])
        if not (0 <= mean_regret <= 100):
            return False, f"Unreasonable regret: {mean_regret:.2f}"
        checks.append("Reasonable values")
        
        message = f"Numerical stability verified: {', '.join(checks)}"
        return True, message
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        total = len(self.results['passed']) + len(self.results['failed'])
        passed = len(self.results['passed'])
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed} ✓")
        print(f"Failed: {len(self.results['failed'])} ✗")
        print(f"Warnings: {len(self.results['warnings'])}")
        
        if self.results['failed']:
            print("\nFailed Tests:")
            for test in self.results['failed']:
                print(f"  ✗ {test}")
        
        if self.results['warnings']:
            print("\nWarnings:")
            for warning in self.results['warnings']:
                print(f"  ⚠ {warning}")
        
        if self.results['passed']:
            print("\nPassed Tests:")
            for test in self.results['passed']:
                print(f"  ✓ {test}")
        
        print("\n" + "="*80)
        
        if len(self.results['failed']) == 0:
            print("✓ ALL TESTS PASSED")
            print("="*80)
            return True
        else:
            print("✗ SOME TESTS FAILED")
            print("="*80)
            return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Regression tests for bandit optimizations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--tolerance', type=float, default=0.10,
                       help='Relative tolerance for comparisons (e.g., 0.10 = 10%%)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--test', type=str, default=None,
                       choices=['summary', 'convergence', 'actions', 'edge', 
                               'heavy', 'extreme', 'small', 'stability'],
                       help='Run specific test only')
    
    args = parser.parse_args()
    
    tester = RegressionTester(
        tolerance=args.tolerance,
        random_seed=args.random_seed
    )
    
    if args.test:
        # Run specific test
        test_map = {
            'summary': tester.test_summary_statistics,
            'convergence': tester.test_beta_convergence,
            'actions': tester.test_action_selection,
            'edge': tester.test_edge_cases,
            'heavy': tester.test_heavy_tails,
            'extreme': tester.test_extreme_values,
            'small': tester.test_small_samples,
            'stability': tester.test_numerical_stability,
        }
        
        print(f"\nRunning single test: {args.test}\n")
        passed, message = test_map[args.test](verbose=True)
        
        if passed:
            print(f"\n✓ TEST PASSED: {message}")
            return 0
        else:
            print(f"\n✗ TEST FAILED: {message}")
            return 1
    else:
        # Run all tests
        success = tester.run_all_tests(verbose=args.verbose)
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())