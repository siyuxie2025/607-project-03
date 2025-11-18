"""
Profiling Guide for Risk-Aware Bandit Simulation Study
"""

import time
import cProfile
import pstats
import io
from memory_profiler import profile
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple
import warnings
import logging
from pathlib import Path

# Setup logging for numerical issues
logging.basicConfig(
    filename='simulation_warnings.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class NumericalIssueTracker:
    """Track numerical warnings and convergence issues"""
    def __init__(self):
        self.warnings = []
        self.convergence_failures = []
        self.exceptions = []
    
    def log_warning(self, message: str, context: Dict):
        """Log a numerical warning with context"""
        entry = {'message': message, 'context': context, 'timestamp': time.time()}
        self.warnings.append(entry)
        logging.warning(f"{message} | Context: {context}")
    
    def log_convergence_failure(self, method: str, context: Dict):
        """Log convergence failure"""
        entry = {'method': method, 'context': context, 'timestamp': time.time()}
        self.convergence_failures.append(entry)
        logging.error(f"Convergence failure in {method} | Context: {context}")
    
    def log_exception(self, exception: Exception, context: Dict):
        """Log exception"""
        entry = {'exception': str(exception), 'type': type(exception).__name__, 
                 'context': context, 'timestamp': time.time()}
        self.exceptions.append(entry)
        logging.error(f"Exception: {exception} | Context: {context}")
    
    def summary(self) -> Dict:
        """Return summary of all tracked issues"""
        return {
            'total_warnings': len(self.warnings),
            'total_convergence_failures': len(self.convergence_failures),
            'total_exceptions': len(self.exceptions),
            'warnings': self.warnings,
            'convergence_failures': self.convergence_failures,
            'exceptions': self.exceptions
        }

# Global tracker instance
issue_tracker = NumericalIssueTracker()


# ====================
# 1. RUNTIME PROFILING
# ====================

def profile_with_cprofile(func: Callable, *args, **kwargs):
    """
    Profile a function using cProfile
    
    Usage:
        profile_with_cprofile(run_simulation, n_rounds=1000, n_arms=5, df=2.0)
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    
    # Print stats to console
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    print("=" * 80)
    print("PROFILING RESULTS (sorted by cumulative time)")
    print("=" * 80)
    print(s.getvalue())
    
    # Save to file
    stats.dump_stats('simulation_profile.prof')
    print("\nProfile saved to: simulation_profile.prof")
    print("View with: python -m pstats simulation_profile.prof")
    
    return result


def simple_timer(func: Callable, *args, **kwargs) -> Tuple[float, any]:
    """Simple timer wrapper that returns elapsed time and result"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


# Example function to profile (replace with your actual simulation)
def run_single_simulation(n_rounds: int, n_arms: int, df: float, 
                         method: str = "quantile", tracker=None):
    """
    Placeholder for your actual simulation function.
    Replace this with your actual bandit simulation code.
    """
    if tracker is None:
        tracker = issue_tracker
    
    # Simulate your bandit algorithm
    results = []
    
    # Catch numerical warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            for round_idx in range(n_rounds):
                # Your simulation logic here
                # For example: quantile regression, OLS update, etc.
                
                # Simulate heavy-tailed errors
                if df < 2:
                    # Very heavy tails - more likely to have numerical issues
                    errors = np.random.standard_t(df, size=n_arms)
                    if np.any(np.abs(errors) > 1e10):
                        tracker.log_warning(
                            "Extremely large errors detected",
                            {'round': round_idx, 'df': df, 'max_error': np.max(np.abs(errors))}
                        )
                
                # Check for NaN or Inf
                if np.any(np.isnan(errors)) or np.any(np.isinf(errors)):
                    tracker.log_warning(
                        "NaN or Inf detected in errors",
                        {'round': round_idx, 'df': df}
                    )
                
                results.append(errors)
            
            # Log any warnings that occurred
            if len(w) > 0:
                for warning in w:
                    tracker.log_warning(
                        str(warning.message),
                        {'category': warning.category.__name__, 'n_rounds': n_rounds, 'df': df}
                    )
        
        except Exception as e:
            tracker.log_exception(e, {'n_rounds': n_rounds, 'n_arms': n_arms, 'df': df})
            raise
    
    return np.array(results)


# ======================================
# 2. COMPUTATIONAL COMPLEXITY ANALYSIS
# ======================================

def empirical_complexity_analysis(
    func: Callable,
    param_name: str,
    param_values: List,
    fixed_kwargs: Dict = None,
    n_repeats: int = 5
) -> Dict:
    """
    Analyze empirical computational complexity
    
    Args:
        func: Function to profile (e.g., run_simulation)
        param_name: Name of parameter to vary (e.g., 'n_rounds', 'n_arms')
        param_values: List of parameter values to test
        fixed_kwargs: Other fixed parameters
        n_repeats: Number of times to repeat each measurement
    
    Returns:
        Dictionary with timing results and complexity estimates
    """
    if fixed_kwargs is None:
        fixed_kwargs = {}
    
    timings = []
    std_devs = []
    
    print(f"\n{'='*80}")
    print(f"EMPIRICAL COMPLEXITY ANALYSIS: varying {param_name}")
    print(f"{'='*80}\n")
    
    for value in param_values:
        kwargs = fixed_kwargs.copy()
        kwargs[param_name] = value
        
        run_times = []
        for repeat in range(n_repeats):
            elapsed, _ = simple_timer(func, **kwargs)
            run_times.append(elapsed)
        
        mean_time = np.mean(run_times)
        std_time = np.std(run_times)
        timings.append(mean_time)
        std_devs.append(std_time)
        
        print(f"{param_name}={value:>6}: {mean_time:>8.4f}s ± {std_time:.4f}s")
    
    # Fit power law: time = a * n^b
    log_params = np.log(param_values)
    log_times = np.log(timings)
    
    # Linear fit on log-log scale
    coeffs = np.polyfit(log_params, log_times, 1)
    b = coeffs[0]  # Exponent
    a = np.exp(coeffs[1])  # Coefficient
    
    print(f"\n{'='*80}")
    print(f"COMPLEXITY ESTIMATE")
    print(f"{'='*80}")
    print(f"Time complexity: O({param_name}^{b:.2f})")
    print(f"Fitted model: time ≈ {a:.6f} * {param_name}^{b:.2f}")
    
    # R-squared
    predicted = a * np.array(param_values) ** b
    ss_res = np.sum((np.array(timings) - predicted) ** 2)
    ss_tot = np.sum((np.array(timings) - np.mean(timings)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R² = {r_squared:.4f}")
    
    return {
        'param_name': param_name,
        'param_values': param_values,
        'timings': timings,
        'std_devs': std_devs,
        'complexity_exponent': b,
        'coefficient': a,
        'r_squared': r_squared
    }


def plot_complexity(results: Dict, save_path: str = 'complexity_analysis.pdf'):
    """Plot empirical complexity analysis results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    param_values = results['param_values']
    timings = results['timings']
    std_devs = results['std_devs']
    
    # Linear scale plot
    ax1.errorbar(param_values, timings, yerr=std_devs, 
                 marker='o', capsize=5, label='Measured')
    predicted = results['coefficient'] * np.array(param_values) ** results['complexity_exponent']
    ax1.plot(param_values, predicted, 'r--', 
             label=f"O(n^{results['complexity_exponent']:.2f})")
    ax1.set_xlabel(results['param_name'])
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Runtime vs Parameter (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log-log scale plot
    ax2.errorbar(param_values, timings, yerr=std_devs, 
                 marker='o', capsize=5, label='Measured')
    ax2.plot(param_values, predicted, 'r--',
             label=f"Slope = {results['complexity_exponent']:.2f}")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(f"{results['param_name']} (log scale)")
    ax2.set_ylabel('Time (log scale)')
    ax2.set_title('Log-Log Plot for Complexity Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


# ===============================
# 3. MEMORY PROFILING
# ===============================

# For memory profiling, use the @profile decorator
# Run with: python -m memory_profiler your_script.py

def memory_profile_example():
    """
    Example showing memory profiling with line-by-line tracking.
    
    To use:
    1. Install: pip install memory_profiler
    2. Add @profile decorator to functions you want to profile
    3. Run: python -m memory_profiler your_script.py
    """
    pass


# ===============================
# 4. EXAMPLE USAGE
# ===============================

def main():
    """
    Main function demonstrating profiling workflow
    """
    print("\n" + "="*80)
    print("BANDIT SIMULATION PROFILING SUITE")
    print("="*80 + "\n")
    
    # Example 1: Profile a single run with cProfile
    print("\n[1] Profiling single simulation run with cProfile...")
    print("-" * 80)
    # profile_with_cprofile(run_single_simulation, 
    #                       n_rounds=1000, n_arms=5, df=2.0)
    
    # Example 2: Analyze complexity by varying n_rounds
    print("\n[2] Analyzing complexity: varying n_rounds...")
    print("-" * 80)
    n_rounds_values = [100, 200, 500, 1000, 2000]
    results_rounds = empirical_complexity_analysis(
        func=run_single_simulation,
        param_name='n_rounds',
        param_values=n_rounds_values,
        fixed_kwargs={'n_arms': 5, 'df': 2.0, 'method': 'quantile'},
        n_repeats=3
    )
    plot_complexity(results_rounds, 'complexity_n_rounds.pdf')
    
    # Example 3: Analyze complexity by varying n_arms
    print("\n[3] Analyzing complexity: varying n_arms...")
    print("-" * 80)
    n_arms_values = [2, 5, 10, 20, 50]
    results_arms = empirical_complexity_analysis(
        func=run_single_simulation,
        param_name='n_arms',
        param_values=n_arms_values,
        fixed_kwargs={'n_rounds': 1000, 'df': 2.0, 'method': 'quantile'},
        n_repeats=3
    )
    plot_complexity(results_arms, 'complexity_n_arms.pdf')
    
    # Example 4: Compare different df values (heavy-tailed distributions)
    print("\n[4] Testing numerical stability across df values...")
    print("-" * 80)
    df_values = [1.5, 2.0, 3.0, 5.0, 10.0]
    for df in df_values:
        print(f"\nTesting df={df}...")
        _ = run_single_simulation(n_rounds=500, n_arms=5, df=df)
    
    # Print numerical issue summary
    print("\n" + "="*80)
    print("NUMERICAL STABILITY REPORT")
    print("="*80)
    summary = issue_tracker.summary()
    print(f"Total warnings: {summary['total_warnings']}")
    print(f"Convergence failures: {summary['total_convergence_failures']}")
    print(f"Exceptions: {summary['total_exceptions']}")
    
    if summary['total_warnings'] > 0:
        print("\nWarning details:")
        for i, warn in enumerate(summary['warnings'][:5], 1):  # Show first 5
            print(f"  {i}. {warn['message']}")
            print(f"     Context: {warn['context']}")
    
    print(f"\nFull log saved to: simulation_warnings.log")
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - simulation_profile.prof (cProfile output)")
    print("  - complexity_n_rounds.pdf (complexity plot)")
    print("  - complexity_n_arms.pdf (complexity plot)")
    print("  - simulation_warnings.log (numerical issues)")
    

if __name__ == "__main__":
    main()


"""
INSTRUCTIONS FOR YOUR PROJECT
==============================

1. RUNTIME PROFILING:
   - Replace `run_single_simulation()` with your actual simulation function
   - Run: python profiling_script.py
   - View detailed profile: python -m pstats simulation_profile.prof
     Then type: sort cumtime, stats 20

2. COMPLEXITY ANALYSIS:
   - Identify key parameters (n_rounds, n_arms, sample_size, etc.)
   - Run empirical_complexity_analysis() for each parameter
   - Generate log-log plots to visualize O(n) behavior
   - Report: "Our algorithm has O(n^b) complexity where b ≈ X.XX"

3. THEORETICAL COMPLEXITY:
   - Analyze your code structure:
     * Quantile regression: typically O(n log n) per iteration
     * OLS: O(n * p^2) where p is number of features
     * For loops over rounds: multiply by n_rounds
   - Example: n_rounds * (n_arms * O(n log n)) = O(n_rounds * n_arms * n log n)

4. NUMERICAL STABILITY:
   - Check simulation_warnings.log for issues
   - Look for patterns with specific df values (heavy tails)
   - Report any convergence failures in quantile regression
   - Monitor for NaN, Inf, or extremely large values

5. MEMORY PROFILING (optional):
   - Install: pip install memory_profiler
   - Add @profile decorator to key functions
   - Run: python -m memory_profiler your_script.py
   - Look for memory leaks in long simulations

6. INTEGRATION WITH YOUR CODE:
   - Import this module: from profiling_script import *
   - Wrap your simulation runs with profile_with_cprofile()
   - Add issue_tracker to your simulation functions
   - Run complexity analysis for your README or report
"""