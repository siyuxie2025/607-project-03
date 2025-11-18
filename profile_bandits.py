# profile_bandits.py
from profiling_analysis import (
    profile_with_cprofile, 
    empirical_complexity_analysis,
    plot_complexity,
    issue_tracker
)
# Import your actual simulation functions
# from your_module import run_simulation, BanditSimulator, etc.

# Profile a single run
def profile_single_run():
    """Profile one complete simulation"""
    profile_with_cprofile(
        your_simulation_function,
        n_rounds=1000,
        n_arms=5,
        df=2.0
    )

# Analyze complexity
def analyze_complexity():
    """Analyze how runtime scales with key parameters"""
    
    # Test 1: How does n_rounds affect runtime?
    print("Analyzing complexity vs n_rounds...")
    results_rounds = empirical_complexity_analysis(
        func=your_simulation_function,
        param_name='n_rounds',
        param_values=[100, 200, 500, 1000, 2000, 5000],
        fixed_kwargs={'n_arms': 5, 'df': 2.0},
        n_repeats=5
    )
    plot_complexity(results_rounds, 'results/complexity_rounds.pdf')
    
    # Test 2: How does n_arms affect runtime?
    print("\nAnalyzing complexity vs n_arms...")
    results_arms = empirical_complexity_analysis(
        func=your_simulation_function,
        param_name='n_arms',
        param_values=[2, 5, 10, 20, 50, 100],
        fixed_kwargs={'n_rounds': 1000, 'df': 2.0},
        n_repeats=5
    )
    plot_complexity(results_arms, 'results/complexity_arms.pdf')
    
    # Test 3: Effect of heavy-tailed distributions
    print("\nAnalyzing complexity vs df (tail heaviness)...")
    results_df = empirical_complexity_analysis(
        func=your_simulation_function,
        param_name='df',
        param_values=[1.5, 2.0, 2.5, 3.0, 5.0, 10.0],
        fixed_kwargs={'n_rounds': 1000, 'n_arms': 5},
        n_repeats=5
    )
    plot_complexity(results_df, 'results/complexity_df.pdf')
    
    return results_rounds, results_arms, results_df

if __name__ == "__main__":
    # Run profiling
    profile_single_run()
    
    # Run complexity analysis
    analyze_complexity()
    
    # Print numerical issues summary
    print("\n" + "="*80)
    print("NUMERICAL STABILITY SUMMARY")
    print("="*80)
    summary = issue_tracker.summary()
    print(f"Warnings: {summary['total_warnings']}")
    print(f"Convergence failures: {summary['total_convergence_failures']}")
    print(f"Exceptions: {summary['total_exceptions']}")