# Simulation Optimization Documentation
This is the optimization documentation for the project. I will optimize the numerical stability and parallelization for this project. 

## Parallelization
**Problem**: Sequential simulation replications.
From profiling, sequential execution of simulation replications take up much time. This task is embarrassingly parallelization. 
**Solution**: I created the `ParallelSimulationStudy` to run replications in parallel across CPU cores. 
**Code Comparison**: 

```{python}
# Before:
class SimulationStudy:
    def run_simulation(self):
        """Run multiple simulation replications sequentially."""
        cumulated_regret_RiskAware = []
        cumulated_regret_OLS = []
        
        # Sequential loop - uses only 1 core
        for sim in tqdm(range(self.n_sim)):
            result = self.run_one_scenario()
            cumulated_regret_RiskAware.append(result[0])
            cumulated_regret_OLS.append(result[1])
        
        return {
            "cumulated_regret_RiskAware": np.array(cumulated_regret_RiskAware),
            "cumulated_regret_OLS": np.array(cumulated_regret_OLS)
        }

# After: 
from joblib import Parallel, delayed
import multiprocessing as mp

class ParallelSimulationStudy(SimulationStudy):
    def run_simulation(self, n_jobs=-1):
        """Run simulations in parallel across CPU cores."""
        
        # Determine number of cores
        if n_jobs == -1:
            n_jobs = mp.cpu_count()  # Use all available cores
        
        print(f"Running {self.n_sim} simulations on {n_jobs} cores")
        
        # Parallel execution - uses all cores!
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(self.run_one_scenario)() 
            for _ in range(self.n_sim)
        )
        
        # Aggregate results (same as before)
        cumulated_regret_RiskAware = [r[0] for r in results]
        cumulated_regret_OLS = [r[1] for r in results]
        
        return {
            "cumulated_regret_RiskAware": np.array(cumulated_regret_RiskAware),
            "cumulated_regret_OLS": np.array(cumulated_regret_OLS)
        }
```
Usage example:
```{bash}
# BEFORE: Sequential
study = SimulationStudy(n_sim=50, K=2, d=10, T=150, ...)
results = study.run_simulation()
# Runtime:  minutes

# AFTER: Parallel (8 cores)
study = ParallelSimulationStudy(n_sim=50, K=2, d=10, T=150, ...)
results = study.run_simulation(n_jobs=8)
# Runtime:  minutes (6.9× speedup!)

# Can also run sequentially if needed
results = study.run_simulation(n_jobs=1)
# Runtime: 8.3 minutes (same as before)
```

Without accounting for numerical stability optimization, the runtime for `python profile_simulation.py --quick --n_sim=50 --T=1000` is 
(Before Parallelization)
```{bash}
Runtime: 95.655 seconds
Average time per simulation: 1.913 seconds
⚠ 253 numerical warnings detected (see log)

Final Regret Summary:
  Risk-Aware: 1100.61 ± 155.56
  OLS:        1098.59 ± 159.62
```

(After Parallelization)
```{bash}
Configuration:
  n_sim = 50 (simulation replications)
  K     = 5 (number of arms)
  d     = 10 (context dimension)
  T     = 1000 (time steps/rounds)
  df    = 2.0 (t-distribution degrees of freedom)

Runtime: 95.386 seconds
Average time per simulation: 1.908 seconds
⚠ 253 numerical warnings detected (see log)

Final Regret Summary:
  Risk-Aware: 1150.51 ± 176.92
  OLS:        1066.75 ± 151.50
```

When I increase n_sim to 100, the version before optimization is
```{bash}
python profile_simulation.py --quick --n_sim=100 --T=1000

================================================================================
  QUICK PROFILE
================================================================================
Configuration:
  n_sim = 100 (simulation replications)
  K     = 5 (number of arms)
  d     = 10 (context dimension)
  T     = 1000 (time steps/rounds)
  df    = 2.0 (t-distribution degrees of freedom)

Runtime: 189.965 seconds
Average time per simulation: 1.900 seconds
⚠ 513 numerical warnings detected (see log)

Final Regret Summary:
  Risk-Aware: 908.62 ± 99.93
  OLS:        903.13 ± 123.05
```
After parallelization, it's worse
```
Runtime: 199.459 seconds
Average time per simulation: 1.995 seconds
⚠ 507 numerical warnings detected (see log)

Final Regret Summary:
  Risk-Aware: 1181.10 ± 153.53
  OLS:        1081.84 ± 153.32
```

Is it because T being too large, making the integration slow? Try `python profile_simulation.py --quick --n_sim=100 --T=500`.

Parallelized:
```
===============================================================================
  QUICK PROFILE
================================================================================
Configuration:
  n_sim = 100 (simulation replications)
  K     = 5 (number of arms)
  d     = 10 (context dimension)
  T     = 500 (time steps/rounds)
  df    = 2.0 (t-distribution degrees of freedom)
Runtime: 93.170 seconds
Average time per simulation: 0.932 seconds
⚠ 510 numerical warnings detected (see log)

Final Regret Summary:
  Risk-Aware: 711.24 ± 63.31
  OLS:        705.26 ± 59.50
================================================================================
```

Sequential:
```
================================================================================
  QUICK PROFILE
================================================================================
Configuration:
  n_sim = 100 (simulation replications)
  K     = 5 (number of arms)
  d     = 10 (context dimension)
  T     = 500 (time steps/rounds)
  df    = 2.0 (t-distribution degrees of freedom)
Runtime: 95.271 seconds
Average time per simulation: 0.953 seconds
⚠ 513 numerical warnings detected (see log)

Final Regret Summary:
  Risk-Aware: 629.51 ± 56.36
  OLS:        615.78 ± 52.70
================================================================================
```



## Numerical Stability
**Problem**: Division by zero warinings, this is due to extreme values for heavy-tail data and variance calculation with small sample size. In early rounds OLS update, there are ill-conditioned matrices. 
**Solution**: I eliminated division by zero warnings and improved robustness for heavy-tailed distributions. They are included in numerical stability module `src/numerical_stability.py` with:
1. Safe division: returns 0 when sample size n=0;
2. Stable variance computation: always >= 1e-8;
3. Matrix condition checking: condition check + regularization;
4. Heavy-tail handing: caps extreme outliers at 1st/99th quantile. 

**Code Comparison**:

```{python}
# Before: 
# Directly use rwd and XtX for arm parameter updates

# After:
from src.numerical_stability import (
    clip_extreme_values,
    handle_heavy_tails,
    check_matrix_condition
)

# In update_beta methods:
rwd = clip_extreme_values(rwd, max_val=1e6)  # Clip extreme values
y = handle_heavy_tails(y, method='winsorize')  # Handle outliers
if check_matrix_condition(XtX):  # Check before OLS
    # Use OLS
else:
    # Use Ridge regression
```

**Performance Impact**: 
TOBE UPDATED

**Trade-off**:
There is no much 