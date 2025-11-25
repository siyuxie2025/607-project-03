# Baseline Performance Documentation

---
## Runtime profiling
The total runtime of entire simulation study was approximately 2 minutes. 

### Performance Benchmarks

Runtime on [MacBook Pro M1, 16GB RAM]:

| Configuration | Runtime | Speedup Factor |
|--------------|---------|----------------|
| T=50, K=2, d=10 | 2.1s | 1.0× (baseline) |
| T=100, K=2, d=10 | 8.7s | 4.1× (confirms O(T²)) |
| T=200, K=2, d=10 | 35.2s | 16.8× (confirms O(T²)) |
| T=150, K=5, d=10 | 24.3s | 11.6× (T and K scale) |

### Quick Profile
Default: n_sim = 10, K=5, d=10, T=100, df=2.0
`python profile_simulation.py --quick`
Output:
```{bash}
Runtime: 1.832 seconds
Average time per simulation: 0.183 seconds
⚠ 51 numerical warnings detected (see log)

Final Regret Summary:
  Risk-Aware: 145.17 ± 14.29
  OLS:        140.93 ± 17.64
```

Test larger scale
`python profile_simulation.py --quick --T=1000 --K=10`

### Full Profiling Suite
`python profile_simulation.py --full`

This will:
1. Profile with cProfile - identify bottlenecks
2. Analyze T complexity - how runtime scales with number of rounds
3. Analyze K complexity - how runtime scales with number of arms
4. Analyze d complexity - how runtime scales with context dimension
5. Test numerical stability - across df values (1.5, 2.0, 2.25, 3.0, 5.0, 10.0)

**Generated Files**
```
results/
├── simulation.prof              # cProfile data
├── complexity_T.pdf            # Shows O(T²) growth
├── complexity_K.pdf            # Shows linear scaling
├── complexity_d.pdf            # Context dimension effect
├── complexity_df.pdf           # Numerical stability test
├── complexity_summary.csv      # Table for your report
├── simulation_warnings.log     # Detailed issues
└── numerical_issues_summary.json
```


From the profiling result, running the code below gives us specific runtime results. 
```{bash}
python -m pstats results/simulation.prof
(pstats) sort cumtime
(pstats) stats 30
```
Among the functions, run_simulation, run_one_scenario, _run_one_timestep, update_beta take up most time. 

| ncalls | tottime | percall | cumtime | percall | Source (Function) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **2/1** | 0.000 | 0.000 | **3.657** | 3.657 | threading.py (wait) |
| **2/1** | 0.000 | 0.000 | **3.657** | 3.657 | profile_simulation.py (run_simulation_with_tracking) |
| **2/1** | 0.001 | 0.000 | **3.654** | 3.654 | simulation.py (run_simulation) |
| **10** | 0.002 | 0.000 | **3.633** | 0.363 | simulation.py (run_one_scenario) |
| **1000** | 0.005 | 0.000 | **3.623** | 0.004 | simulation.py (_run_one_timestep) |
| **1000** | 0.009 | 0.000 | **3.463** | 0.003 | methods.py (update_beta) |
| **400** | 0.002 | 0.000 | **3.434** | 0.009 | quantes/linear.py (fit) |
| **400** | 0.148 | 0.000 | **3.413** | 0.009 | quantes/solvers.py (minimize) |
| **40880** | 0.233 | 0.000 | **2.282** | 0.000 | scipy/special/_logsumexp.py (logsumexp) |
| **40880** | 0.359 | 0.000 | **1.391** | 0.000 | scipy/special/_logsumexp.py (_logsumexp) |
| **54444** | 0.053 | 0.000 | 0.521 | 0.000 | quantes/linear.py (<lambda>) |
| **54444** | 0.040 | 0.000 | 0.468 | 0.000 | quantes/utils.py (smooth_check) |
| **81760** | 0.018 | 0.000 | 0.430 | 0.000 | scipy/_lib/array_api_extra/_lib/_at.py (set) |
| **81760** | 0.142 | 0.000 | 0.412 | 0.000 | scipy/_lib/array_api_extra/_lib/_at.py (_op) |
| **54444** | 0.129 | 0.000 | 0.346 | 0.000 | quantes/linear.py (<lambda>) |
| **165520** | 0.089 | 0.000 | 0.231 | 0.000 | numpy/_core/numerictypes.py (isdtype) |
| **40880** | 0.074 | 0.000 | 0.221 | 0.000 | scipy/_lib/_array_api.py (xp_promote) |
| **54444** | 0.038 | 0.000 | 0.217 | 0.000 | quantes/utils.py (conquer_weight) |

### Bottleneck Analysis

1. Quantile regression: RiskAware bottleneck
2. Main loop: unavoidable, parallization could be helpful
3. Arm selection
4. OLS updates: already very fast


## Complexity Analysis

### Numerical Analysis:
Complexity Analysis Summary:

| Parameter | Exponent ($b$) | Complexity | $R^2$ | Interpretation | Min $\to$ Max Time |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **T** | 0.83 | $O(T^{0.83})$ | 0.911 | approximately linear $O(n)$ | $0.65s \to 4.71s$ |
| **K** | 0.54 | $O(K^{0.54})$ | 0.996 | sublinear (very efficient!) | $0.64s \to 2.18s$ |
| **d** | -0.32 | $O(d^{-0.32})$ | 0.521 | sublinear (very efficient!) | $0.90s \to 2.39s$ |
| **df** | -0.02 | $O(df^{-0.02})$ | 0.022 | sublinear (very efficient!) | $0.81s \to 1.09s$ |

Here is the rest of my document.

### Theoretical Analysis:
Risk-Aware Bandit has theoretical complexity $O(K\times T^2 \log T)$. 
 - Quantile regression requires $O(t \log t)$ per arm.
 - There are $K$ arms. 

OLS Bandit requires $O(K\times T\times d^3)$.
 - Matrix operation: $O(d^3)$ per arm (for $d\times d$ matrix).
 - $K$ arms. 
 

## Numerical Stability Report
Total warnings: 1833
Convergence failures: 0
Exceptions: 0
Large values detected: 0

Warnings by df (tail heaviness):
```{bash}
  df=1.5: 102 warnings
  df=2.0: 1326 warnings
  df=2.25: 101 warnings
  df=3.0: 102 warnings
  df=5.0: 101 warnings
  df=10.0: 101 warnings
```
```{bash}
RuntimeWarning: invalid value encountered in divide
```