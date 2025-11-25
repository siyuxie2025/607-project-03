# 607-project-03
This is the third project for 25FA STATS607. 

## **Project Description**
This project is based on project 2 with computational efficiency and nemerical stability improved. 
Project 2 repository includes code on the comparison between a Risk-aware Contextual Bandit (using quantile regression as update) and an OLS bandit. The comparison mostly focuses on the scenarios with heavy-tailed errors.

## **Setup Instructions**

1. **Clone the repository:**
```bash
git clone https://github.com/siyuxie2025/607-project-03
cd 607-project-03
```

2. **Run Analysis**

## Running Simulations

### Quick Start
```bash
make install      # Install dependencies
make test-all     # Verify everything works
make run-all      # Run full study (6-8 min)
```

### Profiling
```bash
make profile      # Profile performance
make complexity   # Analyze O(n) complexity
make benchmark    # Compare optimizations
```

### Results
All results saved to `results/` directory:
- Data: `results/data/`
- Figures: `results/figures/`
- Profiling: `results/*.prof`

See `make help` for all available commands.


```
# Complete workflow (install → test → profile → run → visualize)
make full-workflow
```

After running `make run-all` and related commands:
```
results/
├── simulation.prof                    # cProfile data
├── complexity_T.pdf                   # Complexity plots
├── complexity_K.pdf
├── complexity_d.pdf
├── complexity_df.pdf
├── complexity_summary.csv             # Summary table
├── benchmark_results.csv              # Performance comparison
├── optimization_comparison.csv        # Before/after
├── stability_comparison.csv           # Stability results
├── simulation_warnings.log            # Detailed log
├── numerical_issues_summary.json      # Issue tracker
├── data/                              # Simulation results
│   └── simulation_*.pkl
└── figures/                           # Generated plots
    ├── regret_comparison.pdf
    ├── beta_error_comparison.pdf
    └── comparison_by_df.pdf
```

## **Estimated Runtime**
About 5 minutes with make all command. 

## **Summary of Key Findings**
![Cumulative regret comparison](results/main_regret_comparison_K2_d10_T1000.png)

When the number of arms $K=2$, the cumulative regret for risk-aware bandit is slightly better than the OLS bandit, especially when the error distribution is heavy-tailed. But as the degree of freedom increases, the advantage becomes less. 

![Beta MSE comparison](results/main_beta_error_comparison_K2_d10_T1000.png)
RAB (Risk-aware Bandit) has an obvious advantage for the recovery of the true beta value, especially when the error has heavy-tailed distributions. 

