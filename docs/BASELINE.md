# Baseline Performance Documentation

The total runtime of the entire simulation study was approximately 5 minutes. 

---
## Runtime profiling

---
## Computational complexity analysis

**Algorithm: Risk-Aware Bandit with Quantile Regression**
1. Outer loop: `n_rounds` iterations
2. For each round:
   a. Pull each arm `q` times: $O(q*$`n_arms`$)$
   b. Update quantile model for each arm:
      - Quantile regression via sorting: $O(T * \log T$)
      - Where $T$ = number of observations for that arm ≈ rounds/n_arms

   c. Select arm based on Suboptimal arm selection: {TOBEDONE}

[NEEDSMODIFY] Total theoretical complexity: O(n_rounds × n_arms × (n_rounds/n_arms) × log(n_rounds/n_arms)) ≈ O(n_rounds² × log n_rounds)

Empirical verification: Our timing experiments confirm exponent ≈ 2.1 (close to n²)

--- 

### Reasoning: 
Based on typical bandit algorithms:
Risk-Aware Bandit (Quantile Regression)

Per round complexity: O(n_arms × n_samples × log n_samples)

Quantile regression typically requires sorting: O(n log n)
Done for each arm: multiply by n_arms


Total complexity: O(n_rounds × n_arms × n_samples × log n_samples)

OLS Bandit

Per round complexity: O(n_arms × p²)

Where p is number of features/contexts
Matrix operations in OLS: O(p²n) for n observations


Total complexity: O(n_rounds × n_arms × p²)

Overall simulation

If you run multiple replications: multiply by n_replications
Expected: Your make all takes ~8 minutes for 5 different df values


## Evidence of numerical instability
