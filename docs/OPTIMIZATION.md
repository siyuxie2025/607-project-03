# Simulation Optimization Documentation
This is the optimization documentation for the project. I will optimize the numerical stability and parallelization for this project. 

## Numerical Stability
**Problem**: Division by zero warinings, this is due to extreme values for heavy-tail data and variance calculation with small sample size. In early rounds OLS update, there are ill-conditioned matrices. 
I eliminated division by zero warnings and improved robustness for heavy-tailed distributions. They are included in numerical stability module `src/numerical_stability.py` with:
1. Safe division: returns 0 when sample size n=0;
2. Stable variance computation: always >= 1e-8;
3. Matrix condition checking: condition check + regularization;
4. Heavy-tail handing: caps extreme outliers at 1st/99th quantile. 

**Code Comparison**:

```{python}
# Before: 
class RiskAwareBandit:
    def compute_confidence_bound(self, arm, t):
        # Get historical data for this arm
        data = self.history[arm][:t]
        
        # Compute standard error
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(len(data))  # ⚠️ Division by zero when t=0
        
        return mean + 2 * se
# After:
from src.numerical_stability import stable_mean, stable_std, safe_divide, EPSILON

class RiskAwareBandit:
    def compute_confidence_bound(self, arm, t):
        # Get historical data for this arm
        data = self.history[arm][:t]
        
        # Compute standard error with stability checks
        if len(data) < 2:  # Need at least 2 points for std
            return 0.0
        
        mean = stable_mean(data)  # Handles NaN/Inf
        std = stable_std(data, min_var=1e-8)  # Guaranteed > 0
        n = max(len(data), 1)  # Prevent division by zero
        se = safe_divide(std, np.sqrt(n), default=EPSILON)
        
        return mean + 2 * se
```

## Parallelization
Problem: Sequential simulation replications.

