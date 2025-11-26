# 607-project-03

This is the third project for 25FA STATS607.

## **Project Description**

This project is based on project 2 with computational efficiency and numerical stability improved.
Project 2 repository includes code on the comparison between a Risk-aware Contextual Bandit (using quantile regression as update) and an OLS bandit. The comparison mostly focuses on the scenarios with heavy-tailed errors.

### Key Improvements (Project 3)

This project extends Project 2 with significant optimizations:

1. **Numerical Stability** - Eliminated 1833+ runtime warnings through:
   - Safe division operations
   - Stable variance computation
   - Matrix condition checking with regularization
   - Heavy-tail handling via winsorization
   - Extreme value clipping

2. **Parallelization** - Implemented parallel simulation execution:
   - Multi-core processing using joblib
   - Configurable number of cores
   - Average speedup: 2.59×
   - Best speedup: 3.41× (at T=100)

3. **Comprehensive Testing** - Regression test suite with 87.5% pass rate:
   - Beta convergence verification
   - Action selection consistency
   - Edge case handling
   - Heavy-tailed distribution robustness
   - Numerical stability validation

## **Setup Instructions**

### 1. Clone the repository

```bash
git clone https://github.com/siyuxie2025/607-project-03
cd 607-project-03
```

### 2. Install Dependencies

```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
make install

# Or manually install:
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
make test-all     # Run all tests (unit + stability + parallel)
```

## **Running Simulations**

### Quick Start

```bash
make install      # Install dependencies
make test-all     # Verify everything works
make all          # Run full simulation pipeline
```

### Available Commands

Run `make help` to see all available commands. Key targets:

#### Original Pipeline
- `make all` - Run complete simulation pipeline
- `make simulate` - Run simulations and save results
- `make analyze` - Process results and generate statistics
- `make figures` - Create all visualizations

#### Optimization & Profiling
- `make profile` - Profile representative simulation
- `make complexity` - Analyze computational complexity O(n)
- `make benchmark` - Compare sequential vs parallel
- `make stability-check` - Check numerical warnings/issues
- `make parallel` - Run optimized parallel simulation

#### Testing
- `make test` - Run basic test suite
- `make test-all` - Run all tests (unit + stability + parallel)
- `make test-stability` - Test numerical stability functions
- `make test-parallel` - Test parallel implementation

#### Quick Actions
- `make quick` - Quick test simulation (5 sims, T=100)
- `make full` - Full simulation (100 sims, T=2000)
- `make info` - Show configuration

### Profiling

```bash
make profile      # Profile performance
make complexity   # Analyze O(n) complexity
make benchmark    # Compare optimizations
```

### Custom Parameters

```bash
# Custom simulation parameters
make simulate N_SIM=100 T=500 K=3

# Custom parallelization
make parallel N_JOBS=16

# Custom profiling
make profile N_SIM=20 T=200
```

## **Project Structure**

```
607project3/
├── src/
│   ├── simulation.py              # Main simulation (baseline)
│   ├── parallel_simulation.py     # Optimized parallel version
│   ├── methods.py                 # Bandit algorithms (RAB, OLS)
│   ├── generators.py              # Data generators (T, Normal, etc.)
│   ├── numerical_stability.py     # Stability utilities
│   ├── analyze_results.py         # Analysis and visualization
│   ├── create_performance_plots.py # Performance visualization
│   └── tests.py                   # Unit tests
├── docs/
│   ├── BASELINE.md                # Baseline performance documentation
│   └── OPTIMIZATION.md            # Optimization documentation
├── results/
│   ├── figures/                   # Generated plots
│   │   ├── complexity_comparison.pdf
│   │   ├── speedup_analysis.pdf
│   │   └── component_breakdown.pdf
│   ├── data/                      # Simulation data
│   ├── performance_comparison.csv
│   ├── speedup_data.csv
│   ├── regression_test_results.txt
│   └── regression_test_summary.md
├── test_regression.py             # Regression test suite
├── profile_simulation.py          # Profiling script
├── Makefile                       # Build automation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## **Results**

All results saved to `results/` directory.

See `make help` for all available commands.

### Generated Files

After running `make all` and related commands:

```
results/
├── simulation.prof                    # cProfile data
├── complexity_T.pdf                   # Complexity plots
├── complexity_K.pdf
├── complexity_d.pdf
├── complexity_df.pdf
├── complexity_summary.csv             # Summary table
├── performance_comparison.csv         # Performance comparison
├── speedup_data.csv                   # Speedup metrics
├── simulation_warnings.log            # Detailed log
├── numerical_issues_summary.json      # Issue tracker
├── figures/
│   ├── complexity_comparison.pdf      # Before/after complexity
│   ├── speedup_analysis.pdf           # Parallelization scaling
│   ├── component_breakdown.pdf        # Time breakdown by component
│   ├── regret_comparison.pdf          # Regret curves
│   ├── beta_error_comparison.pdf      # Beta error convergence
│   └── comparison_by_df.pdf           # Results by tail heaviness
├── data/
│   └── simulation_*.pkl               # Raw simulation data
├── regression_test_results.txt        # Full test output
└── regression_test_summary.md         # Test summary report
```

## **Estimated Runtime**

About 5 minutes with make all command.

### Detailed Timing

- `make quick` - ~30 seconds (5 sims, T=100)
- `make all` - ~5 minutes (50 sims, T=1000)
- `make full` - ~20 minutes (100 sims, T=2000)
- `make profile` - ~2 minutes (representative profiling)
- `make complexity` - ~10-15 minutes (comprehensive analysis)
- `make benchmark` - ~2-3 minutes (sequential vs parallel)

## **Performance Results**

### Speedup Analysis

| Configuration | Baseline Time | Optimized Time | Speedup |
|--------------|---------------|----------------|---------|
| T=50 | 0.71s | 0.94s | 0.76× |
| T=100 | 1.02s | 0.30s | **3.41×** |
| T=200 | 3.99s | 1.34s | 2.97× |
| T=500 | 9.70s | 3.35s | 2.90× |
| T=1000 | 20.63s | 7.10s | 2.91× |

**Average Speedup**: 2.59×
**Best Speedup**: 3.41× (at T=100)
**Total Time Saved**: 23.0 seconds

### Parallel Scaling

| Cores | Runtime | Speedup | Efficiency |
|-------|---------|---------|------------|
| 1 | 8.3s | 1.0× | 100% |
| 2 | 4.5s | 1.84× | 92% |
| 4 | 2.7s | 3.07× | 77% |
| 6 | 2.1s | 3.95× | 66% |
| 8 | 1.8s | 4.61× | 58% |

### Numerical Stability

- **Before Optimization**: 1833+ warnings
- **After Optimization**: <10 warnings
- **Convergence Failures**: 0
- **Exceptions**: 0

## **Documentation**

- [docs/BASELINE.md](docs/BASELINE.md) - Baseline performance analysis
  - Runtime profiling results
  - Computational complexity analysis (empirical + theoretical)
  - Numerical stability issues
  - Bottleneck identification

- [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) - Optimization strategies
  - Parallelization implementation
  - Numerical stability improvements
  - Performance impact analysis
  - Lessons learned

- [results/regression_test_summary.md](results/regression_test_summary.md) - Test results
  - 7/8 tests passed (87.5% success rate)
  - Detailed test analysis
  - Verification of correctness preservation

## **Testing**

### Run All Tests

```bash
make test-all
```

This runs:
1. Unit tests (basic functionality)
2. Numerical stability tests
3. Parallel simulation tests
4. Regression tests (correctness verification)

### Regression Test Suite

```bash
python test_regression.py --verbose
```

Tests include:
- ✅ Beta convergence
- ✅ Action selection consistency
- ✅ Edge case handling
- ✅ Heavy-tailed distributions
- ✅ Extreme value handling
- ✅ Small sample behavior
- ✅ Numerical stability
- ⚠️ Summary statistics (minor calibration issue)

**Result**: 7/8 tests passed (87.5%)

## **Key Features**

### Algorithms Implemented

1. **Risk-Aware Bandit (RAB)**
   - Uses quantile regression for robust estimation
   - Better performance with heavy-tailed errors
   - Computational complexity: O(K × T² log T)

2. **OLS Bandit**
   - Standard linear regression approach
   - Fast computation with Gaussian errors
   - Computational complexity: O(K × T × d³)

### Optimization Techniques

1. **Numerical Stability**
   - Safe division with default values
   - Stable variance computation (min_var ≥ 1e-8)
   - Matrix condition checking + Ridge regularization
   - Winsorization for heavy tails (1st/99th percentile)
   - Extreme value clipping (max 1e6)

2. **Parallelization**
   - Multi-core execution with joblib
   - Parallel across simulation replicates
   - Configurable core count
   - Automatic CPU detection

### Comprehensive Testing

- Unit tests for core functionality
- Stability tests for numerical operations
- Parallel execution verification
- Regression tests for correctness
- Edge case coverage
- Heavy-tail robustness

## **Dependencies**

Key packages (see `requirements.txt` for full list):
- numpy >= 2.3.4
- scipy >= 1.16.2
- pandas >= 2.3.3
- matplotlib >= 3.10.7
- scikit-learn >= 1.7.2
- joblib >= 1.5.2
- quantes >= 2.0.8
- tqdm >= 4.67.1

## **Requirements Compliance**

This project satisfies all Project 3 requirements:

- ✅ **Baseline Performance Documentation** - Comprehensive profiling and complexity analysis
- ✅ **Optimization Implementation** - Two categories: Numerical Stability + Parallelization
- ✅ **Optimization Documentation** - Detailed before/after comparisons and lessons learned
- ✅ **Updated Makefile** - All required targets (profile, complexity, benchmark, parallel, stability-check)
- ✅ **Performance Visualization** - Multiple plots comparing baseline vs optimized
- ✅ **Regression Tests** - Comprehensive test suite with 87.5% pass rate

## **Citation**

If you use this code, please cite:

```bibtex
@misc{xie2025bandit,
  author = {Xie, Siyu},
  title = {Optimized Risk-Aware Contextual Bandit Simulation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/siyuxie2025/607-project-03}
}
```

## **License**

This project is for educational purposes as part of STATS607 coursework.

## **Contact**

For questions or issues, please open an issue on GitHub or contact the author.

---

**Complete workflow (install → test → profile → run → visualize)**
```bash
make full-pipeline
```

This will run the entire analysis pipeline with comprehensive profiling and optimization.
