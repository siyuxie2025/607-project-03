# Makefile for Bandit Algorithm Simulation Pipeline
# ==================================================
# Combines original simulation pipeline with optimization/profiling targets

# Python interpreter - auto-detect python3 or python
PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)
PIP := $(shell command -v pip3 2>/dev/null || command -v pip 2>/dev/null || echo pip3)

# Set PYTHONPATH to include project root
export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

# Directories
SRC_DIR := src
RESULTS_DIR := results
FIGURES_DIR := $(RESULTS_DIR)/figures
DATA_DIR := $(RESULTS_DIR)/data
TESTS_DIR := tests
DOCS_DIR := docs

# Main scripts
MAIN_SCRIPT := $(SRC_DIR)/main.py
ANALYZE_SCRIPT := $(SRC_DIR)/analyze_results.py

# Configuration - Original parameters
N_SIM := 50
K := 2
D := 10
T := 1000
Q := 2
TAU := 0.5
SEED := 1010

# Configuration - Optimization parameters
N_JOBS := 8  # Number of parallel jobs

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Phony targets
.PHONY: all help setup clean clean-all simulate analyze figures test check install \
        profile profile-quick profile-full complexity benchmark parallel stability-check \
        run-sequential run-parallel compare visualize view-profile clean-profile \
        quick full lint coverage watch info test-all test-stability test-parallel \
        diagnose

# ==================================================
# HELP TARGET (shows all targets organized by category)
# ==================================================

help:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  Bandit Algorithm Simulation Pipeline                         ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)═══ Original Pipeline Targets ═══$(NC)"
	@echo "  $(CYAN)make all$(NC)           - Run complete simulation pipeline"
	@echo "  $(CYAN)make simulate$(NC)      - Run simulations and save results"
	@echo "  $(CYAN)make analyze$(NC)       - Process results and generate statistics"
	@echo "  $(CYAN)make figures$(NC)       - Create all visualizations"
	@echo "  $(CYAN)make test$(NC)          - Run test suite"
	@echo ""
	@echo "$(GREEN)═══ Optimization & Profiling ═══$(NC)"
	@echo "  $(CYAN)make profile$(NC)       - Profile representative simulation"
	@echo "  $(CYAN)make complexity$(NC)    - Analyze computational complexity O(n)"
	@echo "  $(CYAN)make benchmark$(NC)     - Compare sequential vs parallel"
	@echo "  $(CYAN)make stability-check$(NC) - Check numerical warnings/issues"
	@echo "  $(CYAN)make parallel$(NC)      - Run optimized parallel simulation"
	@echo "  $(CYAN)make compare$(NC)       - Compare before/after optimization"
	@echo "  $(CYAN)make diagnose$(NC)      - Diagnose performance issues"
	@echo ""
	@echo "$(GREEN)═══ Setup & Installation ═══$(NC)"
	@echo "  $(CYAN)make setup$(NC)         - Create directories"
	@echo "  $(CYAN)make install$(NC)       - Install dependencies"
	@echo "  $(CYAN)make test-all$(NC)      - Run all tests (unit + stability + parallel)"
	@echo ""
	@echo "$(GREEN)═══ Cleaning ═══$(NC)"
	@echo "  $(CYAN)make clean$(NC)         - Remove generated files (keep data)"
	@echo "  $(CYAN)make clean-all$(NC)     - Remove ALL files including data"
	@echo "  $(CYAN)make clean-profile$(NC) - Remove profiling outputs"
	@echo ""
	@echo "$(GREEN)═══ Quick Actions ═══$(NC)"
	@echo "  $(CYAN)make quick$(NC)         - Quick test simulation (5 sims, T=100)"
	@echo "  $(CYAN)make full$(NC)          - Full simulation (100 sims, T=2000)"
	@echo "  $(CYAN)make info$(NC)          - Show configuration"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make all                      # Run entire pipeline"
	@echo "  make simulate T=500           # Custom time horizon"
	@echo "  make parallel N_JOBS=16       # Use 16 cores"
	@echo "  make profile N_SIM=20         # Profile with 20 simulations"
	@echo ""
	@echo "$(BLUE)════════════════════════════════════════════════════════════════$(NC)"

# ==================================================
# MAIN PIPELINE TARGETS (Original)
# ==================================================

all: setup simulate analyze figures
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✓ Complete pipeline finished successfully!$(NC)"
	@echo "$(GREEN)════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)Results:  $(RESULTS_DIR)$(NC)"
	@echo "$(BLUE)Figures:  $(FIGURES_DIR)$(NC)"
	@echo "$(BLUE)Data:     $(DATA_DIR)$(NC)"

simulate: setup
	@echo "$(YELLOW)Running simulations...$(NC)"
	@echo "$(BLUE)Parameters: N_SIM=$(N_SIM), K=$(K), D=$(D), T=$(T)$(NC)"
	@cd src && $(PYTHON) -c " \
		from simulation import SimulationStudy; \
		from generators import TGenerator, TruncatedNormalGenerator; \
		print('Running $(N_SIM) simulation replications...'); \
		study = SimulationStudy( \
			n_sim=$(N_SIM), K=$(K), d=$(D), T=$(T), \
			q=$(Q), h=0.5, tau=$(TAU), \
			err_generator=TGenerator(df=2.25, scale=0.7), \
			context_generator=TruncatedNormalGenerator(0, 1), \
			random_seed=$(SEED) \
		); \
		results = study.run_simulation(); \
		study.save_results(); \
		study.save_summary_statistics(); \
		print('Results saved to results/ directory') \
	"
	@echo "$(GREEN)✓ Simulations complete! Results saved in $(DATA_DIR)$(NC)"

analyze: 
	@echo "$(YELLOW)Analyzing results...$(NC)"
	@if [ ! -d "$(DATA_DIR)" ] || [ -z "$$(ls -A $(DATA_DIR) 2>/dev/null)" ]; then \
		echo "$(RED)Error: No simulation data found in $(DATA_DIR)$(NC)"; \
		echo "$(YELLOW)Please run 'make simulate' first$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(ANALYZE_SCRIPT) --data_dir $(DATA_DIR) --output_dir $(RESULTS_DIR)
	@echo "$(GREEN)✓ Analysis complete! Summary saved in $(RESULTS_DIR)$(NC)"

figures:
	@echo "$(YELLOW)Generating figures...$(NC)"
	@if [ ! -d "$(DATA_DIR)" ] || [ -z "$$(ls -A $(DATA_DIR) 2>/dev/null)" ]; then \
		echo "$(RED)Error: No simulation data found$(NC)"; \
		echo "$(YELLOW)Please run 'make simulate' first$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(ANALYZE_SCRIPT) --data_dir $(DATA_DIR) --output_dir $(FIGURES_DIR) --figures_only
	@echo "$(GREEN)✓ Figures generated in $(FIGURES_DIR)$(NC)"

# ==================================================
# SETUP AND INSTALLATION
# ==================================================

setup:
	@echo "$(YELLOW)Setting up directories...$(NC)"
	@mkdir -p $(RESULTS_DIR) $(FIGURES_DIR) $(DATA_DIR) $(DOCS_DIR)
	@mkdir -p scripts  # For profiling scripts
	@echo "$(GREEN)✓ Directories created$(NC)"

install:
	@echo "$(YELLOW)Installing required packages...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Core packages installed$(NC)"

install-dev: install
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	$(PIP) install pytest pytest-cov black flake8 mypy isort joblib memory_profiler line_profiler
	@echo "$(GREEN)✓ Development packages installed$(NC)"

# ==================================================
# TESTING
# ==================================================

test:
	@echo "$(YELLOW)Running test suite...$(NC)"
	@if [ -d "$(TESTS_DIR)" ]; then \
		$(PYTHON) -m pytest $(TESTS_DIR) -v --color=yes; \
	else \
		echo "$(YELLOW)No tests directory found. Running basic import tests...$(NC)"; \
		cd $(SRC_DIR) && $(PYTHON) -c "from simulation import SimulationStudy; print('✓ Imports OK')"; \
		cd $(SRC_DIR) && $(PYTHON) -c "from methods import RiskAwareBandit, OLSBandit; print('✓ Methods OK')"; \
	fi
	@echo "$(GREEN)✓ Tests passed!$(NC)"

test-stability:
	@echo "$(YELLOW)Testing numerical stability...$(NC)"
	@cd $(SRC_DIR) && $(PYTHON) -c "from numerical_stability import *; \
		print('Testing safe_divide...'); \
		assert safe_divide(4, 2) == 2.0; \
		assert safe_divide(4, 0, default=0) == 0.0; \
		print('✓ safe_divide works'); \
		import numpy as np; \
		var = stable_variance(np.array([1,1,1]), min_var=1e-8); \
		assert var >= 1e-8; \
		print('✓ stable_variance works'); \
		assert check_matrix_condition(np.eye(3)); \
		print('✓ check_matrix_condition works')"
	@echo "$(GREEN)✓ Numerical stability tests passed$(NC)"

test-parallel:
	@echo "$(YELLOW)Testing parallel simulation...$(NC)"
	@cd $(SRC_DIR) && $(PYTHON) -c "from parallel_simulation import ParallelSimulationStudy; \
		from generators import TGenerator, TruncatedNormalGenerator; \
		study = ParallelSimulationStudy( \
			n_sim=3, K=2, d=5, T=20, q=2, h=0.5, tau=0.5, \
			err_generator=TGenerator(df=2.0, scale=0.7), \
			context_generator=TruncatedNormalGenerator(0, 1), \
			random_seed=42 \
		); \
		results = study.run_simulation(n_jobs=2, verbose=0); \
		assert 'cumulated_regret_RiskAware' in results; \
		print('✓ Parallel simulation works')"
	@echo "$(GREEN)✓ Parallel simulation tests passed$(NC)"

test-all: test test-stability test-parallel
	@echo "$(GREEN)╔════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║  ✓ All tests passed successfully!    ║$(NC)"
	@echo "$(GREEN)╚════════════════════════════════════════╝$(NC)"

check:
	@echo "$(YELLOW)Running code quality checks...$(NC)"
	@echo "$(BLUE)Checking code style with flake8...$(NC)"
	-$(PYTHON) -m flake8 $(SRC_DIR) --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "$(BLUE)Checking imports with isort...$(NC)"
	-$(PYTHON) -m isort $(SRC_DIR) --check-only
	@echo "$(GREEN)✓ Code quality checks complete!$(NC)"

# ==================================================
# PROFILING & OPTIMIZATION
# ==================================================

profile: setup
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  Running Performance Profiling                            ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Configuration:$(NC)"
	@echo "  n_sim = $(N_SIM), K = $(K), d = $(D), T = $(T)"
	@echo "  Output: $(RESULTS_DIR)/simulation.prof"
	@echo ""
	@$(PYTHON) profile_simulation.py --quick \
		--n_sim=$(N_SIM) --K=$(K) --d=$(D) --T=$(T) --df=2.25
	@echo ""
	@echo "$(GREEN)✓ Profiling complete$(NC)"
	@echo "$(YELLOW)View with:$(NC) make view-profile"

profile-quick: setup
	@echo "$(BLUE)Running quick profile (n_sim=10, T=50)...$(NC)"
	@$(PYTHON) profile_simulation.py --quick --n_sim=10 --T=50 --df=2.25
	@echo "$(GREEN)✓ Quick profile complete$(NC)"

profile-full: setup
	@echo "$(RED)WARNING: Full profiling takes 15-20 minutes!$(NC)"
	@echo "Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	@echo "$(BLUE)Running full profiling suite...$(NC)"
	@$(PYTHON) profile_simulation.py --full
	@echo "$(GREEN)✓ Full profiling complete$(NC)"

view-profile:
	@if [ ! -f $(RESULTS_DIR)/simulation.prof ]; then \
		echo "$(RED)Error: No profile data found$(NC)"; \
		echo "$(YELLOW)Run 'make profile' first$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Opening profile viewer...$(NC)"
	@echo "$(YELLOW)Commands: sort cumtime, stats 20, quit$(NC)"
	@$(PYTHON) -m pstats $(RESULTS_DIR)/simulation.prof

complexity: setup
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  Running Complexity Analysis O(n)                          ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Analyzing how runtime scales with T, K, d, df...$(NC)"
	@echo "$(YELLOW)This will take 10-15 minutes...$(NC)"
	@echo ""
	@if [ -f scripts/complexity_analysis.py ]; then \
		$(PYTHON) scripts/complexity_analysis.py; \
	else \
		echo "$(YELLOW)Running inline complexity analysis...$(NC)"; \
		cd $(SRC_DIR) && $(PYTHON) -c "from profile_simulation import empirical_complexity_analysis, plot_complexity; \
			print('Testing T complexity...'); \
			r1 = empirical_complexity_analysis('T', [50,100,200,500], \
				{'n_sim':5, 'K':$(K), 'd':$(D), 'df':2.0}, n_repeats=3); \
			plot_complexity(r1); \
			print('Testing K complexity...'); \
			r2 = empirical_complexity_analysis('K', [2,5,10,20], \
				{'n_sim':5, 'd':$(D), 'T':100, 'df':2.0}, n_repeats=3); \
			plot_complexity(r2); \
			print('Complexity analysis complete!')"; \
	fi
	@echo "$(GREEN)✓ Complexity analysis complete$(NC)"

benchmark: setup
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  Benchmark: Sequential vs Parallel                        ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@if [ -f scripts/benchmark.py ]; then \
		$(PYTHON) scripts/benchmark.py; \
	else \
		echo "$(YELLOW)Running inline benchmark...$(NC)"; \
		cd $(SRC_DIR) && $(PYTHON) -c "import time; \
			from simulation import SimulationStudy; \
			from parallel_simulation import ParallelSimulationStudy; \
			from generators import TGenerator, TruncatedNormalGenerator; \
			config = {'n_sim': 10, 'K': 2, 'd': 10, 'T': 50, 'q': 2, 'h': 0.5, 'tau': 0.5, \
				'err_generator': TGenerator(df=2.25, scale=0.7), \
				'context_generator': TruncatedNormalGenerator(0, 1), 'random_seed': 42}; \
			print('Sequential:'); \
			s = SimulationStudy(**config); \
			start = time.time(); \
			s.run_simulation(); \
			seq = time.time() - start; \
			print(f'  Time: {seq:.2f}s'); \
			print('Parallel (4 cores):'); \
			p = ParallelSimulationStudy(**config); \
			start = time.time(); \
			p.run_simulation(n_jobs=4, verbose=0); \
			par = time.time() - start; \
			print(f'  Time: {par:.2f}s'); \
			print(f'Speedup: {seq/par:.2f}×')"; \
	fi
	@echo ""
	@echo "$(GREEN)✓ Benchmark complete$(NC)"

stability-check: setup
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  Numerical Stability Check                                 ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@if [ -f scripts/stability_check.py ]; then \
		$(PYTHON) scripts/stability_check.py; \
	else \
		echo "$(YELLOW)Testing stability across df values...$(NC)"; \
		$(PYTHON) -c "from profile_simulation import simple_timer, issue_tracker; \
			import warnings; \
			for df in [1.5, 2.0, 2.25, 3.0, 5.0]: \
				print(f'df={df}:', end=' '); \
				with warnings.catch_warnings(record=True) as w: \
					warnings.simplefilter('always'); \
					_, _ = simple_timer(n_sim=5, K=2, d=10, T=50, df=df); \
					print(f'{len(w)} warnings' if len(w) > 0 else '✓ OK')"; \
	fi
	@echo ""
	@echo "$(GREEN)✓ Stability check complete$(NC)"

parallel: setup
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  Running Parallel Simulation                               ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Configuration: n_sim=$(N_SIM), n_jobs=$(N_JOBS)$(NC)"
	@if [ -f scripts/run_parallel.py ]; then \
		$(PYTHON) scripts/run_parallel.py --n_sim=$(N_SIM) --n_jobs=$(N_JOBS) --K=$(K) --d=$(D) --T=$(T); \
	else \
		echo "$(YELLOW)Running inline parallel simulation...$(NC)"; \
		cd $(SRC_DIR) && $(PYTHON) -c "from parallel_simulation import ParallelSimulationStudy; \
			from generators import TGenerator, TruncatedNormalGenerator; \
			study = ParallelSimulationStudy( \
				n_sim=$(N_SIM), K=$(K), d=$(D), T=$(T), q=$(Q), h=0.5, tau=$(TAU), \
				err_generator=TGenerator(df=2.25, scale=0.7), \
				context_generator=TruncatedNormalGenerator(0, 1), random_seed=$(SEED) \
			); \
			print('Running parallel simulation...'); \
			results = study.run_simulation(n_jobs=$(N_JOBS)); \
			study.save_results(); \
			print('✓ Parallel simulation complete!')"; \
	fi
	@echo "$(GREEN)✓ Parallel simulation complete$(NC)"

compare: setup
	@echo "$(BLUE)Comparing Sequential vs Parallel Performance$(NC)"
	@if [ -f scripts/compare_optimization.py ]; then \
		$(PYTHON) scripts/compare_optimization.py; \
	else \
		$(MAKE) benchmark; \
	fi
	@echo "$(GREEN)✓ Comparison complete$(NC)"

diagnose: setup
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  Performance Diagnostic                                    ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@if [ -f scripts/diagnose_performance.py ]; then \
		$(PYTHON) scripts/diagnose_performance.py; \
	else \
		echo "$(YELLOW)Create diagnose_performance.py from the debugging artifact$(NC)"; \
		echo "$(YELLOW)Or run: make benchmark$(NC)"; \
	fi

# ==================================================
# CLEANING
# ==================================================

clean:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	@rm -rf $(FIGURES_DIR)/*.pdf $(FIGURES_DIR)/*.png
	@rm -rf $(RESULTS_DIR)/*.csv $(RESULTS_DIR)/*.json
	@rm -rf $(RESULTS_DIR)/*_metadata.json
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@rm -rf .pytest_cache
	@echo "$(GREEN)✓ Generated files cleaned (data preserved)$(NC)"

clean-profile:
	@echo "$(YELLOW)Cleaning profiling outputs...$(NC)"
	@rm -f $(RESULTS_DIR)/simulation.prof
	@rm -f $(RESULTS_DIR)/complexity_*.pdf
	@rm -f $(RESULTS_DIR)/complexity_summary.csv
	@rm -f $(RESULTS_DIR)/simulation_warnings.log
	@rm -f $(RESULTS_DIR)/numerical_issues_summary.json
	@echo "$(GREEN)✓ Profiling files cleaned$(NC)"

clean-all: clean clean-profile
	@echo "$(RED)Removing ALL results including raw data...$(NC)"
	@rm -rf $(DATA_DIR)/*.pkl
	@rm -rf $(RESULTS_DIR)
	@echo "$(GREEN)✓ All files removed$(NC)"

# ==================================================
# ADVANCED TARGETS
# ==================================================

quick: setup
	@echo "$(YELLOW)Running quick test simulation...$(NC)"
	@cd src && $(PYTHON) -c " \
		from simulation import SimulationStudy; \
		from generators import TGenerator, TruncatedNormalGenerator; \
		print('Starting quick simulation (n_sim=5, T=100)...'); \
		study = SimulationStudy( \
			n_sim=5, K=2, d=5, T=100, q=2, h=0.5, tau=0.5, \
			err_generator=TGenerator(df=2.25, scale=0.7), \
			context_generator=TruncatedNormalGenerator(0, 1), \
			random_seed=1010 \
		); \
		results = study.run_simulation(); \
		print('✓ Quick simulation complete!') \
	"
	@echo "$(GREEN)✓ Quick simulation complete!$(NC)"

full: setup
	@echo "$(YELLOW)Running full simulation (this may take a while)...$(NC)"
	@cd src && $(PYTHON) -c " \
		from simulation import SimulationStudy; \
		from generators import TGenerator, TruncatedNormalGenerator; \
		print('Running full simulation (n_sim=100, T=2000)...'); \
		study = SimulationStudy( \
			n_sim=100, K=2, d=10, T=2000, q=2, h=0.5, tau=0.5, \
			err_generator=TGenerator(df=2.25, scale=0.7), \
			context_generator=TruncatedNormalGenerator(0, 1), \
			random_seed=$(SEED) \
		); \
		results = study.run_simulation(); \
		study.save_results(); \
		print('Full simulation complete!') \
	"
	@echo "$(GREEN)✓ Full simulation complete!$(NC)"

lint:
	@echo "$(YELLOW)Formatting code...$(NC)"
	-$(PYTHON) -m black $(SRC_DIR)
	-$(PYTHON) -m isort $(SRC_DIR)
	@echo "$(GREEN)✓ Code formatted!$(NC)"

coverage:
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	@if [ -d "$(TESTS_DIR)" ]; then \
		$(PYTHON) -m pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term; \
		echo "$(GREEN)✓ Coverage report in htmlcov/$(NC)"; \
	else \
		echo "$(YELLOW)No tests directory found$(NC)"; \
	fi

watch:
	@echo "$(YELLOW)Watching for changes...$(NC)"
	$(PYTHON) -m pytest_watch $(TESTS_DIR)

info:
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)Configuration Information$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"
	@echo "$(CYAN)Python:$(NC)          $$($(PYTHON) --version)"
	@echo "$(CYAN)System:$(NC)          $$(uname -srm)"
	@echo ""
	@echo "$(CYAN)Directories:$(NC)"
	@echo "  Source:        $(SRC_DIR)"
	@echo "  Results:       $(RESULTS_DIR)"
	@echo "  Figures:       $(FIGURES_DIR)"
	@echo "  Data:          $(DATA_DIR)"
	@echo "  Tests:         $(TESTS_DIR)"
	@echo ""
	@echo "$(CYAN)Simulation Parameters:$(NC)"
	@echo "  N_SIM:         $(N_SIM)"
	@echo "  K (arms):      $(K)"
	@echo "  D (context):   $(D)"
	@echo "  T (horizon):   $(T)"
	@echo "  Q (samples):   $(Q)"
	@echo "  TAU (quantile):$(TAU)"
	@echo "  SEED:          $(SEED)"
	@echo ""
	@echo "$(CYAN)Optimization:$(NC)"
	@echo "  N_JOBS:        $(N_JOBS)"
	@echo "  CPU cores:     $$($(PYTHON) -c 'import multiprocessing as mp; print(mp.cpu_count())')"
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"

status:
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)Status Report$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"
	@if [ -d $(DATA_DIR) ]; then \
		COUNT=$$(ls -1 $(DATA_DIR)/*.pkl 2>/dev/null | wc -l); \
		echo "$(CYAN)Data files:$(NC) $$COUNT"; \
	else \
		echo "$(YELLOW)No data directory$(NC)"; \
	fi
	@if [ -d $(FIGURES_DIR) ]; then \
		COUNT=$$(ls -1 $(FIGURES_DIR)/*.pdf 2>/dev/null | wc -l); \
		echo "$(CYAN)Figures:$(NC) $$COUNT"; \
	else \
		echo "$(YELLOW)No figures directory$(NC)"; \
	fi
	@if [ -f $(RESULTS_DIR)/simulation.prof ]; then \
		echo "$(GREEN)✓ Profile data available$(NC)"; \
	else \
		echo "$(YELLOW)✗ No profile data$(NC) (run: make profile)"; \
	fi
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"

# ==================================================
# COMBINED WORKFLOWS
# ==================================================

## quick-test: Quick end-to-end test
quick-test: setup test-all quick
	@echo "$(GREEN)✓ Quick test workflow complete!$(NC)"

## full-pipeline: Complete pipeline with profiling
full-pipeline: install setup test-all profile benchmark all
	@echo "$(GREEN)╔════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║  Full pipeline complete!              ║$(NC)"
	@echo "$(GREEN)╚════════════════════════════════════════╝$(NC)"

## optimize: Run optimization suite (profile + benchmark + stability)
optimize: profile benchmark stability-check
	@echo "$(GREEN)╔════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║  Optimization analysis complete!      ║$(NC)"
	@echo "$(GREEN)╚════════════════════════════════════════╝$(NC)"
	@echo "$(YELLOW)Results:$(NC)"
	@echo "  - Profiling: $(RESULTS_DIR)/simulation.prof"
	@echo "  - Complexity: $(RESULTS_DIR)/complexity_*.pdf"
	@echo "  - Stability: $(RESULTS_DIR)/simulation_warnings.log"