# Makefile for Bandit Algorithm Simulation Pipeline
# ==================================================

# Python interpreter
PYTHON := python

# Directories
SRC_DIR := src
RESULTS_DIR := results
FIGURES_DIR := $(RESULTS_DIR)/figures
DATA_DIR := $(RESULTS_DIR)/data
TESTS_DIR := tests

# Main scripts
MAIN_SCRIPT := $(SRC_DIR)/main.py
ANALYZE_SCRIPT := $(SRC_DIR)/analyze_results.py

# Configuration
N_SIM := 50
K := 2
D := 10
T := 1000
Q := 2
TAU := 0.5
SEED := 1010

# Color output for better readability
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Phony targets (targets that are not files)
.PHONY: all help setup clean clean-all simulate analyze figures test check install

# ==================================================
# Main Pipeline Targets
# ==================================================

## all: Run complete simulation pipeline and generate all outputs
all: setup simulate analyze figures
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✓ Complete pipeline finished successfully!$(NC)"
	@echo "$(GREEN)════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)Results saved in: $(RESULTS_DIR)$(NC)"
	@echo "$(BLUE)Figures saved in: $(FIGURES_DIR)$(NC)"
	@echo "$(BLUE)Data saved in: $(DATA_DIR)$(NC)"

## simulate: Run simulations and save raw results
simulate: setup
	@echo "$(YELLOW)Running simulations...$(NC)"
	@echo "$(BLUE)Parameters: N_SIM=$(N_SIM), K=$(K), D=$(D), T=$(T)$(NC)"
	$(PYTHON) $(MAIN_SCRIPT) --n_sim $(N_SIM) --K $(K) --d $(D) --T $(T) --q $(Q) --tau $(TAU) --seed $(SEED) --save
	@echo "$(GREEN)✓ Simulations complete! Results saved in $(DATA_DIR)$(NC)"

## analyze: Process raw results and generate summary statistics
analyze: 
	@echo "$(YELLOW)Analyzing results...$(NC)"
	@if [ ! -d "$(DATA_DIR)" ] || [ -z "$$(ls -A $(DATA_DIR) 2>/dev/null)" ]; then \
		echo "$(RED)Error: No simulation data found in $(DATA_DIR)$(NC)"; \
		echo "$(YELLOW)Please run 'make simulate' first$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(ANALYZE_SCRIPT) --data_dir $(DATA_DIR) --output_dir $(RESULTS_DIR)
	@echo "$(GREEN)✓ Analysis complete! Summary statistics saved in $(RESULTS_DIR)$(NC)"

## figures: Create all visualizations
figures:
	@echo "$(YELLOW)Generating figures...$(NC)"
	@if [ ! -d "$(DATA_DIR)" ] || [ -z "$$(ls -A $(DATA_DIR) 2>/dev/null)" ]; then \
		echo "$(RED)Error: No simulation data found in $(DATA_DIR)$(NC)"; \
		echo "$(YELLOW)Please run 'make simulate' first$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(ANALYZE_SCRIPT) --data_dir $(DATA_DIR) --output_dir $(FIGURES_DIR) --figures_only
	@echo "$(GREEN)✓ Figures generated! Saved in $(FIGURES_DIR)$(NC)"

## test: Run test suite
test:
	@echo "$(YELLOW)Running test suite...$(NC)"
	$(PYTHON) -m pytest $(TESTS_DIR) -v --color=yes
	@echo "$(GREEN)✓ All tests passed!$(NC)"

## check: Run code quality checks (linting and type checking)
check:
	@echo "$(YELLOW)Running code quality checks...$(NC)"
	@echo "$(BLUE)Checking code style with flake8...$(NC)"
	-$(PYTHON) -m flake8 $(SRC_DIR) --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "$(BLUE)Checking imports with isort...$(NC)"
	-$(PYTHON) -m isort $(SRC_DIR) --check-only
	@echo "$(GREEN)✓ Code quality checks complete!$(NC)"

# ==================================================
# Setup and Installation Targets
# ==================================================

## setup: Create necessary directories
setup:
	@echo "$(YELLOW)Setting up directories...$(NC)"
	@mkdir -p $(RESULTS_DIR)
	@mkdir -p $(FIGURES_DIR)
	@mkdir -p $(DATA_DIR)
	@echo "$(GREEN)✓ Directories created$(NC)"

## install: Install required Python packages
install:
	@echo "$(YELLOW)Installing required packages...$(NC)"
	$(PYTHON) -m pip install -r requirements.txt
	@echo "$(GREEN)✓ Packages installed$(NC)"

# ==================================================
# Cleaning Targets
# ==================================================

## clean: Remove generated files (keep raw simulation data)
clean:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	@rm -rf $(FIGURES_DIR)/*.pdf
	@rm -rf $(FIGURES_DIR)/*.png
	@rm -rf $(RESULTS_DIR)/*.csv
	@rm -rf $(RESULTS_DIR)/*.json
	@rm -rf $(RESULTS_DIR)/*_metadata.json
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned generated files (raw data preserved)$(NC)"

## clean-all: Remove all generated files including raw simulation data
clean-all: clean
	@echo "$(RED)Removing all results including raw data...$(NC)"
	@rm -rf $(DATA_DIR)/*.pkl
	@rm -rf $(RESULTS_DIR)
	@echo "$(GREEN)✓ All generated files removed$(NC)"

# ==================================================
# Advanced Targets
# ==================================================

## quick: Run a quick test simulation (small parameters)
quick: setup
	@echo "$(YELLOW)Running quick test simulation...$(NC)"
	$(PYTHON) $(MAIN_SCRIPT) --n_sim 5 --K 2 --d 5 --T 100 --q 2 --tau 0.5 --seed $(SEED) --save
	@echo "$(GREEN)✓ Quick simulation complete!$(NC)"

## full: Run full simulation with publication-quality parameters
full: setup
	@echo "$(YELLOW)Running full simulation (this may take a while)...$(NC)"
	$(PYTHON) $(MAIN_SCRIPT) --n_sim 100 --K 2 --d 10 --T 2000 --q 2 --tau 0.5 --seed $(SEED) --save
	@echo "$(GREEN)✓ Full simulation complete!$(NC)"

## lint: Run comprehensive code formatting and linting
lint:
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(PYTHON) -m black $(SRC_DIR)
	$(PYTHON) -m isort $(SRC_DIR)
	@echo "$(GREEN)✓ Code formatted!$(NC)"

## coverage: Run tests with coverage report
coverage:
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

## watch: Watch for file changes and run tests automatically
watch:
	@echo "$(YELLOW)Watching for changes...$(NC)"
	$(PYTHON) -m pytest_watch $(TESTS_DIR)

## info: Display current configuration
info:
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)Current Configuration:$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"
	@echo "Python:          $$($(PYTHON) --version)"
	@echo "Source Dir:      $(SRC_DIR)"
	@echo "Results Dir:     $(RESULTS_DIR)"
	@echo "Figures Dir:     $(FIGURES_DIR)"
	@echo "Data Dir:        $(DATA_DIR)"
	@echo "Tests Dir:       $(TESTS_DIR)"
	@echo ""
	@echo "Simulation Parameters:"
	@echo "  N_SIM:         $(N_SIM)"
	@echo "  K (arms):      $(K)"
	@echo "  D (context):   $(D)"
	@echo "  T (horizon):   $(T)"
	@echo "  Q (samples):   $(Q)"
	@echo "  TAU (quantile):$(TAU)"
	@echo "  SEED:          $(SEED)"
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"

# ==================================================
# Help Target
# ==================================================

## help: Show this help message
help:
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)Bandit Algorithm Simulation Pipeline$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(GREEN)Required targets (as specified):$(NC)"
	@echo "  make all       - Run complete pipeline and generate all outputs"
	@echo "  make simulate  - Run simulations and save raw results"
	@echo "  make analyze   - Process raw results and generate summary statistics"
	@echo "  make figures   - Create all visualizations"
	@echo "  make clean     - Remove generated files"
	@echo "  make test      - Run test suite"
	@echo ""
	@echo "$(GREEN)Setup targets:$(NC)"
	@echo "  make setup     - Create necessary directories"
	@echo "  make install   - Install required packages"
	@echo ""
	@echo "$(GREEN)Advanced targets:$(NC)"
	@echo "  make quick     - Quick test simulation (small parameters)"
	@echo "  make full      - Full publication-quality simulation"
	@echo "  make check     - Run code quality checks"
	@echo "  make lint      - Format code with black and isort"
	@echo "  make coverage  - Run tests with coverage report"
	@echo "  make clean-all - Remove ALL generated files (including raw data)"
	@echo "  make info      - Display current configuration"
	@echo ""
	@echo "$(YELLOW)Usage examples:$(NC)"
	@echo "  make all                     # Run entire pipeline"
	@echo "  make simulate T=500 N_SIM=10 # Custom parameters"
	@echo "  make test                    # Run all tests"
	@echo "  make figures                 # Only regenerate figures"
	@echo "  make clean && make all       # Clean and rerun everything"
	@echo ""
	@echo "$(BLUE)════════════════════════════════════════════════$(NC)"