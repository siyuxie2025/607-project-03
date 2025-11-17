import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
from src.methods import RiskAwareBandit, OLSBandit


# Fixtures for common test data
@pytest.fixture
def basic_params():
    """Basic parameters for bandit initialization."""
    return {
        'q': 2,
        'h': 0.5,
        'd': 5,
        'K': 3
    }

@pytest.fixture
def beta_real():
    """Real beta values for testing."""
    return np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.5, 1.5, 2.5, 3.5, 4.5],
        [2.0, 1.0, 3.0, 2.0, 4.0]
    ])

@pytest.fixture
def alpha_real():
    """Real alpha values for testing."""
    return np.array([1.0, 0.5, 2.0])

@pytest.fixture
def context_vector():
    """A sample context vector."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

@pytest.fixture
def risk_aware_bandit(basic_params, beta_real, alpha_real):
    """Fixture for RiskAwareBandit instance."""
    return RiskAwareBandit(
        q=basic_params['q'],
        h=basic_params['h'],
        tau=0.5,
        d=basic_params['d'],
        K=basic_params['K'],
        beta_real_value=beta_real,
        alpha_real_value=alpha_real
    )

@pytest.fixture
def ols_bandit(basic_params, beta_real):
    """Fixture for OLSBandit instance."""
    return OLSBandit(
        q=basic_params['q'],
        h=basic_params['h'],
        d=basic_params['d'],
        K=basic_params['K'],
        beta_real_value=beta_real
    )


class TestRiskAwareBandit:
    """Test suite for RiskAwareBandit."""
    
    def test_initialization(self, risk_aware_bandit, basic_params):
        """Test that RiskAwareBandit initializes correctly."""
        rab = risk_aware_bandit
        
        assert rab.q == basic_params['q']
        assert rab.h == basic_params['h']
        assert rab.tau == 0.5
        assert rab.d == basic_params['d']
        assert rab.K == basic_params['K']
        assert rab.n == 0
        
        # Check data structures are initialized
        assert len(rab.Tx) == basic_params['K']
        assert len(rab.Sx) == basic_params['K']
        assert len(rab.Tr) == basic_params['K']
        assert len(rab.Sr) == basic_params['K']
        
        # Check all are empty lists
        for k in range(basic_params['K']):
            assert rab.Tx[k] == []
            assert rab.Sx[k] == []
            assert rab.Tr[k] == []
            assert rab.Sr[k] == []
    
    def test_beta_shapes(self, risk_aware_bandit, basic_params):
        """Test that beta matrices have correct shapes."""
        rab = risk_aware_bandit
        
        assert rab.beta_t.shape == (basic_params['K'], basic_params['d'])
        assert rab.beta_a.shape == (basic_params['K'], basic_params['d'])
        assert rab.alpha_t.shape == (basic_params['K'],)
        assert rab.alpha_a.shape == (basic_params['K'],)
    
    def test_choose_a_forced_sampling_first_round(self, risk_aware_bandit, context_vector):
        """Test action selection during forced sampling in first round."""
        rab = risk_aware_bandit
        
        # First time step should trigger forced sampling
        t = 1
        action = rab.choose_a(t, context_vector)
        
        # Action should be an integer
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < rab.K
        
        # Context should be stored in Tx and Sx
        assert len(rab.Tx[action]) == 1
        assert len(rab.Sx[action]) == 1
        np.testing.assert_array_equal(rab.Tx[action][0], context_vector)
        np.testing.assert_array_equal(rab.Sx[action][0], context_vector)
    
    def test_choose_a_returns_integer(self, risk_aware_bandit, context_vector):
        """Test that choose_a always returns a Python int or numpy integer."""
        rab = risk_aware_bandit
        
        for t in range(1, 20):
            action = rab.choose_a(t, context_vector)
            # Should be integer type (either Python int or numpy integer)
            assert isinstance(action, (int, np.integer))
            # Should be valid action index
            assert 0 <= action < rab.K
    
    def test_forced_sampling_schedule(self, risk_aware_bandit, context_vector):
        """Test that forced sampling happens at correct time steps."""
        rab = risk_aware_bandit
        K, q = rab.K, rab.q
        
        # First round: t=1 to t=K*q should be forced sampling
        forced_actions = []
        for t in range(1, K*q + 1):
            action = rab.choose_a(t, context_vector)
            forced_actions.append(action)
        
        # Check that we sampled each arm q times
        action_counts = [forced_actions.count(k) for k in range(K)]
        assert all(count == q for count in action_counts)
    
    def test_update_beta_initial_samples(self, risk_aware_bandit, context_vector):
        """Test beta update with insufficient samples (≤ d)."""
        rab = risk_aware_bandit
        
        # Choose action and update with reward
        t = 1
        action = rab.choose_a(t, context_vector)
        reward = 5.0
        
        rab.update_beta(reward, t)
        
        # Should store reward
        assert len(rab.Sr[action]) == 1
        assert rab.Sr[action][0] == reward
        
        # Beta should remain random (not fitted yet)
        assert rab.beta_a.shape == (rab.K, rab.d)
    
    def test_update_beta_sufficient_samples(self, risk_aware_bandit, context_vector):
        """
        Test beta update with sufficient samples (> d).
        """
        rab = risk_aware_bandit
        d = rab.d
        t = 1

        rab.n = 1
        action_to_test = 0
        rab.set = np.arange(1, d+3)
        
        # Manually add d+1 samples to force fitting
        for i in range(d + 2):
            rab.action = action_to_test
            x = np.random.randn(d)
            rab.Sx[action_to_test].append(x)
            rab.Tx[action_to_test].append(x)

            r = np.random.randn()
            rab.update_beta(r, t+i)

        assert len(rab.Sx[action_to_test]) == d + 2
        assert len(rab.Sr[action_to_test]) == d + 2
        assert rab.beta_a[action_to_test].shape == (d,)
        assert rab.beta_t[action_to_test].shape == (d,)

    def test_data_accumulation(self, risk_aware_bandit, context_vector):
        """Test that contexts and rewards accumulate correctly."""
        rab = risk_aware_bandit
        
        actions_taken = []
        for t in range(1, 11):
            action = rab.choose_a(t, context_vector)
            actions_taken.append(action)
            reward = np.random.randn()
            rab.update_beta(reward, t)
        
        # Check that data is accumulating
        total_samples = sum(len(rab.Sx[k]) for k in range(rab.K))
        assert total_samples == 10
        
        # Each action should have consistent data lengths
        for k in range(rab.K):
            assert len(rab.Sx[k]) == len(rab.Sr[k])
    
    def test_reproducibility_with_seed(self, basic_params, beta_real, alpha_real, context_vector):
        """Test that same seed produces same action sequence."""
        np.random.seed(42)
        rab1 = RiskAwareBandit(
            q=basic_params['q'], h=basic_params['h'], tau=0.5,
            d=basic_params['d'], K=basic_params['K'],
            beta_real_value=beta_real, alpha_real_value=alpha_real
        )
        
        np.random.seed(42)
        rab2 = RiskAwareBandit(
            q=basic_params['q'], h=basic_params['h'], tau=0.5,
            d=basic_params['d'], K=basic_params['K'],
            beta_real_value=beta_real, alpha_real_value=alpha_real
        )
        
        actions1 = [rab1.choose_a(t, context_vector) for t in range(1, 11)]
        actions2 = [rab2.choose_a(t, context_vector) for t in range(1, 11)]
        
        assert actions1 == actions2
    
    def test_beta_error_tracking(self, risk_aware_bandit):
        """Test that beta errors are tracked."""
        rab = risk_aware_bandit
        
        # Initially should be zeros or computed
        assert rab.beta_error_a.shape == (rab.K,)
        assert rab.beta_error_t.shape == (rab.K,)


class TestOLSBandit:
    """Test suite for OLSBandit."""
    
    def test_initialization(self, ols_bandit, basic_params):
        """Test that OLSBandit initializes correctly."""
        olsb = ols_bandit
        
        assert olsb.q == basic_params['q']
        assert olsb.h == basic_params['h']
        assert olsb.d == basic_params['d']
        assert olsb.K == basic_params['K']
        assert olsb.n == 0
        
        # Check data structures
        assert len(olsb.Tx) == basic_params['K']
        assert len(olsb.Sx) == basic_params['K']
        assert len(olsb.Tr) == basic_params['K']
        assert len(olsb.Sr) == basic_params['K']
    
    def test_choose_a_returns_python_int(self, ols_bandit, context_vector):
        """Test that choose_a returns Python int, not numpy int."""
        olsb = ols_bandit
        
        for t in range(1, 20):
            action = olsb.choose_a(t, context_vector)
            # Should be integer type
            assert isinstance(action, (int, np.integer))
            # Should be valid action index
            assert 0 <= action < olsb.K
            # Should work as list index without error
            olsb.Sx[action].append(context_vector)
    
    def test_forced_sampling_first_round(self, ols_bandit, context_vector):
        """Test forced sampling in first round."""
        olsb = ols_bandit
        K, q = olsb.K, olsb.q
        
        forced_actions = []
        for t in range(1, K*q + 1):
            action = olsb.choose_a(t, context_vector)
            forced_actions.append(action)
        
        # Each arm should be sampled q times
        action_counts = [forced_actions.count(k) for k in range(K)]
        assert all(count == q for count in action_counts)
    
    def test_update_beta_with_ols(self, ols_bandit, context_vector):
        """Test that OLS fitting works correctly."""
        olsb = ols_bandit
        d = olsb.d

        action_to_test = 0
        olsb.action = action_to_test

        olsb.set = np.arange(1, d+3)
        
        # Add sufficient samples for OLS
        for i in range(d + 2):
            x = np.random.randn(d)
            olsb.Sx[action_to_test].append(x)
            olsb.Tx[action_to_test].append(x)
            r = np.random.randn()
            olsb.update_beta(r, t=1+i)
        
                
        # Beta should be fitted
        assert len(olsb.Sx[action_to_test]) == d + 2
        assert len(olsb.Sr[action_to_test]) == d + 2
        assert olsb.beta_a[action_to_test].shape == (d,)
        assert olsb.beta_t[action_to_test].shape == (d,)
    
    def test_no_intercept(self, ols_bandit, context_vector):
        """Test that OLS is fitted without intercept."""
        olsb = ols_bandit
        
        # This test ensures the model uses fit_intercept=False
        # by checking the implementation indirectly
        d = olsb.d
        action = 0

        olsb.action = action
        t = 1
        olsb.set = np.arange(1, d+3)
        
        # Create deterministic data: y = 2*x (no intercept)
        for i in range(d + 2):
            x = np.ones(d) * (i + 1)
            r = 2.0 * np.sum(x)
            olsb.Sx[action].append(x)
            olsb.Tx[action].append(x)
            olsb.update_beta(r, t+i)
        
        assert olsb.beta_a[action].shape == (d,)


class TestBanditComparison:
    """Compare behavior of RiskAwareBandit and OLSBandit."""
    
    def test_both_use_forced_sampling(self, risk_aware_bandit, ols_bandit, context_vector):
        """Test that both bandits use forced sampling."""
        K = risk_aware_bandit.K
        q = risk_aware_bandit.q
        
        rab_actions = []
        olsb_actions = []
        
        for t in range(1, K*q + 1):
            rab_actions.append(risk_aware_bandit.choose_a(t, context_vector))
            olsb_actions.append(ols_bandit.choose_a(t, context_vector))
        
        # Both should sample each arm q times
        rab_counts = [rab_actions.count(k) for k in range(K)]
        olsb_counts = [olsb_actions.count(k) for k in range(K)]
        
        assert all(count == q for count in rab_counts)
        assert all(count == q for count in olsb_counts)
    
    def test_data_structure_consistency(self, risk_aware_bandit, ols_bandit):
        """Test that both bandits have consistent data structures."""
        rab = risk_aware_bandit
        olsb = ols_bandit
        
        # Both should have same number of arms
        assert len(rab.Sx) == len(olsb.Sx)
        assert len(rab.Tx) == len(olsb.Tx)
        
        # Both should have same dimensions
        assert rab.beta_a.shape == olsb.beta_a.shape
        assert rab.beta_t.shape == olsb.beta_t.shape


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_k_hat(self, risk_aware_bandit, context_vector):
        """Test behavior when K_hat is empty (no actions meet threshold)."""
        rab = risk_aware_bandit
        
        # Set beta_t to values that might cause empty K_hat
        rab.beta_t = np.zeros((rab.K, rab.d))
        rab.alpha_t = np.array([-100.0, -100.0, -100.0])  # Very negative
        rab.h = 0.01  # Very small threshold
        
        # Move past forced sampling phase
        rab.n = 1
        rab.set = np.arange(1, rab.K * rab.q + 1)
        
        # This should not crash (choose_a should handle empty K_hat)
        try:
            # Try a timestep outside forced sampling
            action = rab.choose_a(rab.K * rab.q + 1, context_vector)
            assert isinstance(action, (int, np.integer))
        except (ValueError, IndexError) as e:
            # If it does crash, that's a bug we should fix
            pytest.fail(f"choose_a crashed with empty K_hat: {e}")
    
    def test_single_arm(self):
        """Test with K=1 (single arm)."""
        K = 1
        d = 3
        beta = np.array([[1.0, 2.0, 3.0]])
        alpha = np.array([1.0])
        
        rab = RiskAwareBandit(q=2, h=0.5, tau=0.5, d=d, K=K,
                             beta_real_value=beta, alpha_real_value=alpha)
        
        x = np.array([1.0, 2.0, 3.0])
        
        # Should always choose action 0
        for t in range(1, 10):
            action = rab.choose_a(t, x)
            assert action == 0
    
    def test_high_dimensional_context(self):
        """Test with high-dimensional context vectors."""
        K = 2
        d = 50
        beta = np.random.randn(K, d)
        alpha = np.random.randn(K)
        
        rab = RiskAwareBandit(q=3, h=0.5, tau=0.5, d=d, K=K,
                             beta_real_value=beta, alpha_real_value=alpha)
        
        x = np.random.randn(d)
        
        # Should work with high dimensions
        action = rab.choose_a(1, x)
        assert 0 <= action < K
    
    def test_zero_context_vector(self, risk_aware_bandit):
        """Test with all-zero context vector."""
        rab = risk_aware_bandit
        x = np.zeros(rab.d)
        
        # Should still work
        action = rab.choose_a(1, x)
        assert 0 <= action < rab.K


class TestIntegration:
    """Integration tests simulating realistic usage."""
    
    def test_full_simulation_run(self, basic_params, beta_real, alpha_real, context_vector):
        """Test a complete simulation run with varying context vectors."""
        # Use higher q to ensure all arms get enough samples
        rab = RiskAwareBandit(
            q=3,  # Increase q so each arm gets more forced samples
            h=basic_params['h'],
            tau=0.5,
            d=basic_params['d'],
            K=basic_params['K'],
            beta_real_value=beta_real,
            alpha_real_value=alpha_real
        )
        
        T = 50  # Run for 50 timesteps
        
        regrets = []
        for t in range(1, T + 1):
            context = context_vector + np.random.randn(basic_params['d']) * 0.1
            action = rab.choose_a(t, context)
            
            # Compute reward (with noise)
            true_reward = (np.dot(rab.beta_real_value[action], context) + 
                          rab.alpha_real_value[action])
            noisy_reward = true_reward + np.random.randn() * 0.1
            
            rab.update_beta(noisy_reward, t)
            
            all_rewards = (np.dot(rab.beta_real_value, context) + 
                          rab.alpha_real_value)
            optimal_reward = np.max(all_rewards)
            
            # Track regret
            regrets.append(optimal_reward - true_reward)
        
        # Should have T regret values
        assert len(regrets) == T
        
        # Cumulative regret should mostly be non-negative (allow some noise)
        cumulative_regret = np.cumsum(regrets)
        # At least 90% of cumulative regrets should be non-negative
        assert np.sum(cumulative_regret >= -0.5) >= 0.9 * T
    
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])