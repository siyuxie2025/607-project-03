import pytest
from src.generators import DataGenerator, NormalGenerator, UniformGenerator, TGenerator
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


RANDOM_SEED = 42

class TestNormalGenerator:
    """Test suite for NormalGenerator."""
    
    def test_initialization(self):
        """Test that generator initializes with correct parameters."""
        gen = NormalGenerator(mean=5.0, std=2.0)
        assert gen.mean == 5.0
        assert gen.std == 2.0
    
    def test_default_initialization(self):
        """Test default parameters."""
        gen = NormalGenerator()
        assert gen.mean == 0.0
        assert gen.std == 1.0
    
    def test_generate_shape(self):
        """Test that generate returns correct shape."""
        gen = NormalGenerator(mean=0.0, std=1.0)
        data = gen.generate(100)
        assert data.shape == (100,)
        assert isinstance(data, np.ndarray)
    
    def test_generate_different_sizes(self):
        """Test generation with various sample sizes."""
        gen = NormalGenerator()
        for n in [1, 10, 100, 1000]:
            data = gen.generate(n)
            assert len(data) == n
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different results."""
        gen = NormalGenerator()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)
        
        data1 = gen.generate(100, rng=rng1)
        data2 = gen.generate(100, rng=rng2)
        
        assert not np.allclose(data1, data2)
    
    def test_statistical_properties(self):
        """Test that generated data has approximately correct mean and std."""
        gen = NormalGenerator(mean=5.0, std=2.0)
        rng = np.random.default_rng(42)
        data = gen.generate(10000, rng=rng)
        
        # With large sample, should be close to true parameters
        assert np.abs(np.mean(data) - 5.0) < 0.1
        assert np.abs(np.std(data, ddof=1) - 2.0) < 0.1
    
    def test_name_property(self):
        """Test that name property returns correct string."""
        gen = NormalGenerator(mean=5.0, std=2.0)
        assert gen.name == 'Normal(mean=5.0, std=2.0)'
    
    def test_no_rng_provided(self):
        """Test that generator works without explicit rng."""
        gen = NormalGenerator()
        data = gen.generate(100)
        assert len(data) == 100
        assert isinstance(data, np.ndarray)

class TestUniformGenerator:
    """Test suite for UniformGenerator."""
    
    def test_initialization(self):
        """Test that generator initializes with correct parameters."""
        gen = UniformGenerator(low=0.0, high=10.0)
        assert gen.low == 0.0
        assert gen.high == 10.0
    
    def test_default_initialization(self):
        """Test default parameters."""
        gen = UniformGenerator()
        assert gen.low == 0.0
        assert gen.high == 2.0
    
    def test_generate_shape(self):
        """Test that generate returns correct shape."""
        gen = UniformGenerator(low=0.0, high=1.0)
        data = gen.generate(100)
        assert data.shape == (100,)
    
    def test_values_in_range(self):
        """Test that all generated values are within bounds."""
        gen = UniformGenerator(low=0.0, high=10.0)
        rng = np.random.default_rng(42)
        data = gen.generate(1000, rng=rng)
        
        assert np.all(data >= 0.0)
        assert np.all(data <= 10.0)
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        gen = UniformGenerator(low=0.0, high=1.0)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        data1 = gen.generate(100, rng=rng1)
        data2 = gen.generate(100, rng=rng2)
        
        np.testing.assert_array_equal(data1, data2)
    
    def test_statistical_properties(self):
        """Test that generated data has approximately correct mean."""
        gen = UniformGenerator(low=0.0, high=10.0)
        rng = np.random.default_rng(42)
        data = gen.generate(10000, rng=rng)
        
        # Mean of Uniform(0, 10) should be 5
        assert np.abs(np.mean(data) - 5.0) < 0.1
        # Variance of Uniform(0, 10) should be (10-0)^2/12 ≈ 8.33
        expected_var = (10.0 - 0.0) ** 2 / 12
        assert np.abs(np.var(data, ddof=1) - expected_var) < 0.5
    
    def test_name_property(self):
        """Test that name property returns correct string."""
        gen = UniformGenerator(low=0.0, high=10.0)
        assert gen.name == 'Uniform(low=0.0, high=10.0)'
    
    def test_null_value_property(self):
        """Test that null_value returns the midpoint."""
        gen = UniformGenerator(low=0.0, high=10.0)
        assert gen.null_value == 5.0


class TestTGenerator:
    """Test suite for TGenerator."""
    
    def test_initialization(self):
        """Test that generator initializes with correct parameters."""
        gen = TGenerator(df=5)
        assert gen.df == 5
    
    def test_generate_shape(self):
        """Test that generate returns correct shape."""
        gen = TGenerator(df=5)
        data = gen.generate(100)
        assert data.shape == (100,)
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        gen = TGenerator(df=5)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        data1 = gen.generate(100, rng=rng1)
        data2 = gen.generate(100, rng=rng2)
        
        np.testing.assert_array_equal(data1, data2)
    
    def test_statistical_properties(self):
        """Test that generated data has approximately correct mean."""
        gen = TGenerator(df=10)
        rng = np.random.default_rng(42)
        data = gen.generate(10000, rng=rng)
        
        # t-distribution with df > 1 has mean 0
        assert np.abs(np.mean(data)) < 0.1
    
    def test_name_property(self):
        """Test that name property returns correct string."""
        gen = TGenerator(df=5)
        assert gen.name == 't(df=5)'
    
    def test_null_value_property(self):
        """Test that null_value returns 0."""
        gen = TGenerator(df=5)
        assert gen.null_value == 0.0
    
    def test_different_degrees_of_freedom(self):
        """Test generation with various df values."""
        for df in [1, 3, 10, 30]:
            gen = TGenerator(df=df)
            data = gen.generate(100)
            assert len(data) == 100
