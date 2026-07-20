"""
Face Unlock Prototype - Face Matching Tests
MIT License

Unit tests for face embedding operations and matching logic.
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

from lib.face_core import FaceEmbedding


class TestFaceEmbedding:
    """Test cases for FaceEmbedding class."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample face embeddings for testing."""
        # Create 5 similar embeddings (simulating same person)
        base_embedding = np.random.rand(128)
        embeddings = []
        
        for i in range(5):
            # Add small random noise to base embedding
            noise = np.random.normal(0, 0.01, 128)
            embedding = base_embedding + noise
            embeddings.append(embedding)
        
        return embeddings
    
    @pytest.fixture
    def different_embeddings(self):
        """Create embeddings from different 'people'."""
        return [np.random.rand(128) for _ in range(3)]
    
    def test_average_embeddings(self, sample_embeddings):
        """Test embedding averaging calculation."""
        averaged = FaceEmbedding.average_embeddings(sample_embeddings)
        
        # Check output shape
        assert averaged.shape == (128,)
        
        # Check that average is close to manual calculation
        manual_average = np.mean(np.array(sample_embeddings), axis=0)
        np.testing.assert_array_almost_equal(averaged, manual_average)
    
    def test_average_embeddings_empty_list(self):
        """Test that empty embedding list raises error."""
        with pytest.raises(ValueError, match="Cannot average empty embedding list"):
            FaceEmbedding.average_embeddings([])
    
    def test_average_embeddings_single(self):
        """Test averaging single embedding."""
        single_embedding = np.random.rand(128)
        averaged = FaceEmbedding.average_embeddings([single_embedding])
        
        np.testing.assert_array_equal(averaged, single_embedding)
    
    def test_compute_distance(self):
        """Test distance computation between embeddings."""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        
        distance = FaceEmbedding.compute_distance(embedding1, embedding2)
        
        # Should be sqrt(2) for these vectors
        expected_distance = np.sqrt(2.0)
        assert abs(distance - expected_distance) < 1e-10
    
    def test_compute_distance_identical(self):
        """Test distance between identical embeddings."""
        embedding = np.random.rand(128)
        distance = FaceEmbedding.compute_distance(embedding, embedding)
        
        assert distance == 0.0
    
    def test_is_match_positive(self):
        """Test positive match detection."""
        embedding1 = np.array([1.0, 0.0])
        embedding2 = np.array([1.1, 0.0])  # Close to embedding1
        
        # With generous threshold
        assert FaceEmbedding.is_match(embedding1, embedding2, threshold=0.5)
        
        # With strict threshold
        assert not FaceEmbedding.is_match(embedding1, embedding2, threshold=0.05)
    
    def test_is_match_negative(self):
        """Test negative match detection."""
        embedding1 = np.array([1.0, 0.0])
        embedding2 = np.array([0.0, 1.0])  # Far from embedding1
        
        # Should not match even with generous threshold
        assert not FaceEmbedding.is_match(embedding1, embedding2, threshold=0.5)
    
    def test_threshold_behavior(self, sample_embeddings, different_embeddings):
        """Test matching behavior with different thresholds."""
        # Average similar embeddings
        averaged = FaceEmbedding.average_embeddings(sample_embeddings)
        
        # Test against one of the original embeddings (should be close)
        close_distance = FaceEmbedding.compute_distance(averaged, sample_embeddings[0])
        
        # Test against different embedding (should be far)
        far_distance = FaceEmbedding.compute_distance(averaged, different_embeddings[0])
        
        # Close distance should be smaller than far distance
        assert close_distance < far_distance
        
        # With appropriate threshold, close should match, far should not
        threshold = (close_distance + far_distance) / 2
        assert FaceEmbedding.is_match(averaged, sample_embeddings[0], threshold)
        assert not FaceEmbedding.is_match(averaged, different_embeddings[0], threshold)
    
    def test_create_enrollment_data(self, sample_embeddings):
        """Test enrollment data creation."""
        name = "test_user"
        enrollment_data = FaceEmbedding.create_enrollment_data(name, sample_embeddings)
        
        # Check required fields
        assert enrollment_data["name"] == name
        assert enrollment_data["num_samples"] == len(sample_embeddings)
        assert "created" in enrollment_data
        assert "embedding" in enrollment_data
        
        # Check embedding is list (for JSON serialization)
        assert isinstance(enrollment_data["embedding"], list)
        assert len(enrollment_data["embedding"]) == 128
        
        # Check that embedding matches manual average
        manual_average = np.mean(np.array(sample_embeddings), axis=0)
        enrollment_embedding = np.array(enrollment_data["embedding"])
        np.testing.assert_array_almost_equal(enrollment_embedding, manual_average)
    
    def test_create_enrollment_data_empty(self):
        """Test enrollment data creation with empty embeddings."""
        with pytest.raises(ValueError, match="Cannot create enrollment data from empty embeddings"):
            FaceEmbedding.create_enrollment_data("test", [])


class TestDistanceProperties:
    """Test mathematical properties of distance calculations."""
    
    def test_distance_symmetry(self):
        """Test that distance(a, b) == distance(b, a)."""
        embedding1 = np.random.rand(128)
        embedding2 = np.random.rand(128)
        
        distance1 = FaceEmbedding.compute_distance(embedding1, embedding2)
        distance2 = FaceEmbedding.compute_distance(embedding2, embedding1)
        
        assert distance1 == distance2
    
    def test_distance_triangle_inequality(self):
        """Test triangle inequality: d(a,c) <= d(a,b) + d(b,c)."""
        embedding_a = np.random.rand(128)
        embedding_b = np.random.rand(128)
        embedding_c = np.random.rand(128)
        
        dist_ac = FaceEmbedding.compute_distance(embedding_a, embedding_c)
        dist_ab = FaceEmbedding.compute_distance(embedding_a, embedding_b)
        dist_bc = FaceEmbedding.compute_distance(embedding_b, embedding_c)
        
        assert dist_ac <= dist_ab + dist_bc + 1e-10  # Small epsilon for floating point
    
    def test_distance_non_negative(self):
        """Test that distances are always non-negative."""
        embedding1 = np.random.rand(128)
        embedding2 = np.random.rand(128)
        
        distance = FaceEmbedding.compute_distance(embedding1, embedding2)
        assert distance >= 0.0


class TestThresholdRecommendations:
    """Test threshold recommendations with synthetic data."""
    
    def test_threshold_ranges(self):
        """Test behavior with different threshold ranges."""
        # Create two distinct embeddings
        embedding1 = np.zeros(128)
        embedding2 = np.ones(128)
        
        distance = FaceEmbedding.compute_distance(embedding1, embedding2)
        
        # Very strict threshold - should not match
        assert not FaceEmbedding.is_match(embedding1, embedding2, 0.1)
        
        # Very loose threshold - should match
        assert FaceEmbedding.is_match(embedding1, embedding2, distance + 1.0)
        
        # Threshold exactly at distance - should not match (< not <=)
        assert not FaceEmbedding.is_match(embedding1, embedding2, distance)
        
        # Threshold slightly above distance - should match
        assert FaceEmbedding.is_match(embedding1, embedding2, distance + 0.001)


if __name__ == "__main__":
    pytest.main([__file__])
