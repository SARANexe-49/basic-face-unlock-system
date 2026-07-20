"""
Face Unlock Prototype - Storage Tests
MIT License

Unit tests for encryption, decryption, and file storage functionality.
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

from lib.storage import SecureStorage, EncryptionError, get_secure_passphrase
from lib.face_core import FaceEmbedding


class TestSecureStorage:
    """Test cases for SecureStorage class."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        storage = SecureStorage(temp_dir)
        yield storage
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_enrollment_data(self):
        """Create sample enrollment data for testing."""
        # Create fake face embeddings
        fake_embeddings = [np.random.rand(128) for _ in range(5)]
        return FaceEmbedding.create_enrollment_data("test_user", fake_embeddings)
    
    def test_encryption_decryption_roundtrip(self, temp_storage, sample_enrollment_data):
        """Test that data can be encrypted and decrypted correctly."""
        passphrase = "test_passphrase_123"
        
        # Encrypt data
        encrypted_data = temp_storage.encrypt_data(sample_enrollment_data, passphrase)
        
        # Verify encrypted data is bytes
        assert isinstance(encrypted_data, bytes)
        assert len(encrypted_data) > 0
        
        # Decrypt data
        decrypted_data = temp_storage.decrypt_data(encrypted_data, passphrase)
        
        # Verify decrypted data matches original
        assert decrypted_data["name"] == sample_enrollment_data["name"]
        assert decrypted_data["created"] == sample_enrollment_data["created"]
        assert decrypted_data["num_samples"] == sample_enrollment_data["num_samples"]
        
        # Verify embedding data
        original_embedding = np.array(sample_enrollment_data["embedding"])
        decrypted_embedding = np.array(decrypted_data["embedding"])
        np.testing.assert_array_equal(original_embedding, decrypted_embedding)
    
    def test_wrong_passphrase_fails(self, temp_storage, sample_enrollment_data):
        """Test that wrong passphrase fails decryption."""
        correct_passphrase = "correct_passphrase"
        wrong_passphrase = "wrong_passphrase"
        
        # Encrypt with correct passphrase
        encrypted_data = temp_storage.encrypt_data(sample_enrollment_data, correct_passphrase)
        
        # Try to decrypt with wrong passphrase
        with pytest.raises(EncryptionError):
            temp_storage.decrypt_data(encrypted_data, wrong_passphrase)
    
    def test_save_and_load_enrollment(self, temp_storage, sample_enrollment_data):
        """Test saving and loading enrollment files."""
        name = "test_user"
        passphrase = "test_passphrase"
        
        # Save enrollment
        file_path = temp_storage.save_enrollment(name, sample_enrollment_data, passphrase)
        
        # Verify file was created
        assert os.path.exists(file_path)
        assert file_path.endswith(f"{name}.enc")
        
        # Load enrollment
        loaded_data = temp_storage.load_enrollment(name, passphrase)
        
        # Verify loaded data matches original
        assert loaded_data["name"] == sample_enrollment_data["name"]
        original_embedding = np.array(sample_enrollment_data["embedding"])
        loaded_embedding = np.array(loaded_data["embedding"])
        np.testing.assert_array_equal(original_embedding, loaded_embedding)
    
    def test_enrollment_exists(self, temp_storage, sample_enrollment_data):
        """Test enrollment existence checking."""
        name = "test_user"
        passphrase = "test_passphrase"
        
        # Initially should not exist
        assert not temp_storage.enrollment_exists(name)
        
        # Save enrollment
        temp_storage.save_enrollment(name, sample_enrollment_data, passphrase)
        
        # Now should exist
        assert temp_storage.enrollment_exists(name)
    
    def test_delete_enrollment(self, temp_storage, sample_enrollment_data):
        """Test enrollment deletion."""
        name = "test_user"
        passphrase = "test_passphrase"
        
        # Save enrollment
        file_path = temp_storage.save_enrollment(name, sample_enrollment_data, passphrase)
        assert os.path.exists(file_path)
        
        # Delete enrollment
        success = temp_storage.delete_enrollment(name)
        assert success
        assert not os.path.exists(file_path)
        
        # Try to delete non-existent enrollment
        success = temp_storage.delete_enrollment("nonexistent")
        assert not success
    
    def test_list_enrollments(self, temp_storage, sample_enrollment_data):
        """Test listing enrollments."""
        passphrase = "test_passphrase"
        
        # Initially empty
        enrollments = temp_storage.list_enrollments()
        assert enrollments == []
        
        # Add some enrollments
        names = ["alice", "bob", "charlie"]
        for name in names:
            temp_storage.save_enrollment(name, sample_enrollment_data, passphrase)
        
        # List should contain all names, sorted
        enrollments = temp_storage.list_enrollments()
        assert sorted(enrollments) == sorted(names)
    
    def test_invalid_names(self, temp_storage, sample_enrollment_data):
        """Test handling of invalid user names."""
        passphrase = "test_passphrase"
        invalid_names = ["", "user/with/slash", "user<with>brackets", "user|with|pipes"]
        
        for invalid_name in invalid_names:
            with pytest.raises(ValueError):
                temp_storage.save_enrollment(invalid_name, sample_enrollment_data, passphrase)
    
    def test_file_not_found(self, temp_storage):
        """Test loading non-existent enrollment."""
        with pytest.raises(FileNotFoundError):
            temp_storage.load_enrollment("nonexistent", "passphrase")
    
    def test_verify_passphrase(self, temp_storage, sample_enrollment_data):
        """Test passphrase verification."""
        name = "test_user"
        correct_passphrase = "correct_passphrase"
        wrong_passphrase = "wrong_passphrase"
        
        # Save enrollment
        temp_storage.save_enrollment(name, sample_enrollment_data, correct_passphrase)
        
        # Verify correct passphrase
        assert temp_storage.verify_passphrase(name, correct_passphrase)
        
        # Verify wrong passphrase fails
        assert not temp_storage.verify_passphrase(name, wrong_passphrase)
        
        # Verify non-existent user fails
        assert not temp_storage.verify_passphrase("nonexistent", correct_passphrase)


class TestEncryptionSecurity:
    """Test encryption security properties."""
    
    def test_different_salts_produce_different_ciphertext(self):
        """Test that same data with same passphrase produces different ciphertext."""
        storage = SecureStorage()
        data = {"test": "data", "number": 42}
        passphrase = "same_passphrase"
        
        # Encrypt same data twice
        encrypted1 = storage.encrypt_data(data, passphrase)
        encrypted2 = storage.encrypt_data(data, passphrase)
        
        # Should be different due to random salt and nonce
        assert encrypted1 != encrypted2
        
        # But both should decrypt to same data
        decrypted1 = storage.decrypt_data(encrypted1, passphrase)
        decrypted2 = storage.decrypt_data(encrypted2, passphrase)
        assert decrypted1 == decrypted2 == data
    
    def test_encryption_parameters(self):
        """Test that encryption uses correct parameters."""
        storage = SecureStorage()
        
        # Check constants
        assert storage.KEY_LENGTH == 32  # 256-bit key
        assert storage.NONCE_LENGTH == 12  # 96-bit nonce for GCM
        assert storage.SALT_LENGTH == 32  # 256-bit salt
        assert storage.PBKDF2_ITERATIONS == 100000  # 100k iterations
    
    def test_encrypted_data_structure(self):
        """Test structure of encrypted data."""
        storage = SecureStorage()
        data = {"test": "data"}
        passphrase = "test_passphrase"
        
        encrypted = storage.encrypt_data(data, passphrase)
        
        # Check minimum length (salt + nonce + tag + some ciphertext)
        min_length = storage.SALT_LENGTH + storage.NONCE_LENGTH + 16  # 16-byte tag
        assert len(encrypted) > min_length


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('lib.storage.safe_input')
    def test_get_secure_passphrase(self, mock_input):
        """Test secure passphrase input."""
        mock_input.return_value = "test_passphrase"
        
        passphrase = get_secure_passphrase("Enter passphrase: ")
        assert passphrase == "test_passphrase"
    
    @patch('lib.storage.safe_input')
    def test_get_secure_passphrase_with_confirmation(self, mock_input):
        """Test secure passphrase input with confirmation."""
        mock_input.side_effect = ["test_passphrase", "test_passphrase"]
        
        passphrase = get_secure_passphrase("Enter passphrase: ", confirm=True)
        assert passphrase == "test_passphrase"
    
    @patch('lib.storage.safe_input')
    def test_get_secure_passphrase_mismatch(self, mock_input):
        """Test passphrase confirmation mismatch."""
        mock_input.side_effect = ["passphrase1", "passphrase2"]
        
        with pytest.raises(ValueError, match="Passphrases do not match"):
            get_secure_passphrase("Enter passphrase: ", confirm=True)
    
    @patch('lib.storage.safe_input')
    def test_empty_passphrase(self, mock_input):
        """Test empty passphrase rejection."""
        mock_input.return_value = ""
        
        with pytest.raises(ValueError, match="Passphrase cannot be empty"):
            get_secure_passphrase("Enter passphrase: ")


if __name__ == "__main__":
    pytest.main([__file__])
