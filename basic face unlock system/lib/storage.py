"""
Face Unlock Prototype - Secure Storage
MIT License

Handles encryption, decryption, and file I/O for face embeddings.
Uses AES-GCM with PBKDF2 key derivation for secure storage.
"""

import os
import json
import secrets
from typing import Dict, Any, Optional
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
from Crypto.Random import get_random_bytes

from .utils import ensure_directory, validate_name, get_timestamp


class EncryptionError(Exception):
    """Custom exception for encryption/decryption errors."""
    pass


class SecureStorage:
    """Handles secure encryption and storage of face embeddings."""
    
    # Encryption parameters
    KEY_LENGTH = 32  # 256-bit key
    NONCE_LENGTH = 12  # 96-bit nonce for GCM
    SALT_LENGTH = 32  # 256-bit salt for PBKDF2
    PBKDF2_ITERATIONS = 100000  # 100k iterations
    
    def __init__(self, storage_dir: str = "face_data"):
        """
        Initialize secure storage.
        
        Args:
            storage_dir: Directory to store encrypted files
        """
        self.storage_dir = storage_dir
        ensure_directory(storage_dir)
    
    def _derive_key(self, passphrase: str, salt: bytes) -> bytes:
        """
        Derive encryption key from passphrase using PBKDF2.
        
        Args:
            passphrase: User passphrase
            salt: Random salt bytes
            
        Returns:
            Derived key bytes
        """
        return PBKDF2(
            passphrase.encode('utf-8'),
            salt,
            dkLen=self.KEY_LENGTH,
            count=self.PBKDF2_ITERATIONS,
            hmac_hash_module=SHA256
        )
    
    def encrypt_data(self, data: Dict[str, Any], passphrase: str) -> bytes:
        """
        Encrypt data using AES-GCM.
        
        Args:
            data: Data dictionary to encrypt
            passphrase: Encryption passphrase
            
        Returns:
            Encrypted data bytes (salt + nonce + tag + ciphertext)
        """
        try:
            # Serialize data to JSON
            json_data = json.dumps(data, separators=(',', ':')).encode('utf-8')
            
            # Generate random salt and nonce
            salt = get_random_bytes(self.SALT_LENGTH)
            nonce = get_random_bytes(self.NONCE_LENGTH)
            
            # Derive key from passphrase
            key = self._derive_key(passphrase, salt)
            
            # Create cipher and encrypt
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            ciphertext, tag = cipher.encrypt_and_digest(json_data)
            
            # Combine all components: salt + nonce + tag + ciphertext
            encrypted_data = salt + nonce + tag + ciphertext
            
            return encrypted_data
            
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}")
    
    def decrypt_data(self, encrypted_data: bytes, passphrase: str) -> Dict[str, Any]:
        """
        Decrypt data using AES-GCM.
        
        Args:
            encrypted_data: Encrypted data bytes
            passphrase: Decryption passphrase
            
        Returns:
            Decrypted data dictionary
        """
        try:
            # Extract components
            salt = encrypted_data[:self.SALT_LENGTH]
            nonce = encrypted_data[self.SALT_LENGTH:self.SALT_LENGTH + self.NONCE_LENGTH]
            tag = encrypted_data[self.SALT_LENGTH + self.NONCE_LENGTH:self.SALT_LENGTH + self.NONCE_LENGTH + 16]
            ciphertext = encrypted_data[self.SALT_LENGTH + self.NONCE_LENGTH + 16:]
            
            # Derive key from passphrase
            key = self._derive_key(passphrase, salt)
            
            # Create cipher and decrypt
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            json_data = cipher.decrypt_and_verify(ciphertext, tag)
            
            # Parse JSON
            data = json.loads(json_data.decode('utf-8'))
            
            return data
            
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}")
    
    def save_enrollment(self, name: str, enrollment_data: Dict[str, Any], passphrase: str) -> str:
        """
        Save encrypted enrollment data to file.
        
        Args:
            name: User name
            enrollment_data: Enrollment data dictionary
            passphrase: Encryption passphrase
            
        Returns:
            Path to saved file
        """
        if not validate_name(name):
            raise ValueError(f"Invalid name: {name}")
        
        # Add metadata
        enrollment_data["saved_at"] = get_timestamp()
        enrollment_data["version"] = "1.0"
        
        # Encrypt data
        encrypted_data = self.encrypt_data(enrollment_data, passphrase)
        
        # Save to file
        file_path = os.path.join(self.storage_dir, f"{name}.enc")
        
        try:
            with open(file_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions (owner read/write only)
            os.chmod(file_path, 0o600)
            
            return file_path
            
        except Exception as e:
            raise EncryptionError(f"Failed to save enrollment: {e}")
    
    def load_enrollment(self, name: str, passphrase: str) -> Dict[str, Any]:
        """
        Load and decrypt enrollment data from file.
        
        Args:
            name: User name
            passphrase: Decryption passphrase
            
        Returns:
            Decrypted enrollment data dictionary
        """
        if not validate_name(name):
            raise ValueError(f"Invalid name: {name}")
        
        file_path = os.path.join(self.storage_dir, f"{name}.enc")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No enrollment found for user: {name}")
        
        try:
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            
            return self.decrypt_data(encrypted_data, passphrase)
            
        except Exception as e:
            if isinstance(e, EncryptionError):
                raise
            else:
                raise EncryptionError(f"Failed to load enrollment: {e}")
    
    def enrollment_exists(self, name: str) -> bool:
        """
        Check if enrollment file exists for user.
        
        Args:
            name: User name
            
        Returns:
            True if enrollment file exists
        """
        if not validate_name(name):
            return False
        
        file_path = os.path.join(self.storage_dir, f"{name}.enc")
        return os.path.exists(file_path)
    
    def delete_enrollment(self, name: str) -> bool:
        """
        Delete enrollment file for user.
        
        Args:
            name: User name
            
        Returns:
            True if file was deleted, False if not found
        """
        if not validate_name(name):
            return False
        
        file_path = os.path.join(self.storage_dir, f"{name}.enc")
        
        if os.path.exists(file_path):
            try:
                # Securely overwrite file before deletion
                file_size = os.path.getsize(file_path)
                with open(file_path, 'r+b') as f:
                    f.write(secrets.token_bytes(file_size))
                    f.flush()
                    os.fsync(f.fileno())
                
                os.remove(file_path)
                return True
                
            except Exception as e:
                raise EncryptionError(f"Failed to delete enrollment: {e}")
        
        return False
    
    def list_enrollments(self) -> list[str]:
        """
        List all available enrollment names.
        
        Returns:
            List of user names with enrollments
        """
        try:
            files = os.listdir(self.storage_dir)
            names = []
            
            for file in files:
                if file.endswith('.enc'):
                    name = file[:-4]  # Remove .enc extension
                    if validate_name(name):
                        names.append(name)
            
            return sorted(names)
            
        except Exception:
            return []
    
    def get_enrollment_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get enrollment file info without decrypting.
        
        Args:
            name: User name
            
        Returns:
            Dictionary with file info or None if not found
        """
        if not validate_name(name):
            return None
        
        file_path = os.path.join(self.storage_dir, f"{name}.enc")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            stat = os.stat(file_path)
            return {
                "name": name,
                "file_path": file_path,
                "file_size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime
            }
        except Exception:
            return None
    
    def verify_passphrase(self, name: str, passphrase: str) -> bool:
        """
        Verify passphrase without fully loading enrollment.
        
        Args:
            name: User name
            passphrase: Passphrase to verify
            
        Returns:
            True if passphrase is correct
        """
        try:
            self.load_enrollment(name, passphrase)
            return True
        except EncryptionError:
            return False
        except FileNotFoundError:
            return False
    
    def change_passphrase(self, name: str, old_passphrase: str, new_passphrase: str) -> bool:
        """
        Change passphrase for existing enrollment.
        
        Args:
            name: User name
            old_passphrase: Current passphrase
            new_passphrase: New passphrase
            
        Returns:
            True if passphrase was changed successfully
        """
        try:
            # Load with old passphrase
            enrollment_data = self.load_enrollment(name, old_passphrase)
            
            # Update timestamp
            enrollment_data["passphrase_changed"] = get_timestamp()
            
            # Save with new passphrase
            self.save_enrollment(name, enrollment_data, new_passphrase)
            
            return True
            
        except Exception as e:
            raise EncryptionError(f"Failed to change passphrase: {e}")


class BackupManager:
    """Handles secure backup and restore of enrollment data."""
    
    def __init__(self, storage: SecureStorage):
        """
        Initialize backup manager.
        
        Args:
            storage: SecureStorage instance
        """
        self.storage = storage
    
    def create_backup(self, backup_path: str, passphrase: str) -> Dict[str, Any]:
        """
        Create encrypted backup of all enrollments.
        
        Args:
            backup_path: Path for backup file
            passphrase: Backup encryption passphrase
            
        Returns:
            Backup metadata dictionary
        """
        enrollments = self.storage.list_enrollments()
        
        if not enrollments:
            raise ValueError("No enrollments found to backup")
        
        backup_data = {
            "version": "1.0",
            "created": get_timestamp(),
            "enrollments": {},
            "count": len(enrollments)
        }
        
        # Note: This would require individual passphrases for each enrollment
        # For simplicity, we'll document this limitation
        raise NotImplementedError(
            "Backup functionality requires individual enrollment passphrases. "
            "Use file system backup of encrypted .enc files instead."
        )
    
    def restore_backup(self, backup_path: str, passphrase: str) -> int:
        """
        Restore enrollments from encrypted backup.
        
        Args:
            backup_path: Path to backup file
            passphrase: Backup decryption passphrase
            
        Returns:
            Number of enrollments restored
        """
        raise NotImplementedError("See create_backup limitation")


# Utility functions for CLI usage
def get_secure_passphrase(prompt: str = "Enter passphrase: ", confirm: bool = False) -> str:
    """
    Get passphrase securely from user input.
    
    Args:
        prompt: Input prompt
        confirm: Whether to ask for confirmation
        
    Returns:
        Passphrase string
    """
    from .utils import safe_input
    
    passphrase = safe_input(prompt, hidden=True)
    
    if not passphrase:
        raise ValueError("Passphrase cannot be empty")
    
    if confirm:
        confirm_passphrase = safe_input("Confirm passphrase: ", hidden=True)
        if passphrase != confirm_passphrase:
            raise ValueError("Passphrases do not match")
    
    return passphrase


def create_storage_instance(storage_dir: str = "face_data") -> SecureStorage:
    """
    Create and return SecureStorage instance.
    
    Args:
        storage_dir: Storage directory path
        
    Returns:
        SecureStorage instance
    """
    return SecureStorage(storage_dir)
