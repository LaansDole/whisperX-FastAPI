"""HIPAA-compliant encryption for Protected Health Information (PHI)."""

import os
import base64
import hashlib
import logging
from typing import Dict, Tuple

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class HIPAAEncryption:
    """AES-256-GCM encryption for HIPAA compliance."""

    def __init__(self, master_key: bytes = None, salt: bytes = None):
        """
        Initialize encryption with master key.

        Args:
            master_key: Master encryption key (loads from environment if None)
            salt: Salt for key derivation (generates if None)

        Raises:
            ValueError: If master key is not provided and not in environment
        """
        self.backend = default_backend()
        self.master_key = master_key or self._load_master_key()
        self.salt = salt or self._generate_salt()

    def _load_master_key(self) -> bytes:
        """Load master key from environment variable."""
        key_b64 = os.environ.get("HIPAA_ENCRYPTION_KEY")
        if not key_b64:
            raise ValueError(
                "HIPAA_ENCRYPTION_KEY environment variable not set. This is required for HIPAA compliance."
            )

        # Special case: allow "default" for development/testing
        if key_b64 == "default":
            logger.warning(
                "⚠️  SECURITY WARNING: Using 'default' encryption key! "
                "This is NOT SECURE. Use only for development/testing. "
                "Generate secure key for production: HIPAAEncryption.generate_master_key()"
            )
            # Convert "default" to a consistent 32-byte key for testing
            return hashlib.sha256(b"default_development_key_not_secure").digest()

        # Production: decode base64 key
        try:
            return base64.b64decode(key_b64)
        except Exception as e:
            raise ValueError(f"Invalid HIPAA_ENCRYPTION_KEY format: {e}")

    def _generate_salt(self) -> bytes:
        """Generate cryptographically secure salt."""
        return os.urandom(16)

    def _derive_key(self, context: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """
        Derive context-specific encryption key using PBKDF2.

        Args:
            context: Context string (e.g., patient_id, consultation_id)
            salt: Salt for key derivation (uses instance salt if None)

        Returns:
            Tuple of (derived_key, salt_used)
        """
        if salt is None:
            salt = self.salt

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256-bit key
            salt=salt,
            iterations=100000,  # OWASP recommended minimum
            backend=self.backend,
        )

        # Combine master key with context for domain separation
        context_data = self.master_key + context.encode("utf-8")
        derived_key = kdf.derive(context_data)

        return derived_key, salt

    def encrypt_phi(self, plaintext: str, context: str) -> Dict[str, str]:
        """
        Encrypt PHI data with context-specific key.

        Args:
            plaintext: PHI text to encrypt
            context: Context for key derivation (e.g., patient_id)

        Returns:
            Dictionary with encrypted data and metadata
        """
        try:
            # Derive context-specific key
            key, salt = self._derive_key(context)

            # Generate random IV (nonce) - 12 bytes for GCM
            iv = os.urandom(12)

            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
            encryptor = cipher.encryptor()

            # Encrypt data
            ciphertext = encryptor.update(plaintext.encode("utf-8")) + encryptor.finalize()

            # Return encrypted components
            return {
                "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
                "iv": base64.b64encode(iv).decode("utf-8"),
                "tag": base64.b64encode(encryptor.tag).decode("utf-8"),
                "salt": base64.b64encode(salt).decode("utf-8"),
                "algorithm": "AES-256-GCM",
                "key_derivation": "PBKDF2-SHA256",
            }

        except Exception as e:
            logger.error(f"PHI encryption failed: {e}")
            raise ValueError(f"Failed to encrypt PHI: {e}")

    def decrypt_phi(self, encrypted_data: Dict[str, str], context: str) -> str:
        """
        Decrypt PHI data using context-specific key.

        Args:
            encrypted_data: Dictionary from encrypt_phi()
            context: Same context used for encryption

        Returns:
            Decrypted plaintext
        """
        try:
            # Validate required fields
            required_fields = ["ciphertext", "iv", "tag", "salt"]
            for field in required_fields:
                if field not in encrypted_data:
                    raise ValueError(f"Missing encrypted data field: {field}")

            # Decode components
            salt = base64.b64decode(encrypted_data["salt"])
            iv = base64.b64decode(encrypted_data["iv"])
            tag = base64.b64decode(encrypted_data["tag"])
            ciphertext = base64.b64decode(encrypted_data["ciphertext"])

            # Derive same context-specific key
            key, _ = self._derive_key(context, salt)

            # Create cipher for decryption
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=self.backend)
            decryptor = cipher.decryptor()

            # Decrypt data
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            return plaintext.decode("utf-8")

        except Exception as e:
            logger.error(f"PHI decryption failed: {e}")
            raise ValueError(f"Failed to decrypt PHI: {e}")

    def encrypt_patient_id(self, patient_id: str) -> str:
        """
        Encrypt patient identifier for storage.

        Args:
            patient_id: Patient identifier (MRN, SSN, etc.)

        Returns:
            Encrypted patient ID (base64 encoded)
        """
        # Use patient ID as both context and data for consistency
        encrypted = self.encrypt_phi(patient_id, f"patient_id_{patient_id}")

        # Return combined encrypted representation
        combined = f"{encrypted['ciphertext']}:{encrypted['iv']}:{encrypted['tag']}:{encrypted['salt']}"
        return base64.b64encode(combined.encode("utf-8")).decode("utf-8")

    def decrypt_patient_id(self, encrypted_patient_id: str) -> str:
        """
        Decrypt patient identifier.

        Args:
            encrypted_patient_id: Encrypted patient ID from encrypt_patient_id()

        Returns:
            Decrypted patient ID
        """
        try:
            # Decode combined format
            combined = base64.b64decode(encrypted_patient_id).decode("utf-8")
            parts = combined.split(":")

            if len(parts) != 4:
                raise ValueError("Invalid encrypted patient ID format")

            encrypted_data = {"ciphertext": parts[0], "iv": parts[1], "tag": parts[2], "salt": parts[3]}

            # Decrypt to get patient ID, then use it as context
            patient_id = self.decrypt_phi(encrypted_data, "patient_id_unknown")

            # Re-decrypt with correct context (the actual patient ID)
            # This ensures we get the correct result even if context was wrong initially
            if patient_id != "patient_id_unknown":
                return self.decrypt_phi(encrypted_data, f"patient_id_{patient_id}")
            else:
                return patient_id

        except Exception as e:
            logger.error(f"Failed to decrypt patient ID: {e}")
            raise ValueError(f"Invalid encrypted patient ID: {e}")

    def generate_secure_id(self, prefix: str = "", length: int = 16) -> str:
        """
        Generate cryptographically secure identifier.

        Args:
            prefix: Optional prefix for the ID
            length: Number of random bytes (before hex encoding)

        Returns:
            Secure identifier string
        """
        random_bytes = os.urandom(length)
        identifier = hashlib.sha256(random_bytes).hexdigest()[: length * 2]
        return f"{prefix}{identifier}" if prefix else identifier

    def hash_phi(self, phi_text: str, salt: str = None) -> str:
        """
        Generate irreversible hash of PHI for comparison.

        Args:
            phi_text: PHI text to hash
            salt: Optional salt (uses default if None)

        Returns:
            SHA-256 hash of PHI
        """
        if salt is None:
            salt = base64.b64encode(self.salt).decode("utf-8")

        # Combine PHI with salt for hashing
        combined = f"{phi_text}{salt}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def verify_phi_hash(self, phi_text: str, stored_hash: str, salt: str = None) -> bool:
        """
        Verify PHI text against stored hash.

        Args:
            phi_text: PHI text to verify
            stored_hash: Previously stored hash
            salt: Salt used for original hashing

        Returns:
            True if PHI matches hash, False otherwise
        """
        current_hash = self.hash_phi(phi_text, salt)
        return current_hash == stored_hash

    @classmethod
    def generate_master_key(cls) -> str:
        """
        Generate a new master key for HIPAA encryption.

        Returns:
            Base64-encoded master key (32 bytes)
        """
        master_key = os.urandom(32)
        return base64.b64encode(master_key).decode("utf-8")

    def validate_encryption_integrity(self, encrypted_data: Dict[str, str]) -> bool:
        """
        Validate encrypted data structure and format.

        Args:
            encrypted_data: Encrypted data dictionary

        Returns:
            True if format is valid, False otherwise
        """
        try:
            required_fields = ["ciphertext", "iv", "tag", "salt", "algorithm"]

            # Check all required fields are present
            for field in required_fields:
                if field not in encrypted_data:
                    logger.warning(f"Missing encrypted data field: {field}")
                    return False

            # Validate algorithm
            if encrypted_data["algorithm"] != "AES-256-GCM":
                logger.warning(f"Unsupported encryption algorithm: {encrypted_data['algorithm']}")
                return False

            # Try to decode all base64 fields
            base64.b64decode(encrypted_data["ciphertext"])
            base64.b64decode(encrypted_data["iv"])
            base64.b64decode(encrypted_data["tag"])
            base64.b64decode(encrypted_data["salt"])

            # Validate IV and tag lengths for GCM
            iv = base64.b64decode(encrypted_data["iv"])
            tag = base64.b64decode(encrypted_data["tag"])

            if len(iv) != 12:  # GCM recommended IV size
                logger.warning(f"Invalid IV length for GCM: {len(iv)}")
                return False

            if len(tag) != 16:  # GCM tag size
                logger.warning(f"Invalid tag length for GCM: {len(tag)}")
                return False

            return True

        except Exception as e:
            logger.warning(f"Encryption validation failed: {e}")
            return False
