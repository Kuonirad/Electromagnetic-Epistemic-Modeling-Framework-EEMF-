"""Security module for data encryption and compliance."""

import os
from typing import Any, Dict

from cryptography.fernet import Fernet


class SecurityManager:
    """Handles encryption and security compliance."""

    def __init__(self):
        """Initialize security manager with encryption key."""
        self.key = os.getenv("ENCRYPTION_KEY")
        if not self.key:
            self.key = Fernet.generate_key()
            os.environ["ENCRYPTION_KEY"] = self.key.decode()
        self.cipher = Fernet(
            self.key if isinstance(self.key, bytes) else self.key.encode()
        )

    def encrypt_results(self, data: bytes) -> bytes:
        """Encrypt simulation results.

        Args:
            data: Raw data to encrypt

        Returns:
            Encrypted data
        """
        return self.cipher.encrypt(data)

    def decrypt_results(self, encrypted_data: bytes) -> bytes:
        """Decrypt simulation results.

        Args:
            encrypted_data: Encrypted data to decrypt

        Returns:
            Decrypted data
        """
        return self.cipher.decrypt(encrypted_data)

    def audit_phi(self, data: Dict[str, Any]) -> bool:
        """Check for presence of Protected Health Information.

        Args:
            data: Data to audit

        Returns:
            True if compliant, False if PHI detected

        Raises:
            ComplianceError: If PHI is detected
        """
        phi_fields = ["patient_id", "name", "ssn", "dob"]
        for field in phi_fields:
            if field in data:
                raise ComplianceError(f"PHI detected: {field}")
        return True


class ComplianceError(Exception):
    """Raised when compliance violations are detected."""

    pass
