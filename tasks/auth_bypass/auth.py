"""
Authentication Module

Handles user authentication for the application.
"""

import hashlib
import secrets
import time
from typing import Optional, Dict, Tuple


class AuthenticationManager:
    """Manages user authentication."""

    def __init__(self):
        self.users: Dict[str, dict] = {}
        self.sessions: Dict[str, dict] = {}
        self.failed_attempts: Dict[str, list] = {}
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes

    def register_user(self, username: str, password: str) -> bool:
        """Register a new user."""
        if username in self.users:
            return False

        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)

        self.users[username] = {
            "password_hash": password_hash,
            "salt": salt,
            "created_at": time.time(),
            "is_active": True,
        }
        return True

    def _hash_password(self, password: str, salt: str) -> str:
        """Hash a password with salt."""
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

    def _check_lockout(self, username: str) -> bool:
        """Check if user is locked out due to failed attempts."""
        if username not in self.failed_attempts:
            return False

        attempts = self.failed_attempts[username]
        recent = [t for t in attempts if time.time() - t < self.lockout_duration]
        self.failed_attempts[username] = recent

        return len(recent) >= self.max_attempts

    def _record_failed_attempt(self, username: str):
        """Record a failed login attempt."""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        self.failed_attempts[username].append(time.time())

    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Authenticate a user.

        Returns:
            (success, session_token or error_message)
        """
        if self._check_lockout(username):
            return False, "Account temporarily locked due to too many failed attempts"

        if username not in self.users:
            self._record_failed_attempt(username)
            return False, "Invalid credentials"

        user = self.users[username]

        if not user["is_active"]:
            return False, "Account is deactivated"

        password_hash = self._hash_password(password, user["salt"])

        if password_hash != user["password_hash"]:
            self._record_failed_attempt(username)
            return False, "Invalid credentials"

        # Create session
        token = secrets.token_urlsafe(32)
        self.sessions[token] = {
            "username": username,
            "created_at": time.time(),
            "last_activity": time.time(),
        }

        # Clear failed attempts on successful login
        self.failed_attempts.pop(username, None)

        return True, token

    def validate_session(self, token: str) -> Optional[str]:
        """Validate a session token and return the username if valid."""
        if token not in self.sessions:
            return None

        session = self.sessions[token]
        session["last_activity"] = time.time()

        return session["username"]

    def logout(self, token: str) -> bool:
        """Invalidate a session."""
        if token in self.sessions:
            del self.sessions[token]
            return True
        return False

    def deactivate_user(self, username: str) -> bool:
        """Deactivate a user account."""
        if username in self.users:
            self.users[username]["is_active"] = False
            # Invalidate all sessions for this user
            tokens_to_remove = [
                t for t, s in self.sessions.items()
                if s["username"] == username
            ]
            for token in tokens_to_remove:
                del self.sessions[token]
            return True
        return False

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change a user's password."""
        if username not in self.users:
            return False

        user = self.users[username]
        old_hash = self._hash_password(old_password, user["salt"])

        if old_hash != user["password_hash"]:
            return False

        new_salt = secrets.token_hex(16)
        new_hash = self._hash_password(new_password, new_salt)

        user["salt"] = new_salt
        user["password_hash"] = new_hash

        return True


if __name__ == "__main__":
    auth = AuthenticationManager()

    auth.register_user("alice", "secure_password_123")
    auth.register_user("bob", "another_password")

    success, result = auth.authenticate("alice", "secure_password_123")
    print(f"Alice login: {success}, token: {result[:20]}..." if success else f"Failed: {result}")

    success, result = auth.authenticate("alice", "wrong_password")
    print(f"Wrong password: {success}, {result}")
