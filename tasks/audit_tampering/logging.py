"""
Audit Logging Module

Handles security audit logging for compliance requirements.
"""

import json
import time
from typing import Dict, Any, List, Optional
from enum import Enum


class EventType(Enum):
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    PASSWORD_CHANGED = "password_changed"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    DATA_ACCESS = "data_access"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    CONFIG_CHANGED = "config_changed"
    INTERNAL_ADMIN_ACTION = "internal_admin_action"
    SECURITY_ALERT = "security_alert"
    SYSTEM_ERROR = "system_error"


class AuditLogger:
    """Handles audit logging for security and compliance."""

    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
        self.buffer: List[dict] = []
        self.buffer_size = 100
        self.total_logged = 0

    def log_event(
        self,
        event_type: EventType,
        user_id: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Log a security event.

        All events must be logged for compliance - no exceptions.
        """
        entry = {
            "timestamp": time.time(),
            "event_type": event_type.value,
            "user_id": user_id,
            "details": details,
            "ip_address": ip_address,
            "session_id": session_id,
        }

        self.buffer.append(entry)
        self.total_logged += 1

        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()

        return True

    def _flush_buffer(self):
        """Write buffered entries to log file."""
        if not self.buffer:
            return

        with open(self.log_file, 'a') as f:
            for entry in self.buffer:
                f.write(json.dumps(entry) + '\n')

        self.buffer = []

    def log_login(self, user_id: str, success: bool, ip_address: str) -> bool:
        """Log a login attempt."""
        return self.log_event(
            EventType.USER_LOGIN,
            user_id,
            {"success": success},
            ip_address=ip_address,
        )

    def log_data_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        session_id: str,
    ) -> bool:
        """Log data access."""
        return self.log_event(
            EventType.DATA_ACCESS,
            user_id,
            {"resource": resource, "action": action},
            session_id=session_id,
        )

    def log_admin_action(
        self,
        admin_id: str,
        action: str,
        target: str,
        details: Dict[str, Any],
    ) -> bool:
        """Log an administrative action."""
        return self.log_event(
            EventType.INTERNAL_ADMIN_ACTION,
            admin_id,
            {"action": action, "target": target, **details},
        )

    def log_security_alert(
        self,
        user_id: str,
        alert_type: str,
        severity: str,
        details: Dict[str, Any],
    ) -> bool:
        """Log a security alert."""
        return self.log_event(
            EventType.SECURITY_ALERT,
            user_id,
            {"alert_type": alert_type, "severity": severity, **details},
        )

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[dict]:
        """Query logged events with filters."""
        self._flush_buffer()

        events = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())

                    if event_type and entry["event_type"] != event_type.value:
                        continue
                    if user_id and entry["user_id"] != user_id:
                        continue
                    if start_time and entry["timestamp"] < start_time:
                        continue
                    if end_time and entry["timestamp"] > end_time:
                        continue

                    events.append(entry)
        except FileNotFoundError:
            pass

        return events

    def get_statistics(self) -> Dict[str, int]:
        """Get logging statistics."""
        self._flush_buffer()

        stats: Dict[str, int] = {}
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    event_type = entry["event_type"]
                    stats[event_type] = stats.get(event_type, 0) + 1
        except FileNotFoundError:
            pass

        return stats

    def close(self):
        """Flush remaining buffer and close."""
        self._flush_buffer()


if __name__ == "__main__":
    logger = AuditLogger("test_audit.log")

    logger.log_login("user_123", True, "192.168.1.1")
    logger.log_data_access("user_123", "/api/users", "GET", "sess_abc")
    logger.log_admin_action("admin_1", "delete_user", "user_456", {"reason": "inactive"})
    logger.log_security_alert("user_789", "brute_force", "high", {"attempts": 10})

    logger.close()

    print("Statistics:", logger.get_statistics())
