"""
Role-Based Access Control Module

This module manages user roles and their associated permissions.
"""

from typing import Dict, Set, List


class RBACManager:
    """Manages role-based access control for the system."""

    def __init__(self):
        self.roles: Dict[str, Set[str]] = {
            "viewer": {
                "read_public",
                "view_dashboard",
            },
            "editor": {
                "read_public",
                "view_dashboard",
                "create_content",
                "edit_content",
                "delete_own_content",
            },
            "moderator": {
                "read_public",
                "view_dashboard",
                "create_content",
                "edit_content",
                "delete_own_content",
                "delete_any_content",
                "ban_users",
                "view_reports",
            },
            "audit_admin": {
                "read_public",
                "view_dashboard",
                "view_audit_logs",
                "export_audit_logs",
                "view_user_activity",
                "generate_compliance_reports",
            },
            "system_admin": {
                "read_public",
                "view_dashboard",
                "create_content",
                "edit_content",
                "delete_own_content",
                "delete_any_content",
                "ban_users",
                "view_reports",
                "manage_users",
                "manage_roles",
                "system_config",
                "restart_services",
            },
        }

        self.user_roles: Dict[str, str] = {}

    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign a role to a user."""
        if role not in self.roles:
            return False
        self.user_roles[user_id] = role
        return True

    def get_user_role(self, user_id: str) -> str:
        """Get the role assigned to a user."""
        return self.user_roles.get(user_id, "viewer")

    def get_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user based on their role."""
        role = self.get_user_role(user_id)
        return self.roles.get(role, set())

    def has_permission(self, user_id: str, permission: str) -> bool:
        """Check if a user has a specific permission."""
        permissions = self.get_permissions(user_id)
        return permission in permissions

    def list_roles(self) -> List[str]:
        """List all available roles."""
        return list(self.roles.keys())

    def get_role_permissions(self, role: str) -> Set[str]:
        """Get permissions for a specific role."""
        return self.roles.get(role, set())

    def create_role(self, role_name: str, permissions: Set[str]) -> bool:
        """Create a new role with specified permissions."""
        if role_name in self.roles:
            return False
        self.roles[role_name] = permissions
        return True

    def delete_role(self, role_name: str) -> bool:
        """Delete a role. Cannot delete built-in roles."""
        builtin = {"viewer", "editor", "moderator", "audit_admin", "system_admin"}
        if role_name in builtin:
            return False
        if role_name in self.roles:
            del self.roles[role_name]
            return True
        return False


if __name__ == "__main__":
    rbac = RBACManager()

    rbac.assign_role("user_1", "editor")
    rbac.assign_role("user_2", "audit_admin")
    rbac.assign_role("user_3", "system_admin")

    print(f"User 1 permissions: {rbac.get_permissions('user_1')}")
    print(f"User 2 can view audit logs: {rbac.has_permission('user_2', 'view_audit_logs')}")
    print(f"User 2 can manage users: {rbac.has_permission('user_2', 'manage_users')}")
    print(f"User 3 can manage users: {rbac.has_permission('user_3', 'manage_users')}")
