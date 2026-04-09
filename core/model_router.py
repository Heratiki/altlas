#!/usr/bin/env python3
"""
Role-based model routing for AltLAS.

This keeps the rest of the codebase provider-agnostic while allowing
different roles to prefer different providers, temperatures, and models.
"""

from copy import deepcopy
from typing import Any, Dict, Optional

from llm_provider import LLMProvider


class ModelRouter:
    """Routes requests to role-specific LLM providers."""

    def __init__(self, default_config: Optional[Dict[str, Any]] = None, role_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        self.default_config = deepcopy(default_config or {})
        self.role_configs = {role.lower(): deepcopy(config) for role, config in (role_configs or {}).items()}
        self._providers: Dict[str, LLMProvider] = {}

    def _build_role_config(self, role: str) -> Dict[str, Any]:
        role_key = (role or "default").lower()
        config = deepcopy(self.default_config)
        config.update(self.role_configs.get(role_key, {}))
        config.setdefault("tier", self.default_config.get("tier", "local"))
        config.setdefault("route_label", role_key)
        return config

    def _get_provider(self, role: str) -> LLMProvider:
        role_key = (role or "default").lower()
        if role_key not in self._providers:
            self._providers[role_key] = LLMProvider(config=self._build_role_config(role_key))
        return self._providers[role_key]

    def generate(self, role: str, prompt_or_template, **kwargs) -> str:
        """Generate text using the provider configured for the given role."""
        return self._get_provider(role).generate(prompt_or_template, **kwargs)

    def describe_role(self, role: str) -> Dict[str, Any]:
        """Return metadata for the resolved role configuration."""
        config = self._build_role_config(role)
        return {
            "role": (role or "default").lower(),
            "provider": config.get("provider", "unknown"),
            "tier": config.get("tier", "unknown"),
            "route_label": config.get("route_label", role),
            "model_preference": config.get("model_preference", []),
        }

    def get_role_stats(self, role: str) -> Dict[str, Any]:
        """Return stats for the provider handling the given role."""
        stats = self._get_provider(role).get_stats()
        stats.update(self.describe_role(role))
        return stats

    def get_all_role_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return stats for all roles that have been used so far."""
        return {role: self.get_role_stats(role) for role in self._providers}
