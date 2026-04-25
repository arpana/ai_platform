"""
Policy Engine - Tool governance and access control.

Loads policy rules from YAML configuration files and enforces:
- Tool permission checks (allowed/denied lists)
- Environment-specific access control
- Integration with PII enforcement

Example YAML structure (banking_policy.yaml):
    environment: banking
    tool_rules:
      allowed:
        - loan_checker
        - credit_score_lookup
      denied:
        - order_status
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ai_platform.core.models import PolicyResult


class PolicyEngine:
    """
    Policy engine for tool governance and access control.
    
    Loads environment-specific policy rules from YAML files and provides
    methods to check tool permissions.
    
    Attributes:
        config_dir: Directory containing policy YAML files
        policies: Cached policy configurations by environment
    """
    
    def __init__(self, config_dir: str | Path | None = None):
        """
        Initialize the policy engine.
        
        Args:
            config_dir: Path to directory containing policy YAML files.
                       Defaults to configs/policies/ relative to workspace root.
        """
        if config_dir is None:
            # Default to configs/policies/ in workspace root
            workspace_root = Path(__file__).parent.parent.parent.parent
            config_dir = workspace_root / "configs" / "policies"
        
        self.config_dir = Path(config_dir)
        self.policies: dict[str, dict[str, Any]] = {}
        
        if not self.config_dir.exists():
            raise ValueError(f"Policy config directory not found: {self.config_dir}")
    
    def load_policy(self, environment: str) -> dict[str, Any]:
        """
        Load policy configuration for an environment.
        
        Args:
            environment: Environment name (e.g., "banking", "retail")
            
        Returns:
            Policy configuration dictionary
            
        Raises:
            FileNotFoundError: If policy file doesn't exist
            ValueError: If policy file is invalid
        """
        # Check cache first
        if environment in self.policies:
            return self.policies[environment]
        
        # Load from YAML file
        policy_file = self.config_dir / f"{environment}_policy.yaml"
        
        if not policy_file.exists():
            raise FileNotFoundError(
                f"Policy file not found: {policy_file}. "
                f"Expected {environment}_policy.yaml in {self.config_dir}"
            )
        
        with open(policy_file, "r") as f:
            policy = yaml.safe_load(f)
        
        # Validate policy structure
        if not isinstance(policy, dict):
            raise ValueError(f"Invalid policy file: {policy_file}. Expected dict, got {type(policy)}")
        
        if "environment" not in policy:
            raise ValueError(f"Policy file missing 'environment' key: {policy_file}")
        
        if policy["environment"] != environment:
            raise ValueError(
                f"Policy environment mismatch in {policy_file}. "
                f"Expected '{environment}', got '{policy['environment']}'"
            )
        
        # Cache and return
        self.policies[environment] = policy
        return policy
    
    def is_tool_allowed(self, tool_name: str, environment: str) -> bool:
        """
        Check if a tool is allowed in the given environment.
        
        Args:
            tool_name: Name of the tool to check
            environment: Environment name
            
        Returns:
            True if tool is allowed, False otherwise
        """
        try:
            policy = self.load_policy(environment)
        except (FileNotFoundError, ValueError):
            # If no policy found, default to deny for safety
            return False
        
        tool_rules = policy.get("tool_rules", {})
        allowed = tool_rules.get("allowed", [])
        denied = tool_rules.get("denied", [])
        
        # Explicit deny takes precedence
        if tool_name in denied:
            return False
        
        # Check allowed list
        if tool_name in allowed:
            return True
        
        # If not in either list, default to deny for safety
        return False
    
    def evaluate(self, tool_name: str, environment: str) -> PolicyResult:
        """
        Evaluate tool permission and return a PolicyResult.
        
        Args:
            tool_name: Name of the tool to check
            environment: Environment name
            
        Returns:
            PolicyResult with decision and reason
        """
        allowed = self.is_tool_allowed(tool_name, environment)
        
        if allowed:
            reason = f"Tool '{tool_name}' is allowed in {environment} environment"
        else:
            # Check if explicitly denied or just not allowed
            try:
                policy = self.load_policy(environment)
                tool_rules = policy.get("tool_rules", {})
                denied = tool_rules.get("denied", [])
                
                if tool_name in denied:
                    reason = f"Tool '{tool_name}' is explicitly denied in {environment} environment"
                else:
                    reason = f"Tool '{tool_name}' is not in the allowed list for {environment} environment"
            except (FileNotFoundError, ValueError) as e:
                reason = f"Policy error: {str(e)}"
        
        return PolicyResult(
            allowed=allowed,
            reason=reason,
            action=tool_name,
            environment=environment
        )
    
    def get_allowed_tools(self, environment: str) -> list[str]:
        """
        Get list of allowed tools for an environment.
        
        Args:
            environment: Environment name
            
        Returns:
            List of allowed tool names
        """
        try:
            policy = self.load_policy(environment)
            tool_rules = policy.get("tool_rules", {})
            return tool_rules.get("allowed", [])
        except (FileNotFoundError, ValueError):
            return []
    
    def get_pii_config(self, environment: str) -> dict[str, Any]:
        """
        Get PII enforcement configuration for an environment.
        
        Args:
            environment: Environment name
            
        Returns:
            PII configuration dictionary
        """
        try:
            policy = self.load_policy(environment)
            return policy.get("pii_enforcement", {})
        except (FileNotFoundError, ValueError):
            return {}
