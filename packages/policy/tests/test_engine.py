"""
Tests for PolicyEngine.

Verifies:
- Policy loading from YAML files
- Tool permission checks (allowed/denied)
- Environment-specific rules
- Error handling for missing/invalid configs
"""

import pytest
from pathlib import Path

from ai_platform.policy import PolicyEngine
from ai_platform.core.models import PolicyResult


@pytest.fixture
def policy_engine():
    """Create a PolicyEngine with default config directory."""
    # Use the actual configs/policies directory
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_dir = workspace_root / "configs" / "policies"
    return PolicyEngine(config_dir=config_dir)


class TestPolicyEngineLoading:
    """Test policy configuration loading."""
    
    def test_load_banking_policy(self, policy_engine):
        """Test loading banking policy configuration."""
        policy = policy_engine.load_policy("banking")
        
        assert policy is not None
        assert policy["environment"] == "banking"
        assert "tool_rules" in policy
        assert "allowed" in policy["tool_rules"]
        assert "denied" in policy["tool_rules"]
    
    def test_load_retail_policy(self, policy_engine):
        """Test loading retail policy configuration."""
        policy = policy_engine.load_policy("retail")
        
        assert policy is not None
        assert policy["environment"] == "retail"
        assert "tool_rules" in policy
    
    def test_load_nonexistent_policy(self, policy_engine):
        """Test loading non-existent policy raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            policy_engine.load_policy("nonexistent")
        
        assert "Policy file not found" in str(exc_info.value)
        assert "nonexistent_policy.yaml" in str(exc_info.value)
    
    def test_policy_caching(self, policy_engine):
        """Test that loaded policies are cached."""
        # Load policy twice
        policy1 = policy_engine.load_policy("banking")
        policy2 = policy_engine.load_policy("banking")
        
        # Should return the same cached object
        assert policy1 is policy2


class TestToolPermissions:
    """Test tool permission checks."""
    
    def test_banking_allowed_tools(self, policy_engine):
        """Test that banking-specific tools are allowed."""
        assert policy_engine.is_tool_allowed("loan_checker", "banking") is True
        assert policy_engine.is_tool_allowed("credit_score_lookup", "banking") is True
    
    def test_banking_denied_tools(self, policy_engine):
        """Test that retail tools are denied in banking."""
        assert policy_engine.is_tool_allowed("order_status", "banking") is False
        assert policy_engine.is_tool_allowed("recommendation_engine", "banking") is False
    
    def test_retail_allowed_tools(self, policy_engine):
        """Test that retail-specific tools are allowed."""
        assert policy_engine.is_tool_allowed("order_status", "retail") is True
        assert policy_engine.is_tool_allowed("recommendation_engine", "retail") is True
        assert policy_engine.is_tool_allowed("product_search", "retail") is True
    
    def test_retail_denied_tools(self, policy_engine):
        """Test that banking tools are denied in retail."""
        assert policy_engine.is_tool_allowed("loan_checker", "retail") is False
        assert policy_engine.is_tool_allowed("credit_score_lookup", "retail") is False
    
    def test_unknown_tool_denied_by_default(self, policy_engine):
        """Test that unknown tools are denied by default (fail-safe)."""
        assert policy_engine.is_tool_allowed("unknown_tool", "banking") is False
        assert policy_engine.is_tool_allowed("unknown_tool", "retail") is False
    
    def test_missing_environment_denies_all(self, policy_engine):
        """Test that missing environment denies all tools."""
        assert policy_engine.is_tool_allowed("loan_checker", "nonexistent") is False


class TestPolicyEvaluation:
    """Test PolicyResult evaluation."""
    
    def test_evaluate_allowed_tool(self, policy_engine):
        """Test evaluating an allowed tool returns allowed=True."""
        result = policy_engine.evaluate("loan_checker", "banking")
        
        assert isinstance(result, PolicyResult)
        assert result.allowed is True
        assert result.action == "loan_checker"
        assert result.environment == "banking"
        assert "allowed" in result.reason.lower()
    
    def test_evaluate_denied_tool(self, policy_engine):
        """Test evaluating a denied tool returns allowed=False."""
        result = policy_engine.evaluate("order_status", "banking")
        
        assert isinstance(result, PolicyResult)
        assert result.allowed is False
        assert result.action == "order_status"
        assert result.environment == "banking"
        assert "denied" in result.reason.lower() or "not in the allowed list" in result.reason.lower()
    
    def test_evaluate_unknown_tool(self, policy_engine):
        """Test evaluating an unknown tool returns allowed=False."""
        result = policy_engine.evaluate("unknown_tool", "banking")
        
        assert isinstance(result, PolicyResult)
        assert result.allowed is False


class TestHelperMethods:
    """Test helper methods."""
    
    def test_get_allowed_tools_banking(self, policy_engine):
        """Test getting allowed tools list for banking."""
        allowed = policy_engine.get_allowed_tools("banking")
        
        assert isinstance(allowed, list)
        assert "loan_checker" in allowed
        assert "credit_score_lookup" in allowed
        assert "order_status" not in allowed
    
    def test_get_allowed_tools_retail(self, policy_engine):
        """Test getting allowed tools list for retail."""
        allowed = policy_engine.get_allowed_tools("retail")
        
        assert isinstance(allowed, list)
        assert "order_status" in allowed
        assert "recommendation_engine" in allowed
        assert "loan_checker" not in allowed
    
    def test_get_pii_config_banking(self, policy_engine):
        """Test getting PII config for banking."""
        pii_config = policy_engine.get_pii_config("banking")
        
        assert isinstance(pii_config, dict)
        assert pii_config.get("enabled") is True
        assert pii_config.get("action") == "mask"
    
    def test_get_pii_config_retail(self, policy_engine):
        """Test getting PII config for retail."""
        pii_config = policy_engine.get_pii_config("retail")
        
        assert isinstance(pii_config, dict)
        assert pii_config.get("enabled") is False
