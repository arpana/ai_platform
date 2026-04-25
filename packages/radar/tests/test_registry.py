"""
Tests for TechRadar registry.

Verifies:
- Radar loading from YAML
- Tool status lookups
- Enforcement logic (block/allow)
- Status filtering
"""

import pytest
from pathlib import Path

from ai_platform.radar import TechRadar
from ai_platform.core.models import RadarStatus
from ai_platform.core.exceptions import RadarBlockedError


@pytest.fixture
def radar():
    """Create a TechRadar with actual config."""
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_path = workspace_root / "configs" / "radar" / "tech_radar.yaml"
    return TechRadar(config_path=config_path)


class TestRadarLoading:
    """Test radar configuration loading."""
    
    def test_load_radar_config(self, radar):
        """Test loading radar from YAML."""
        assert radar.radar is not None
        assert len(radar.radar) > 0
    
    def test_approved_tools_loaded(self, radar):
        """Test that approved tools are loaded."""
        entry = radar.get_status("loan_checker")
        assert entry.name == "loan_checker"
        assert entry.status == RadarStatus.APPROVED
        assert entry.category == "banking"
    
    def test_under_review_tools_loaded(self, radar):
        """Test that under_review tools are loaded."""
        entry = radar.get_status("product_search")
        assert entry.name == "product_search"
        assert entry.status == RadarStatus.UNDER_REVIEW
        assert "security review" in entry.notes.lower()
    
    def test_stop_tools_loaded(self, radar):
        """Test that blocked (stop) tools are loaded."""
        entry = radar.get_status("external_pricing_api")
        assert entry.name == "external_pricing_api"
        assert entry.status == RadarStatus.STOP
        assert entry.notes != ""


class TestToolStatusLookup:
    """Test tool status lookup methods."""
    
    def test_get_status_existing_tool(self, radar):
        """Test getting status for an existing tool."""
        entry = radar.get_status("order_status")
        assert entry.name == "order_status"
        assert entry.status == RadarStatus.APPROVED
    
    def test_get_status_unknown_tool(self, radar):
        """Test that unknown tools return STOP status."""
        entry = radar.get_status("unknown_tool")
        assert entry.name == "unknown_tool"
        assert entry.status == RadarStatus.STOP
        assert "Unknown tool" in entry.notes
    
    def test_is_approved_for_approved_tool(self, radar):
        """Test is_approved returns True for approved tools."""
        assert radar.is_approved("loan_checker") is True
        assert radar.is_approved("order_status") is True
    
    def test_is_approved_for_under_review_tool(self, radar):
        """Test is_approved returns False for under_review tools."""
        assert radar.is_approved("product_search") is False
    
    def test_is_approved_for_blocked_tool(self, radar):
        """Test is_approved returns False for blocked tools."""
        assert radar.is_approved("external_pricing_api") is False
    
    def test_is_blocked_for_stop_tool(self, radar):
        """Test is_blocked returns True for STOP tools."""
        assert radar.is_blocked("external_pricing_api") is True
    
    def test_is_blocked_for_approved_tool(self, radar):
        """Test is_blocked returns False for approved tools."""
        assert radar.is_blocked("loan_checker") is False


class TestEnforcement:
    """Test enforcement logic."""
    
    def test_check_and_enforce_approved_tool(self, radar):
        """Test that approved tools pass enforcement."""
        entry = radar.check_and_enforce("loan_checker")
        assert entry.status == RadarStatus.APPROVED
    
    def test_check_and_enforce_under_review_tool(self, radar):
        """Test that under_review tools pass with warning."""
        # Should not raise, returns entry for logging
        entry = radar.check_and_enforce("product_search")
        assert entry.status == RadarStatus.UNDER_REVIEW
    
    def test_check_and_enforce_blocked_tool(self, radar):
        """Test that blocked tools raise RadarBlockedError."""
        with pytest.raises(RadarBlockedError) as exc_info:
            radar.check_and_enforce("external_pricing_api")
        
        assert "blocked by tech radar" in str(exc_info.value).lower()
        assert "external_pricing_api" in str(exc_info.value)
    
    def test_check_and_enforce_unknown_tool(self, radar):
        """Test that unknown tools are blocked."""
        with pytest.raises(RadarBlockedError):
            radar.check_and_enforce("completely_unknown_tool")


class TestListingMethods:
    """Test radar listing and filtering methods."""
    
    def test_list_all_entries(self, radar):
        """Test listing all radar entries."""
        all_entries = radar.list_all()
        assert len(all_entries) > 0
        assert all(hasattr(entry, "name") for entry in all_entries)
        assert all(hasattr(entry, "status") for entry in all_entries)
    
    def test_list_by_status_approved(self, radar):
        """Test filtering by APPROVED status."""
        approved = radar.list_by_status(RadarStatus.APPROVED)
        assert len(approved) > 0
        assert all(entry.status == RadarStatus.APPROVED for entry in approved)
        
        # Verify known approved tools are included
        names = [entry.name for entry in approved]
        assert "loan_checker" in names
        assert "order_status" in names
    
    def test_list_by_status_under_review(self, radar):
        """Test filtering by UNDER_REVIEW status."""
        under_review = radar.list_by_status(RadarStatus.UNDER_REVIEW)
        assert len(under_review) > 0
        assert all(entry.status == RadarStatus.UNDER_REVIEW for entry in under_review)
        
        names = [entry.name for entry in under_review]
        assert "product_search" in names
    
    def test_list_by_status_stop(self, radar):
        """Test filtering by STOP status."""
        blocked = radar.list_by_status(RadarStatus.STOP)
        assert len(blocked) > 0
        assert all(entry.status == RadarStatus.STOP for entry in blocked)
        
        names = [entry.name for entry in blocked]
        assert "external_pricing_api" in names
    
    def test_list_by_category_banking(self, radar):
        """Test filtering by banking category."""
        banking_tools = radar.list_by_category("banking")
        assert len(banking_tools) > 0
        assert all(entry.category == "banking" for entry in banking_tools)
        
        names = [entry.name for entry in banking_tools]
        assert "loan_checker" in names
    
    def test_list_by_category_retail(self, radar):
        """Test filtering by retail category."""
        retail_tools = radar.list_by_category("retail")
        assert len(retail_tools) > 0
        assert all(entry.category == "retail" for entry in retail_tools)
        
        names = [entry.name for entry in retail_tools]
        assert "order_status" in names
        assert "recommendation_engine" in names
