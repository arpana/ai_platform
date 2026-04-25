"""
Tech Radar Registry - Technology governance and approval tracking.

Loads and enforces technology approval status from tech_radar.yaml:
- approved: Tool is allowed for use
- under_review: Tool can be used with warnings logged
- stop: Tool is blocked from execution
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ai_platform.core.models import RadarEntry, RadarStatus
from ai_platform.core.exceptions import RadarBlockedError


class TechRadar:
    """
    Tech Radar registry for technology governance.
    
    Loads tool approval status from YAML configuration and provides
    methods to check tool approval status and enforce governance.
    
    Attributes:
        config_path: Path to tech_radar.yaml file
        radar: Dictionary mapping tool names to RadarEntry objects
    """
    
    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize the Tech Radar.
        
        Args:
            config_path: Path to tech_radar.yaml file.
                        Defaults to configs/radar/tech_radar.yaml
        """
        if config_path is None:
            # Default to configs/radar/ in workspace root
            workspace_root = Path(__file__).parent.parent.parent.parent
            config_path = workspace_root / "configs" / "radar" / "tech_radar.yaml"
        
        self.config_path = Path(config_path)
        self.radar: dict[str, RadarEntry] = {}
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Tech radar config not found: {self.config_path}")
        
        self._load_radar()
    
    def _load_radar(self) -> None:
        """Load radar entries from YAML configuration."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict) or "tools" not in config:
            raise ValueError(f"Invalid radar config: expected 'tools' key in {self.config_path}")
        
        tools = config["tools"]
        
        for tool_name, tool_config in tools.items():
            status_str = tool_config.get("status", "stop")
            
            # Convert string to RadarStatus enum
            try:
                status = RadarStatus(status_str)
            except ValueError:
                # Invalid status defaults to STOP for safety
                status = RadarStatus.STOP
            
            entry = RadarEntry(
                name=tool_name,
                status=status,
                category=tool_config.get("category", ""),
                notes=tool_config.get("notes", "")
            )
            
            self.radar[tool_name] = entry
    
    def get_status(self, tool_name: str) -> RadarEntry:
        """
        Get radar status for a tool.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            RadarEntry with tool status
        """
        if tool_name not in self.radar:
            # Unknown tools default to STOP for safety
            return RadarEntry(
                name=tool_name,
                status=RadarStatus.STOP,
                category="",
                notes="Unknown tool - not in tech radar"
            )
        
        return self.radar[tool_name]
    
    def is_approved(self, tool_name: str) -> bool:
        """
        Check if a tool is approved for use.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool is approved, False otherwise
        """
        entry = self.get_status(tool_name)
        return entry.status == RadarStatus.APPROVED
    
    def is_blocked(self, tool_name: str) -> bool:
        """
        Check if a tool is blocked (STOP status).
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool is blocked, False otherwise
        """
        entry = self.get_status(tool_name)
        return entry.status == RadarStatus.STOP
    
    def check_and_enforce(self, tool_name: str) -> RadarEntry:
        """
        Check tool status and raise exception if blocked.
        
        Enforcement logic:
        - approved: Allow (return entry)
        - under_review: Allow with warning (return entry)
        - stop: Block (raise RadarBlockedError)
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            RadarEntry if tool is allowed (approved or under_review)
            
        Raises:
            RadarBlockedError: If tool is blocked (STOP status)
        """
        entry = self.get_status(tool_name)
        
        if entry.status == RadarStatus.STOP:
            message = f"Tool '{tool_name}' is blocked by tech radar"
            if entry.notes:
                message += f": {entry.notes}"
            
            raise RadarBlockedError(
                message=message,
                tool_name=tool_name,
                status=entry.status.value
            )
        
        # approved or under_review - allow with entry returned for logging
        return entry
    
    def list_all(self) -> list[RadarEntry]:
        """
        Get all radar entries.
        
        Returns:
            List of all RadarEntry objects
        """
        return list(self.radar.values())
    
    def list_by_status(self, status: RadarStatus) -> list[RadarEntry]:
        """
        Get all tools with a specific status.
        
        Args:
            status: RadarStatus to filter by
            
        Returns:
            List of RadarEntry objects with the given status
        """
        return [entry for entry in self.radar.values() if entry.status == status]
    
    def list_by_category(self, category: str) -> list[RadarEntry]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Category name (e.g., "banking", "retail")
            
        Returns:
            List of RadarEntry objects in the category
        """
        return [entry for entry in self.radar.values() if entry.category == category]
