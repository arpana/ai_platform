from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from ai_platform.core.models import RadarEntry
from ai_platform.radar import TechRadar

router = APIRouter(prefix="/radar", tags=["radar"])

# Find workspace root (services/api/routes/radar.py -> ../../../../)
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent


def get_radar_registry() -> TechRadar:
    """
    Get or create the TechRadar instance.
    
    Loads tech radar from configs/radar/tech_radar.yaml.
    """
    config_path = WORKSPACE_ROOT / "configs" / "radar" / "tech_radar.yaml"
    return TechRadar(config_path=config_path)


@router.get("/status", response_model=list[RadarEntry])
def list_radar_entries(radar: TechRadar = Depends(get_radar_registry)):
    """
    List all tools in the tech radar.
    
    Returns all radar entries with their approval status.
    """
    return radar.list_all()


@router.get("/status/{tool_name}", response_model=RadarEntry)
def get_radar_entry(tool_name: str, radar: TechRadar = Depends(get_radar_registry)):
    """
    Get the radar status for a specific tool.
    
    Returns the radar entry for the tool, or a STOP entry if unknown.
    """
    return radar.get_status(tool_name)
