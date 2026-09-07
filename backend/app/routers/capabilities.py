from fastapi import APIRouter, Depends

from app.agent.registry import list_tools
from app.schemas import CapabilitiesResponse
from app.security import require_api_key

router = APIRouter(prefix="/api/v1", tags=["capabilities"])


@router.get("/tools", response_model=CapabilitiesResponse)
async def get_capabilities(_: str = Depends(require_api_key)):
    """Lists the predefined tool registry the agentic controller selects from."""
    return CapabilitiesResponse(tools=list_tools())
