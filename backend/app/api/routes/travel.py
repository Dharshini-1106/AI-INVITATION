"""Google Routes travel planning endpoint."""
import logging

from fastapi import APIRouter, HTTPException

from ...schemas.travel import TravelPlanRequest, TravelPlanResponse
from ...services.travel import TravelPlanningError, plan_travel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["travel"])


@router.post("/travel/plan", response_model=TravelPlanResponse)
def create_travel_plan(request: TravelPlanRequest):
    """Calculate a route only after the user requests travel planning."""
    try:
        return plan_travel(request)
    except TravelPlanningError as exc:
        logger.warning("Travel planning rejected: %s", exc.code)
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "code": exc.code,
                "message": exc.message,
                "google_maps_url": exc.google_maps_url,
            },
        ) from exc
