"""
Artisan Profile Management API Endpoints

This module handles all artisan-related operations including profile management,
search functionality, trust score calculations, and public profile access.

Key Features:
- CRUD operations for artisan profiles with multi-tenant support
- Advanced search with ElasticSearch integration and SQL fallback
- Trust score system with tiered access based on subscription plans
- SIRET verification and business validation
- Rate-limited contact system to prevent spam
- Geospatial search capabilities using PostGIS

Security Features:
- SQL injection prevention through parameterized queries
- Integer overflow protection on pagination
- Rate limiting on sensitive endpoints
- Subscription-based feature access control
"""

from typing import Optional, Any, List
from uuid import UUID
from fastapi import (
    APIRouter, Depends, HTTPException, status,
    Query, Path, BackgroundTasks, Request
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from pydantic import ValidationError
import asyncio
import logging

from app.core.database import get_db
from app.api.deps import (
    get_current_active_user,
    get_current_artisan_profile,
    require_feature,
    get_optional_current_user,
    is_admin
)
from app.models.user import Utilisateur, SubscriptionPlan
from app.models.ArtisanProfile import ArtisanProfile
from app.models.adresse import Adresse
from app.crud.crud_artisan import crud_artisan
from app.schemas.artisan import (
    ArtisanProfilCreate,
    ArtisanProfilUpdate,
    ArtisanProfil as ArtisanProfilSchema,
    SiretCheckResponse
)
from app.schemas.search import (
    SearchResponse,
    ArtisanPublicResponse,
    ContactRequest,
    ContactResponse,
    ArtisanSearchResult
)
from app.schemas.trust_score import (
    TrustScoreBreakdown,
    TrustScoreRecalculResponse,
    TrustScoreSummary
)
from app.services.artisan_service import obtenir_artisan_service
from app.services.search_service import search_service
from app.services.trust_score_service import obtenir_trust_score_service
from app.core.config import settings
from app.core.redis_client import get_redis_client
from app.core.limiter import limiter
from app.workers.artisan_tasks import geocode_adresse_task

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/artisans",
    tags=["artisans"],
    responses={403: {"description": "Forbidden"}}
)

artisan_service = obtenir_artisan_service()

# Pagination and validation limits
MAX_PAGE_NUMBER = 1000000
MAX_LIMIT = 100
MAX_INT_32 = 2147483647
DEFAULT_LIMIT = 20


@router.post("/profile", response_model=ArtisanProfilSchema, status_code=status.HTTP_201_CREATED)
async def create_artisan_profile(
    profile_data: ArtisanProfilCreate,
    background_tasks: BackgroundTasks,
    current_user: Utilisateur = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Create an artisan profile for the authenticated user.
    
    Requirements:
    - User must be of type 'artisan'
    - Company name and valid SIRET number are mandatory
    - Only one profile per user is allowed
    
    Background tasks triggered:
    - SIRET verification with INSEE API
    - Address geocoding if provided
    """
    try:
        if is_admin(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrators cannot create artisan profiles"
            )
        
        if current_user.type_profil != "artisan":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only users with 'artisan' profile type can create a professional profile"
            )
        
        if not profile_data.nom_entreprise or profile_data.nom_entreprise.strip() == "":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Company name is required"
            )
        
        if not profile_data.numero_siret or len(profile_data.numero_siret) != 14:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="SIRET number must be exactly 14 digits"
            )
        
        if profile_data.numero_siret == "00000000000000":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid SIRET number"
            )
        
        if not db.is_active:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database connection error"
            )
        
        return await artisan_service.create_profile(
            db=db,
            current_user=current_user,
            profile_data=profile_data,
            background_tasks=background_tasks,
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile creation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while creating the profile"
        )


@router.get("/profile", response_model=ArtisanProfilSchema)
async def get_current_artisan_profile(
    current_user: Utilisateur = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Retrieve the current artisan's profile with all related data."""
    try:
        if is_admin(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin users don't have artisan profiles"
            )
        
        if current_user.type_profil != "artisan":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint is restricted to artisan users"
            )
        
        profile = await artisan_service.get_current_profile(db, current_user)
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No artisan profile found. Please create your profile first."
            )
        
        if not hasattr(profile, "competences"):
            profile = await artisan_service._reload_profile_with_relations(db, profile.id)
        
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving profile: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving profile"
        )


@router.put("/profile", response_model=ArtisanProfilSchema)
async def update_artisan_profile(
    profile_update: ArtisanProfilUpdate,
    background_tasks: BackgroundTasks,
    current_user: Utilisateur = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Update the current artisan's profile.
    
    Partial updates are supported - only provided fields will be updated.
    Background tasks may be triggered for address geocoding or verification updates.
    """
    try:
        if is_admin(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin users don't have artisan profiles"
            )
        
        if current_user.type_profil != "artisan":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint is restricted to artisan users"
            )
        
        if not db.is_active:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database connection error"
            )
        
        current_profile = await artisan_service.get_current_profile(db, current_user)
        
        return await artisan_service.update_profile(
            db=db,
            current_user=current_user,
            profile_update=profile_update,
            background_tasks=background_tasks,
            existing_profile=current_profile,
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating profile"
        )


@router.post("/profile/geocode", status_code=status.HTTP_202_ACCEPTED)
async def geocode_artisan_address(
    background_tasks: BackgroundTasks,
    current_user: Utilisateur = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """
    Trigger geocoding of the artisan's address.
    
    Returns immediately with 202 Accepted status.
    The geocoding process runs asynchronously in the background.
    """
    try:
        if is_admin(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin users don't have addresses to geocode"
            )
        
        if current_user.type_profil != "artisan":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint is restricted to artisan users"
            )
        
        profile = await artisan_service.get_current_profile(db, current_user)
        
        if not hasattr(profile, "adresses") or profile.adresses is None:
            result = await db.execute(
                select(ArtisanProfile)
                .where(ArtisanProfile.id == profile.id)
                .options(selectinload(ArtisanProfile.adresses))
            )
            profile = result.scalar_one_or_none()
        
        if not profile or not profile.adresses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No address found to geocode"
            )
        
        adresse = next(
            (addr for addr in profile.adresses if addr.type_adresse == "siege_social"),
            profile.adresses[0] if profile.adresses else None
        )
        
        if not adresse:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid address to geocode"
            )
        
        geocode_adresse_task.delay(str(adresse.id))
        
        return {
            "message": "Geocoding started in background",
            "status": "accepted"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Geocoding error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error initiating geocoding"
        )


@router.get("/profile/trust-score/breakdown", response_model=TrustScoreBreakdown)
async def get_trust_score_breakdown(
    current_user: Utilisateur = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> TrustScoreBreakdown:
    """
    Get detailed trust score breakdown.
    
    Premium Feature: Requires PRO or PERFORMANCE subscription plan.
    
    Returns:
    - Score breakdown by category
    - Personalized recommendations
    - Historical evolution
    - Industry comparison
    """
    try:
        if is_admin(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin users don't have trust scores"
            )
        
        if current_user.type_profil != "artisan":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint is restricted to artisan users"
            )
        
        # Premium feature check
        if hasattr(current_user, "subscription_plan"):
            if current_user.subscription_plan not in [SubscriptionPlan.PRO, SubscriptionPlan.PERFORMANCE]:
                profile = await artisan_service.get_current_profile(db, current_user)
                basic_score = profile.trust_score if profile else 0
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "message": "Detailed trust score analysis requires PRO plan (€149/month) or higher",
                        "error_code": "FEATURE_REQUIRES_UPGRADE",
                        "feature": "trust_score_breakdown",
                        "required_plan": "PRO",
                        "current_plan": current_user.subscription_plan.value,
                        "upgrade_url": "/billing/plans",
                        "your_basic_score": basic_score,
                        "premium_benefits": [
                            "Complete score breakdown by category",
                            "Personalized improvement recommendations",
                            "Historical trust score evolution",
                            "Industry average comparison"
                        ]
                    }
                )
        
        if hasattr(current_user, "can_access_feature"):
            if not current_user.can_access_feature("trust_score_advanced"):
                profile = await artisan_service.get_current_profile(db, current_user)
                basic_score = profile.trust_score if profile else 0
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "message": "This feature requires a PRO subscription or higher",
                        "error_code": "FEATURE_REQUIRES_UPGRADE",
                        "feature": "trust_score_breakdown",
                        "required_plan": "PRO",
                        "your_basic_score": basic_score,
                        "upgrade_url": "/billing/plans"
                    }
                )
        
        profile = await artisan_service.get_current_profile(db, current_user)
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artisan profile not found"
            )
        
        breakdown = await artisan_service.get_trust_score_breakdown(db, profile.id)
        return TrustScoreBreakdown(**breakdown)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trust score breakdown error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving trust score breakdown"
        )


@router.post("/profile/trust-score/recalculate", response_model=TrustScoreRecalculResponse)
async def recalculate_trust_score(
    current_user: Utilisateur = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> TrustScoreRecalculResponse:
    """
    Force manual trust score recalculation.
    
    Rate limited to 10 requests per hour to prevent abuse.
    Basic recalculation is available to all users.
    Detailed analysis requires PRO+ subscription.
    """
    try:
        if is_admin(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin users don't have trust scores"
            )
        
        if current_user.type_profil != "artisan":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint is restricted to artisan users"
            )
        
        profile = await artisan_service.get_current_profile(db, current_user)
        
        # Rate limiting implementation
        redis = await get_redis_client()
        if redis:
            rate_key = f"trust_recalc:{current_user.id}"
            count = await redis.incr(rate_key)
            if count == 1:
                await redis.expire(rate_key, 3600)
            if count > 10:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded (10 recalculations per hour)"
                )
        
        result = await artisan_service.recalculate_trust_score(db, profile.id)
        
        if hasattr(current_user, "subscription_plan"):
            if current_user.subscription_plan not in [SubscriptionPlan.PRO, SubscriptionPlan.PERFORMANCE]:
                result["note"] = "Score recalculated. Upgrade to PRO plan for detailed analysis."
        
        return TrustScoreRecalculResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trust score recalculation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error recalculating trust score"
        )


@router.get("/{artisan_id}/trust-score/summary", response_model=TrustScoreSummary)
async def get_trust_score_summary(
    artisan_id: UUID = Path(..., description="Artisan ID"),
    db: AsyncSession = Depends(get_db),
) -> TrustScoreSummary:
    """
    Get public trust score summary for any artisan.
    
    Public endpoint - no authentication required.
    Returns only the overall score and basic metrics.
    """
    try:
        artisan = await crud_artisan.get(
            db, artisan_id, 
            load_relationships=["documents_verification"]
        )
        
        if not artisan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artisan not found"
            )
        
        trust_service = obtenir_trust_score_service()
        niveau = trust_service._determiner_niveau(artisan.trust_score)
        niveau_suivant = trust_service._obtenir_niveau_suivant(niveau)
        
        progression = None
        if niveau_suivant:
            score_actuel_dans_niveau = artisan.trust_score - trust_service.SEUILS_NIVEAUX[niveau]
            score_necessaire = niveau_suivant["score_requis"] - trust_service.SEUILS_NIVEAUX[niveau]
            if score_necessaire > 0:
                progression = (score_actuel_dans_niveau / score_necessaire) * 100
        
        badges_count = len([b for b, v in (artisan.badges_json or {}).items() if v])
        
        return TrustScoreSummary(
            score=artisan.trust_score,
            niveau=niveau.value,
            progression_niveau_suivant=progression,
            badges_count=badges_count,
            derniere_mise_a_jour=(
                artisan.updated_at.isoformat() if artisan.updated_at 
                else artisan.created_at.isoformat()
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trust score summary error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving trust score summary"
        )


@router.get("/profile/{artisan_id}", response_model=ArtisanProfilSchema)
async def get_artisan_profile_by_id(
    artisan_id: UUID,
    current_user: Utilisateur = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Retrieve any artisan profile by ID.
    
    Admin only endpoint for profile management and support.
    Respects multi-tenant boundaries unless system admin.
    """
    try:
        if not is_admin(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator access required"
            )
        
        if artisan_id == UUID("00000000-0000-0000-0000-000000000000"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid artisan ID"
            )
        
        artisan_profile = await crud_artisan.get(
            db, artisan_id,
            load_relationships=["competences", "adresses", "utilisateur"]
        )
        
        if not artisan_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artisan profile not found"
            )
        
        # Multi-tenant check
        if not is_admin(current_user) and artisan_profile.tenant_id != current_user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artisan profile not found in your tenant"
            )
        
        return artisan_profile
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid UUID format"
        )
    except Exception as e:
        logger.error(f"Error retrieving profile by ID: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving profile"
        )


@router.get("/check-siret/{siret}", response_model=SiretCheckResponse)
async def check_siret_availability(
    siret: str = Path(..., pattern="^[0-9]{14}$", description="SIRET number to check"),
    db: AsyncSession = Depends(get_db),
) -> SiretCheckResponse:
    """
    Check if a SIRET number is already registered.
    
    Public endpoint for registration validation.
    Returns company name if SIRET is already in use.
    """
    try:
        if not siret.isdigit() or len(siret) != 14:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="SIRET must be exactly 14 digits"
            )
        
        # Timeout protection for DB queries
        try:
            from async_timeout import timeout
            
            async def check_with_timeout():
                async with timeout(5):
                    return await crud_artisan.check_siret_exists_global(db, siret)
            
            exists = await check_with_timeout()
            
        except (ImportError, asyncio.TimeoutError):
            # Fallback without timeout
            exists = await crud_artisan.check_siret_exists_global(db, siret)
        
        if exists:
            result = await db.execute(
                select(ArtisanProfile.company_name)
                .where(ArtisanProfile.siret_number == siret)
                .limit(1)
            )
            company_name = result.scalar_one_or_none()
            
            return SiretCheckResponse(
                available=False,
                message="This SIRET number is already registered",
                existing_company_name=company_name
            )
        
        return SiretCheckResponse(
            available=True,
            message="This SIRET number is available",
            existing_company_name=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SIRET check error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error checking SIRET availability"
        )


@router.get("/search", response_model=SearchResponse)
async def search_artisans(
    query: Optional[str] = Query(None, description="Text search query"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Latitude"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Longitude"),
    distance_km: int = Query(30, ge=1, le=200, description="Search radius in kilometers"),
    competences: Optional[List[str]] = Query(None, description="Filter by skills"),
    badges: Optional[List[str]] = Query(None, description="Filter by badges"),
    note_min: Optional[float] = Query(None, ge=0, le=5, description="Minimum rating"),
    page: int = Query(1, ge=1, le=MAX_PAGE_NUMBER, description="Page number"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Results per page"),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    Search for artisans with advanced filters.
    
    Public endpoint with ElasticSearch integration for semantic search.
    Falls back to SQL if ElasticSearch is unavailable.
    
    Features:
    - Geospatial search with PostGIS
    - Skill and badge filtering
    - Rating-based filtering
    - Pagination with overflow protection
    """
    try:
        # Overflow protection
        if page > MAX_PAGE_NUMBER:
            page = MAX_PAGE_NUMBER
        
        if limit > MAX_LIMIT:
            limit = MAX_LIMIT
        
        offset = (page - 1) * limit
        if offset > MAX_INT_32:
            return SearchResponse(
                artisans=[], 
                total=0, 
                page=page, 
                limit=limit, 
                pages=0, 
                took_ms=0
            )
        
        # Clean null string values
        if competences and len(competences) == 1 and competences[0] == "null":
            competences = None
        if badges and len(badges) == 1 and badges[0] == "null":
            badges = None
        
        result = await search_service.search_artisans(
            query=query,
            lat=lat,
            lon=lon,
            distance_km=distance_km,
            competences=competences,
            badges=badges,
            note_min=note_min,
            limit=limit,
            offset=offset,
            db=db
        )
        
        return _transform_search_results(result, page, limit)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return SearchResponse(
            artisans=[],
            total=0,
            page=page,
            limit=limit,
            pages=0,
            took_ms=None
        )


@router.get("/public/{slug}", response_model=ArtisanPublicResponse)
async def get_public_artisan_profile(
    slug: str = Path(..., description="SEO-friendly slug"),
    db: AsyncSession = Depends(get_db),
) -> ArtisanPublicResponse:
    """
    Get public profile of an artisan by slug.
    
    Public endpoint for SEO-optimized profile pages.
    Verification badges visibility depends on subscription plan.
    """
    try:
        profile_data = await artisan_service.get_public_profile_by_slug(db, slug)
        
        result = await db.execute(
            select(ArtisanProfile)
            .options(selectinload(ArtisanProfile.utilisateur))
            .where(ArtisanProfile.slug == slug)
        )
        artisan_profile = result.scalar_one_or_none()
        
        if not artisan_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No artisan found with slug '{slug}'"
            )
        
        # Check verified badge visibility based on subscription
        is_verified_public = False
        if artisan_profile.utilisateur:
            if (artisan_profile.utilisateur.can_access_feature("verified_badge_access") 
                and artisan_profile.is_verified):
                is_verified_public = True
        
        if hasattr(profile_data, "is_verified"):
            profile_data.is_verified = is_verified_public
        elif isinstance(profile_data, dict):
            profile_data["is_verified"] = is_verified_public
        
        # Hide verification badges for non-premium users
        if hasattr(profile_data, "badges") and not is_verified_public:
            verification_badges = [
                "siret_verified", "rge_verified", "assurance_decennale",
                "assurance_rc_pro", "qualibat"
            ]
            if isinstance(profile_data.badges, dict):
                profile_data.badges = {
                    k: v for k, v in profile_data.badges.items()
                    if k not in verification_badges
                }
            elif isinstance(profile_data.badges, list):
                profile_data.badges = [
                    badge for badge in profile_data.badges
                    if badge not in verification_badges
                ]
        
        return profile_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Public profile error for {slug}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving profile"
        )


@router.post("/contact/{artisan_id}", response_model=ContactResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/hour")
async def contact_artisan(
    artisan_id: UUID = Path(..., description="Artisan ID to contact"),
    contact_data: ContactRequest = ...,
    background_tasks: BackgroundTasks = ...,
    current_user: Optional[Utilisateur] = Depends(get_optional_current_user),
    request: Request = ...,
    db: AsyncSession = Depends(get_db),
) -> ContactResponse:
    """
    Send a contact request to an artisan.
    
    Public endpoint with rate limiting to prevent spam.
    Tracks requests if user is authenticated.
    """
    try:
        request_ip = request.client.host if request.client else None
        
        return await artisan_service.handle_contact_request(
            db=db,
            artisan_id=artisan_id,
            contact_data=contact_data,
            background_tasks=background_tasks,
            current_user=current_user,
            request_ip=request_ip
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Contact error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error sending contact request"
        )


def _transform_search_results(result: dict, page: int, limit: int) -> SearchResponse:
    """
    Transform service search results to HTTP response format.
    
    Handles badge visibility based on subscription plans.
    """
    from uuid import uuid4
    
    total = result.get("total", 0)
    pages = (total + limit - 1) // limit if total > 0 else 0
    
    artisans = []
    for hit in result.get("artisans", []):
        try:
            # Control badge visibility based on subscription
            if "user_subscription" in hit and "badges" in hit:
                user_subscription = hit.get("user_subscription", "free")
                can_show_verification_badges = user_subscription in ["pro", "premium", "enterprise"]
                
                if not can_show_verification_badges:
                    verification_badges = [
                        "siret_verified", "rge_verified", "assurance_decennale",
                        "assurance_rc_pro", "qualibat"
                    ]
                    if isinstance(hit["badges"], dict):
                        hit["badges"] = {
                            k: v for k, v in hit["badges"].items()
                            if k not in verification_badges
                        }
                    elif isinstance(hit["badges"], list):
                        hit["badges"] = [
                            badge for badge in hit["badges"]
                            if badge not in verification_badges
                        ]
                    hit["is_verified"] = False
            
            artisan_data = {
                "id": str(hit.get("id") or hit.get("_id", uuid4())),
                "utilisateur_id": str(hit.get("user_id") or hit.get("utilisateur_id", "")),
                "slug": hit.get("slug", ""),
                "nom_entreprise": hit.get("company_name", ""),
                "description_entreprise": hit.get("company_description"),
                "specialites": hit.get("specialties"),
                "ville": hit.get("ville"),
                "code_postal": hit.get("code_postal"),
                "departement": hit.get("departement"),
                "distance_km": hit.get("distance_km"),
                "trust_score": float(hit.get("trust_score", 0)),
                "note_moyenne": hit.get("average_rating"),
                "nombre_avis": int(hit.get("review_count", 0)),
                "est_verifie": bool(hit.get("is_verified", False)),
                "annees_experience": hit.get("years_experience"),
                "badges": hit.get("badges", []),
                "competences": hit.get("competences", []),
                "zones_intervention": hit.get("intervention_zones"),
                "rayon_intervention_km": hit.get("intervention_radius_km")
            }
            
            artisan = ArtisanSearchResult(**artisan_data)
            artisans.append(artisan)
            
        except Exception as e:
            logger.warning(f"Result transformation error: {e}")
            continue
    
    return SearchResponse(
        artisans=artisans,
        total=total,
        page=page,
        limit=limit,
        pages=pages,
        took_ms=result.get("took_ms"),
        search_metadata={
            "search_type": result.get("search_type", "unknown"),
            "took_ms": result.get("took_ms")
        }
    )
