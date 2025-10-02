"""
Advanced CRUD Operations for Artisan Marketplace

Implements a hybrid public/private data access pattern for a multi-tenant
marketplace with global search capabilities and tenant-isolated private data.

Key Features:
- Global marketplace with unique slugs across all tenants
- Tenant isolation for private business data
- Optimized search with multiple filtering options
- Prevention of N+1 queries through eager loading
- Robust error handling and SQLAlchemy best practices

Architecture: Repository pattern with async/sync support
Database: PostgreSQL with full-text search and JSONB
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload, joinedload
from sqlalchemy.future import select
from sqlalchemy import func, and_, or_, exists, case
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import logging

from app.models.ArtisanProfile import ArtisanProfile
from app.models.competence import Competence
from app.models.adresse import Adresse
from app.crud.base import TenantAwareCRUD
from app.schemas.artisan import ArtisanProfilCreate, ArtisanProfilUpdate

logger = logging.getLogger(__name__)


class CRUDArtisan(TenantAwareCRUD[ArtisanProfile]):
    """
    Repository for artisan profiles implementing marketplace pattern.
    
    Public operations: Search and view profiles across all tenants
    Private operations: Manage profiles within tenant boundaries
    """
    
    # ========== PRIVATE TENANT-SCOPED OPERATIONS ==========
    
    async def get_by_user_id(
        self, 
        db: AsyncSession, 
        user_id: UUID, 
        tenant_id: UUID
    ) -> Optional[ArtisanProfile]:
        """
        Retrieve artisan profile by user ID within tenant scope.
        
        Security: Enforces tenant isolation for private data access.
        """
        try:
            result = await db.execute(
                select(self.model)
                .options(
                    selectinload(self.model.competences),
                    selectinload(self.model.adresses),
                    selectinload(self.model.documents_verification),
                    selectinload(self.model.utilisateur)
                )
                .where(
                    and_(
                        self.model.user_id == user_id,
                        self.model.tenant_id == tenant_id
                    )
                )
            )
            return result.unique().scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving artisan by user_id {user_id}: {e}")
            return None

    async def get_by_siret(
        self, 
        db: AsyncSession, 
        siret: str, 
        tenant_id: UUID
    ) -> Optional[ArtisanProfile]:
        """
        Retrieve artisan profile by SIRET number within tenant scope.
        """
        try:
            result = await db.execute(
                select(self.model)
                .options(
                    selectinload(self.model.utilisateur),
                    selectinload(self.model.competences)
                )
                .where(
                    and_(
                        self.model.siret_number == siret,
                        self.model.tenant_id == tenant_id
                    )
                )
            )
            return result.unique().scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving artisan by SIRET {siret}: {e}")
            return None

    # ========== PUBLIC MARKETPLACE OPERATIONS ==========
    
    async def get_by_slug_public(
        self, 
        db: AsyncSession, 
        slug: str
    ) -> Optional[ArtisanProfile]:
        """
        Retrieve artisan profile by slug (public marketplace access).
        
        No tenant filtering - slugs are globally unique across the platform.
        """
        try:
            result = await db.execute(
                select(self.model)
                .options(
                    selectinload(self.model.utilisateur),
                    selectinload(self.model.competences),
                    selectinload(self.model.adresses),
                    selectinload(self.model.documents_verification)
                )
                .where(self.model.slug == slug)
            )
            return result.unique().scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving artisan by slug {slug}: {e}")
            return None

    async def search_artisans_public(
        self,
        db: AsyncSession,
        *,
        search_term: Optional[str] = None,
        ville: Optional[str] = None,
        code_postal: Optional[str] = None,
        competences: Optional[List[str]] = None,
        specialties: Optional[List[str]] = None,
        note_min: Optional[float] = None,
        trust_score_min: Optional[float] = None,
        skip: int = 0,
        limit: int = 20
    ) -> List[ArtisanProfile]:
        """
        Search artisans across all tenants with multiple filters.
        
        Implements full-text search with PostgreSQL and supports
        geographic, skill-based, and quality-based filtering.
        """
        try:
            query = (
                select(self.model)
                .options(
                    selectinload(self.model.utilisateur),
                    selectinload(self.model.competences),
                    selectinload(self.model.adresses)
                )
            )
            
            # Text search across multiple fields
            if search_term:
                search_pattern = f"%{search_term}%"
                query = query.where(
                    or_(
                        self.model.company_name.ilike(search_pattern),
                        self.model.company_description.ilike(search_pattern),
                        self.model.specialties.ilike(search_pattern)
                    )
                )
            
            # Specialty filter using PostgreSQL arrays
            if specialties:
                query = query.where(
                    self.model.specialties.overlap(specialties)
                )
            
            # Geographic filters using relationships
            if ville:
                query = query.where(
                    self.model.adresses.any(
                        Adresse.ville.ilike(f"%{ville}%")
                    )
                )
            
            if code_postal:
                query = query.where(
                    or_(
                        self.model.intervention_zones.contains([code_postal]),
                        self.model.adresses.any(
                            Adresse.code_postal == code_postal
                        )
                    )
                )
            
            # Skill-based filtering
            if competences:
                query = query.where(
                    self.model.competences.any(
                        Competence.nom.in_(competences)
                    )
                )
            
            # Quality filters
            if note_min is not None:
                query = query.where(
                    self.model.average_rating >= note_min
                )
            
            if trust_score_min is not None:
                query = query.where(
                    self.model.trust_score >= trust_score_min
                )
            
            # Ranking: verified profiles first, then by trust score
            query = query.order_by(
                self.model.is_verified.desc(),
                self.model.trust_score.desc()
            )
            
            # Pagination
            query = query.offset(skip).limit(limit)
            
            result = await db.execute(query)
            return result.unique().scalars().all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error in public artisan search: {e}")
            return []

    # ========== VALIDATION OPERATIONS ==========
    
    async def check_siret_exists_global(
        self,
        db: AsyncSession,
        siret: str,
        exclude_id: Optional[UUID] = None
    ) -> bool:
        """
        Check if SIRET exists globally across all tenants.
        
        Used for registration validation to ensure SIRET uniqueness.
        """
        try:
            subquery = exists().where(self.model.siret_number == siret)
            
            if exclude_id:
                subquery = subquery.where(self.model.id != exclude_id)
            
            result = await db.execute(select(subquery))
            return result.scalar()
            
        except SQLAlchemyError as e:
            logger.error(f"Error checking SIRET existence {siret}: {e}")
            return False

    async def check_slug_exists_global(
        self,
        db: AsyncSession,
        slug: str,
        exclude_id: Optional[UUID] = None
    ) -> bool:
        """
        Check if slug exists globally across all tenants.
        
        Critical for maintaining URL uniqueness in the marketplace.
        """
        try:
            subquery = exists().where(self.model.slug == slug)
            
            if exclude_id:
                subquery = subquery.where(self.model.id != exclude_id)
            
            result = await db.execute(select(subquery))
            return result.scalar()
            
        except SQLAlchemyError as e:
            logger.error(f"Error checking slug existence {slug}: {e}")
            return False

    # ========== STATISTICS OPERATIONS ==========
    
    async def get_global_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """
        Calculate platform-wide statistics in a single optimized query.
        
        Performance: Single query instead of multiple round-trips.
        """
        try:
            result = await db.execute(
                select(
                    func.count(self.model.id).label("total"),
                    func.count(
                        case((self.model.is_verified == True, self.model.id))
                    ).label("verified"),
                    func.count(
                        case((self.model.average_rating.isnot(None), self.model.id))
                    ).label("rated"),
                    func.avg(self.model.trust_score).label("avg_trust")
                )
            )
            
            stats = result.first()
            total = stats.total or 0
            
            return {
                "total_artisans": total,
                "verified_artisans": stats.verified or 0,
                "rated_artisans": stats.rated or 0,
                "average_trust_score": round(float(stats.avg_trust or 0), 1),
                "verification_rate": round(
                    (stats.verified / total * 100) if total > 0 else 0, 2
                ),
                "rating_rate": round(
                    (stats.rated / total * 100) if total > 0 else 0, 2
                )
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error calculating global statistics: {e}")
            return {
                "total_artisans": 0,
                "verified_artisans": 0,
                "rated_artisans": 0,
                "average_trust_score": 0,
                "verification_rate": 0,
                "rating_rate": 0
            }

    # ========== RANKING OPERATIONS ==========
    
    async def get_top_rated_artisans_global(
        self,
        db: AsyncSession,
        limit: int = 10,
        min_reviews: int = 5
    ) -> List[ArtisanProfile]:
        """
        Retrieve top-rated artisans across the platform.
        
        Ensures statistical significance with minimum review threshold.
        """
        try:
            result = await db.execute(
                select(self.model)
                .options(
                    selectinload(self.model.utilisateur),
                    selectinload(self.model.competences),
                    selectinload(self.model.adresses)
                )
                .where(
                    and_(
                        self.model.review_count >= min_reviews,
                        self.model.average_rating.isnot(None)
                    )
                )
                .order_by(self.model.average_rating.desc())
                .limit(limit)
            )
            return result.unique().scalars().all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving top-rated artisans: {e}")
            return []

    async def get_verified_artisans_global(
        self,
        db: AsyncSession,
        limit: int = 20
    ) -> List[ArtisanProfile]:
        """
        Retrieve verified artisans with high trust scores.
        """
        try:
            result = await db.execute(
                select(self.model)
                .options(
                    selectinload(self.model.utilisateur),
                    selectinload(self.model.competences),
                    selectinload(self.model.adresses)
                )
                .where(
                    and_(
                        self.model.is_verified == True,
                        self.model.trust_score >= 50
                    )
                )
                .order_by(self.model.trust_score.desc())
                .limit(limit)
            )
            return result.unique().scalars().all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving verified artisans: {e}")
            return []

    # ========== CREATE OPERATIONS WITH RELATIONSHIP HANDLING ==========
    
    async def create_with_address(
        self,
        db: AsyncSession,
        artisan_data: ArtisanProfilCreate,
        user_id: UUID,
        tenant_id: UUID,
        slug: str
    ) -> ArtisanProfile:
        """
        Create artisan profile with address and skills.
        
        Pattern: Prevents MissingGreenlet errors by managing
        many-to-many relationships after initial commit.
        """
        try:
            # Create main entity without many-to-many relationships
            artisan = ArtisanProfile(
                user_id=user_id,
                tenant_id=tenant_id,
                company_name=artisan_data.nom_entreprise,
                siret_number=artisan_data.numero_siret,
                years_experience=artisan_data.annees_experience,
                founding_year=artisan_data.annee_creation,
                company_description=artisan_data.description_entreprise,
                specialties=artisan_data.specialites,
                intervention_zones=artisan_data.zones_intervention,
                intervention_radius_km=artisan_data.rayon_intervention_km,
                company_size=artisan_data.effectif_entreprise,
                slug=slug,
                badges_json={},
                trust_score=0.0
            )
            
            db.add(artisan)
            await db.flush()
            
            # Create related address
            if artisan_data.adresse_siege:
                await self._create_address(
                    db, artisan.id, artisan_data.adresse_siege
                )
            
            # Commit before handling many-to-many
            await db.commit()
            
            # Add skills in separate transaction
            if artisan_data.competences:
                await self._add_competences_to_artisan(
                    db, artisan.id, artisan_data.competences
                )
            
            # Reload with all relationships
            return await self._reload_artisan_with_relations(db, artisan.id)
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating artisan with address: {e}")
            raise

    async def _add_competences_to_artisan(
        self,
        db: AsyncSession,
        artisan_id: UUID,
        competence_names: List[str]
    ) -> None:
        """
        Add skills to artisan in separate transaction.
        
        Prevents lazy loading issues with many-to-many relationships.
        """
        try:
            async with db.begin():
                # Reload with eager loading
                result = await db.execute(
                    select(ArtisanProfile)
                    .options(selectinload(ArtisanProfile.competences))
                    .where(ArtisanProfile.id == artisan_id)
                )
                artisan = result.scalar_one()
                
                # Get existing skills
                skills = await self.get_competences_by_names(db, competence_names)
                
                # Safe assignment with eager-loaded relationship
                artisan.competences = skills
                await db.flush()
                
        except SQLAlchemyError as e:
            logger.error(f"Error adding skills to artisan {artisan_id}: {e}")
            raise

    async def _reload_artisan_with_relations(
        self,
        db: AsyncSession,
        artisan_id: UUID
    ) -> ArtisanProfile:
        """
        Reload artisan with all relationships using eager loading.
        """
        try:
            await db.commit()
            
            result = await db.execute(
                select(ArtisanProfile)
                .options(
                    selectinload(ArtisanProfile.utilisateur),
                    selectinload(ArtisanProfile.competences),
                    selectinload(ArtisanProfile.adresses),
                    selectinload(ArtisanProfile.documents_verification)
                )
                .where(ArtisanProfile.id == artisan_id)
            )
            return result.scalar_one()
            
        except SQLAlchemyError as e:
            logger.error(f"Error reloading artisan {artisan_id}: {e}")
            raise

    async def _create_address(
        self,
        db: AsyncSession,
        artisan_id: UUID,
        address_data: Dict[str, Any]
    ) -> None:
        """
        Create address for artisan with street number parsing.
        """
        import re
        
        street = address_data.get("rue", "")
        street_number = None
        street_name = street
        
        if street:
            match = re.match(r'^(\d+\w*)\s+(.+)$', street.strip())
            if match:
                street_number = match.group(1)
                street_name = match.group(2)
        
        address = Adresse(
            artisan_profil_id=artisan_id,
            numero_voie=street_number,
            nom_voie=street_name or "Not specified",
            ville=address_data.get("ville", "Not specified"),
            code_postal=address_data.get("code_postal", "00000"),
            type_adresse="siege_social"
        )
        db.add(address)

    # ========== UTILITY OPERATIONS ==========
    
    async def get_competences_by_names(
        self, 
        db: AsyncSession, 
        names: List[str]
    ) -> List[Competence]:
        """Retrieve skills by names."""
        if not names:
            return []
        
        try:
            result = await db.execute(
                select(Competence).where(Competence.nom.in_(names))
            )
            return list(result.scalars().all())
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving skills {names}: {e}")
            return []

    async def get_popular_competences_global(
        self,
        db: AsyncSession,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most popular skills across the platform.
        """
        try:
            result = await db.execute(
                select(
                    Competence.nom,
                    func.count(self.model.id).label('count')
                )
                .join(self.model.competences)
                .group_by(Competence.nom)
                .order_by(func.count(self.model.id).desc())
                .limit(limit)
            )
            
            return [
                {"name": row.nom, "artisan_count": row.count}
                for row in result.fetchall()
            ]
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving popular skills: {e}")
            return []

    # ========== SYNCHRONOUS OPERATIONS FOR WORKERS ==========
    
    def get_sync(
        self,
        db: Session,
        id: UUID,
        load_relationships: List[str] = None
    ) -> Optional[ArtisanProfile]:
        """
        Synchronous get for background workers (Celery).
        """
        try:
            query = db.query(self.model).filter(self.model.id == id)
            
            if load_relationships:
                for rel in load_relationships:
                    if hasattr(self.model, rel):
                        query = query.options(
                            joinedload(getattr(self.model, rel))
                        )
            
            return query.first()
            
        except SQLAlchemyError as e:
            logger.error(f"Error in sync get for artisan {id}: {e}")
            return None
    
    def exists_sync(self, db: Session, id: UUID) -> bool:
        """
        Synchronous existence check for background workers.
        """
        try:
            return db.query(
                db.query(self.model)
                .filter(self.model.id == id)
                .exists()
            ).scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error in sync exists check {id}: {e}")
            return False


# Global CRUD instance
crud_artisan = CRUDArtisan(ArtisanProfile)
