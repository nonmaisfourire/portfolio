"""
Multi-tenant Architecture Model

Implements data isolation and resource management for SaaS applications.
This model demonstrates enterprise-level multi-tenancy with:
- Resource quotas and usage tracking
- Subscription management
- API key authentication
- Configurable feature flags
- Audit logging capabilities

Design Pattern: Row-level security with tenant isolation
Database: PostgreSQL with JSONB for flexible configuration
"""

from datetime import datetime, timedelta
from typing import Optional, List, TYPE_CHECKING, Dict, Any
from sqlalchemy import (
    String, Boolean, DateTime, Integer, JSON, 
    CheckConstraint, Index, event, Enum, text
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property
import uuid
import hashlib
import secrets
import re

from app.models.base import BaseModel
from app.models.enums import TypeAbonnement

if TYPE_CHECKING:
    from app.models.user import Utilisateur
    from app.models.ArtisanProfile import ArtisanProfile
    from app.models.devis import Devis
    from app.models.template_devis import TemplateDevis


class Tenant(BaseModel):
    """
    Core tenant model for multi-tenant SaaS architecture.
    Each tenant represents an isolated organization with its own data and limits.
    """
    __tablename__ = "tenants"
    
    # Core Information
    nom: Mapped[str] = mapped_column(
        String(255), 
        nullable=False,
        index=True,
        comment="Organization name"
    )
    
    code: Mapped[str] = mapped_column(
        String(50), 
        unique=True, 
        nullable=False,
        index=True,
        comment="Unique tenant identifier (e.g., ACME_CORP)"
    )
    
    # Status Management
    est_actif: Mapped[bool] = mapped_column(
        Boolean, 
        server_default='TRUE', 
        nullable=False,
        index=True
    )
    
    date_desactivation: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    raison_desactivation: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True
    )
    
    # Business Information
    numero_siret: Mapped[Optional[str]] = mapped_column(
        String(14),
        nullable=True,
        unique=True,
        comment="French business registration number"
    )
    
    adresse_siege: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True
    )
    
    code_postal: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True
    )
    
    ville: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True
    )
    
    pays: Mapped[str] = mapped_column(
        String(2),
        server_default='FR',
        nullable=False,
        comment="ISO 3166-1 alpha-2 country code"
    )
    
    # Contact Information
    email_contact: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True
    )
    
    telephone_contact: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True
    )
    
    # Resource Quotas
    limite_utilisateurs: Mapped[int] = mapped_column(
        Integer,
        server_default='10',
        nullable=False
    )
    
    limite_artisans: Mapped[int] = mapped_column(
        Integer,
        server_default='100',
        nullable=False
    )
    
    limite_projets_mensuels: Mapped[int] = mapped_column(
        Integer,
        server_default='1000',
        nullable=False
    )
    
    limite_templates: Mapped[int] = mapped_column(
        Integer,
        server_default='50',
        nullable=False
    )
    
    limite_stockage_mb: Mapped[int] = mapped_column(
        Integer,
        server_default='5000',
        nullable=False
    )
    
    # Subscription
    type_abonnement: Mapped[TypeAbonnement] = mapped_column(
        Enum(TypeAbonnement, values_callable=lambda x: [e.value for e in x]),
        server_default=TypeAbonnement.GRATUIT.value,
        nullable=False,
        index=True
    )
    
    date_debut_abonnement: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    date_fin_abonnement: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True
    )
    
    # Configuration
    configuration: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        server_default=text("'{}'::jsonb"),
        comment="Flexible configuration: theme, features, notifications"
    )
    
    domaines_autorises: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        server_default=text("'[]'::jsonb"),
        comment="Allowed email domains for signup"
    )
    
    # API Security
    api_key_hash: Mapped[Optional[str]] = mapped_column(
        String(128),
        unique=True,
        nullable=True,
        index=True,
        comment="SHA-512 hash of API key"
    )
    
    api_key_prefix: Mapped[Optional[str]] = mapped_column(
        String(16),
        nullable=True,
        comment="First 8 chars for identification"
    )
    
    ip_autorisees: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        server_default=text("'[]'::jsonb"),
        comment="IP whitelist (empty = all allowed)"
    )
    
    mfa_obligatoire: Mapped[bool] = mapped_column(
        Boolean,
        server_default='FALSE',
        nullable=False
    )
    
    # Usage Statistics
    statistiques_usage: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        server_default=text("'{}'::jsonb"),
        comment="Usage metrics and statistics"
    )
    
    # Metadata
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        server_default=text("'{}'::jsonb")
    )
    
    # Billing
    date_dernier_paiement: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    nombre_connexions_mois: Mapped[int] = mapped_column(
        Integer,
        server_default='0',
        nullable=False
    )
    
    # Core Relationships
    utilisateurs: Mapped[List["Utilisateur"]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="select",
        doc="All users belonging to this tenant"
    )
    
    artisan_profiles: Mapped[List["ArtisanProfile"]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan", 
        lazy="select",
        doc="Artisan profiles in this tenant"
    )
    
    templates_devis: Mapped[List["TemplateDevis"]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="select",
        doc="Quote templates for this tenant"
    )
    
    devis: Mapped[List["Devis"]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="select",
        doc="Quotes created by this tenant"
    )
    
    # Table Configuration
    __table_args__ = (
        CheckConstraint(
            "limite_utilisateurs > 0",
            name='check_positive_user_limit'
        ),
        CheckConstraint(
            "limite_artisans > 0",
            name='check_positive_artisan_limit'
        ),
        CheckConstraint(
            "limite_stockage_mb > 0",
            name='check_positive_storage_limit'
        ),
        CheckConstraint(
            "pays ~ '^[A-Z]{2}$'",
            name='check_valid_country_code'
        ),
        Index('idx_tenant_active_subscription', 'est_actif', 'type_abonnement'),
        Index('idx_tenant_code_active', 'code', 'est_actif'),
        Index('idx_tenant_expiry', 'date_fin_abonnement'),
    )
    
    # Validators
    @validates('code')
    def validate_code(self, key: str, value: str) -> str:
        """Validate and normalize tenant code."""
        if not value:
            raise ValueError("Tenant code is required")
        
        value = value.strip().upper()
        if not re.match(r'^[A-Z0-9_-]+$', value):
            raise ValueError("Code must contain only letters, numbers, underscore and hyphen")
        
        if len(value) < 3 or len(value) > 50:
            raise ValueError("Code must be between 3 and 50 characters")
        
        return value
    
    @validates('numero_siret')
    def validate_siret(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate French SIRET number format."""
        if not value:
            return value
        
        value = value.replace(' ', '')
        if not value.isdigit() or len(value) != 14:
            raise ValueError("SIRET must be exactly 14 digits")
        
        return value
    
    @validates('email_contact')
    def validate_email(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate email format."""
        if not value:
            return value
        
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, value.lower()):
            raise ValueError("Invalid email format")
        
        return value.lower()
    
    # Computed Properties
    @hybrid_property
    def is_subscription_active(self) -> bool:
        """Check if subscription is currently active."""
        if not self.est_actif:
            return False
        
        if self.type_abonnement == TypeAbonnement.GRATUIT:
            return True
        
        if self.date_fin_abonnement:
            return datetime.utcnow() < self.date_fin_abonnement
        
        return False
    
    @hybrid_property
    def days_until_expiry(self) -> Optional[int]:
        """Calculate days remaining in subscription."""
        if self.type_abonnement == TypeAbonnement.GRATUIT:
            return None
        
        if self.date_fin_abonnement:
            delta = self.date_fin_abonnement - datetime.utcnow()
            return max(0, delta.days)
        
        return 0
    
    @hybrid_property
    def usage_percentage(self) -> Dict[str, float]:
        """Calculate resource usage percentages."""
        stats = self.statistiques_usage or {}
        
        return {
            'users': (len(self.utilisateurs) / self.limite_utilisateurs * 100) 
                    if self.limite_utilisateurs > 0 else 0,
            'storage': (stats.get('stockage_utilise_mb', 0) / self.limite_stockage_mb * 100) 
                      if self.limite_stockage_mb > 0 else 0,
            'projects': (stats.get('projets_mois_courant', 0) / self.limite_projets_mensuels * 100) 
                       if self.limite_projets_mensuels > 0 else 0
        }
    
    # Business Logic Methods
    def can_add_user(self) -> bool:
        """Check if tenant can add more users."""
        if not self.est_actif:
            return False
        
        return len(self.utilisateurs) < self.limite_utilisateurs
    
    def can_add_artisan(self) -> bool:
        """Check if tenant can add more artisans."""
        if not self.est_actif:
            return False
        
        artisan_count = sum(
            1 for u in self.utilisateurs 
            if u.type_profil == 'artisan'
        )
        return artisan_count < self.limite_artisans
    
    def can_create_project(self) -> bool:
        """Check if tenant can create more projects this month."""
        if not self.est_actif:
            return False
        
        stats = self.statistiques_usage or {}
        return stats.get('projets_mois_courant', 0) < self.limite_projets_mensuels
    
    def increment_project_counter(self) -> None:
        """Increment monthly project counter."""
        if not self.statistiques_usage:
            self.statistiques_usage = {}
        
        current = self.statistiques_usage.get('projets_mois_courant', 0)
        self.statistiques_usage['projets_mois_courant'] = current + 1
        self.statistiques_usage['derniere_activite'] = datetime.utcnow().isoformat()
    
    def reset_monthly_counters(self) -> None:
        """Reset monthly usage counters (called by scheduled job)."""
        if not self.statistiques_usage:
            self.statistiques_usage = {}
        
        self.statistiques_usage['projets_mois_courant'] = 0
        self.nombre_connexions_mois = 0
    
    def verify_ip_allowed(self, ip: str) -> bool:
        """Check if IP address is in whitelist."""
        if not self.ip_autorisees:
            return True  # No whitelist means all IPs allowed
        
        return ip in self.ip_autorisees
    
    def verify_email_domain(self, email: str) -> bool:
        """Check if email domain is allowed for signup."""
        if not self.domaines_autorises:
            return True  # No restriction means all domains allowed
        
        domain = email.split('@')[1] if '@' in email else ''
        return domain in self.domaines_autorises
    
    # API Key Management
    def generate_api_key(self) -> str:
        """Generate a new API key for the tenant."""
        api_key = secrets.token_urlsafe(48)
        self.api_key_hash = hashlib.sha512(api_key.encode()).hexdigest()
        self.api_key_prefix = api_key[:8]
        return api_key
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify provided API key against stored hash."""
        if not self.api_key_hash or not api_key:
            return False
        
        provided_hash = hashlib.sha512(api_key.encode()).hexdigest()
        return secrets.compare_digest(provided_hash, self.api_key_hash)
    
    def revoke_api_key(self) -> None:
        """Revoke current API key."""
        self.api_key_hash = None
        self.api_key_prefix = None
    
    # Feature Access
    def has_feature(self, feature_name: str) -> bool:
        """Check if tenant has access to a specific feature."""
        if not self.configuration:
            return False
        
        features = self.configuration.get('features', {})
        return features.get(feature_name, False)
    
    def get_theme_config(self) -> dict:
        """Get theme configuration for white-labeling."""
        if not self.configuration:
            return {}
        
        return self.configuration.get('theme', {
            'primary_color': '#007bff',
            'logo_url': None,
            'company_name': self.nom
        })
    
    # Serialization
    def to_dict_public(self) -> dict:
        """Convert to dictionary for public API responses."""
        return {
            'id': str(self.id),
            'name': self.nom,
            'subscription_type': self.type_abonnement.value,
            'is_active': self.est_actif,
            'api_key_prefix': self.api_key_prefix,
            'theme': self.get_theme_config(),
            'features': self.configuration.get('features', {}) if self.configuration else {},
            'usage': self.usage_percentage
        }
    
    def to_dict_admin(self) -> dict:
        """Convert to dictionary for admin API responses."""
        return {
            **self.to_dict_public(),
            'code': self.code,
            'limits': {
                'users': self.limite_utilisateurs,
                'artisans': self.limite_artisans,
                'projects_monthly': self.limite_projets_mensuels,
                'storage_mb': self.limite_stockage_mb
            },
            'subscription': {
                'type': self.type_abonnement.value,
                'start_date': self.date_debut_abonnement.isoformat() if self.date_debut_abonnement else None,
                'end_date': self.date_fin_abonnement.isoformat() if self.date_fin_abonnement else None,
                'days_remaining': self.days_until_expiry
            },
            'statistics': self.statistiques_usage or {},
            'contact': {
                'email': self.email_contact,
                'phone': self.telephone_contact
            }
        }
    
    def __repr__(self) -> str:
        return f"<Tenant {self.code}: {self.nom} ({self.type_abonnement.value})>"


# Event Hooks for Automatic Behavior
@event.listens_for(Tenant, 'before_insert')
def tenant_before_insert(mapper, connection, target):
    """Initialize tenant with default values before insertion."""
    # Initialize usage statistics
    if not target.statistiques_usage:
        target.statistiques_usage = {
            'projets_mois_courant': 0,
            'stockage_utilise_mb': 0,
            'derniere_activite': datetime.utcnow().isoformat(),
            'utilisateurs_actifs_30j': 0
        }
    
    # Set subscription dates for paid plans
    if target.type_abonnement != TypeAbonnement.GRATUIT:
        if not target.date_debut_abonnement:
            target.date_debut_abonnement = datetime.utcnow()
        
        if not target.date_fin_abonnement:
            # Default to 30-day trial for new paid subscriptions
            target.date_fin_abonnement = datetime.utcnow() + timedelta(days=30)
    
    # Initialize default configuration
    if not target.configuration:
        target.configuration = {
            'theme': {
                'primary_color': '#007bff',
                'secondary_color': '#6c757d'
            },
            'features': {
                'api_enabled': False,
                'sso_enabled': False,
                'custom_domain': False
            },
            'notifications': {
                'email_enabled': True,
                'sms_enabled': False,
                'webhook_enabled': False
            }
        }


@event.listens_for(Tenant, 'before_update')
def tenant_before_update(mapper, connection, target):
    """Handle tenant status changes before update."""
    # Record deactivation timestamp
    if not target.est_actif and not target.date_desactivation:
        target.date_desactivation = datetime.utcnow()
    
    # Clear deactivation info when reactivating
    if target.est_actif and target.date_desactivation:
        target.date_desactivation = None
        target.raison_desactivation = None
    
    # Update last activity timestamp
    if target.statistiques_usage:
        target.statistiques_usage['derniere_activite'] = datetime.utcnow().isoformat()
