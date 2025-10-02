"""
Unit Tests for Emergency Escalation Service

Demonstrates professional testing practices:
- AAA pattern (Arrange-Act-Assert) for test clarity
- Comprehensive mocking for complete isolation
- Parametrized tests to reduce redundancy
- Fixtures as dependency injection
- Edge case and error handling coverage

Test Coverage: Business logic for emergency escalation system including
timing rules, geographic expansion, notification management, and admin alerts.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
from typing import List, Tuple

from sqlalchemy.exc import SQLAlchemyError

from app.services.urgence_escalation_service import (
    UrgenceEscalationService, 
    EscalationRule
)
from app.models.enums import (
    NiveauUrgence, 
    StatusUrgence, 
    TypeUrgence
)
from app.exceptions.urgence_exceptions import (
    UrgenceEscalationError
)


class TestEscalationRule:
    """Tests for EscalationRule configuration class."""
    
    def test_rule_initialization(self):
        """Verify escalation rule initializes with correct parameters."""
        # Arrange
        niveau = NiveauUrgence.CRITIQUE
        delays = [30, 60, 120]
        radius_increments = [10, 20, 30]
        additional_artisans = [10, 15, 20]
        notify_admin = 3
        
        # Act
        rule = EscalationRule(
            niveau=niveau,
            delays_minutes=delays,
            radius_increments_km=radius_increments,
            additional_artisans=additional_artisans,
            notify_admin_at_level=notify_admin
        )
        
        # Assert
        assert rule.niveau == niveau
        assert rule.delays_minutes == delays
        assert rule.radius_increments_km == radius_increments
        assert rule.additional_artisans == additional_artisans
        assert rule.notify_admin_at_level == notify_admin


class TestServiceInitialization:
    """Tests for service initialization and configuration."""
    
    def test_default_initialization(self):
        """Verify service initializes with default dependencies."""
        # Act
        service = UrgenceEscalationService()
        
        # Assert
        assert service.crud_urgence is not None
        assert service.crud_artisan_for_urgence is not None
        assert service.crud_notification is not None
        assert service.BASE_RADIUS_KM == 25
    
    def test_custom_crud_injection(self, mock_crud_urgence, mock_crud_artisan):
        """Verify dependency injection works correctly."""
        # Act
        service = UrgenceEscalationService(
            crud_urgence=mock_crud_urgence,
            crud_artisan=mock_crud_artisan
        )
        
        # Assert
        assert service.crud_urgence == mock_crud_urgence
        assert service.crud_artisan_for_urgence.artisan_crud == mock_crud_artisan
    
    def test_escalation_rules_configuration(self):
        """Verify escalation rules are configured correctly by urgency level."""
        # Arrange
        service = UrgenceEscalationService()
        
        # Act
        critical_rule = service.get_escalation_rule(NiveauUrgence.CRITIQUE)
        medium_rule = service.get_escalation_rule(NiveauUrgence.MOYEN)
        
        # Assert
        assert critical_rule.delays_minutes == [30, 60, 120]
        assert medium_rule.delays_minutes == [360, 720]


class TestBulkEscalation:
    """Tests for bulk urgency escalation operations."""
    
    @pytest.mark.asyncio
    async def test_no_urgencies_to_escalate(
        self, db_session, mock_crud_urgence
    ):
        """Verify handling when no urgencies require escalation."""
        # Arrange
        mock_crud_urgence.get_urgences_for_escalation = AsyncMock(return_value=[])
        service = UrgenceEscalationService(crud_urgence=mock_crud_urgence)
        
        # Act
        stats = await service.check_and_escalate_all_urgences(db_session)
        
        # Assert
        assert stats['checked'] == 0
        assert stats['escalated'] == 0
        assert stats['notifications_sent'] == 0
        assert stats['errors'] == 0
    
    @pytest.mark.asyncio
    async def test_multiple_urgencies_escalation(
        self, db_session, mock_crud_urgence, 
        mock_urgence_projet, mock_urgence_critique
    ):
        """Verify bulk escalation of multiple urgencies."""
        # Arrange
        urgences = [mock_urgence_projet, mock_urgence_critique]
        mock_crud_urgence.get_urgences_for_escalation = AsyncMock(
            return_value=urgences
        )
        mock_crud_urgence.update_escalation_status = AsyncMock(return_value=True)
        
        service = UrgenceEscalationService(crud_urgence=mock_crud_urgence)
        
        with patch.object(service, '_find_new_artisans', 
                         new_callable=AsyncMock) as mock_find:
            mock_find.return_value = [(MagicMock(), 5.0, 75.0)] * 3
            
            with patch.object(service, '_create_escalation_notifications', 
                            new_callable=AsyncMock) as mock_create:
                mock_create.return_value = 3
                
                # Act
                stats = await service.check_and_escalate_all_urgences(db_session)
        
        # Assert
        assert stats['checked'] == 2
        assert stats['escalated'] == 2
        assert stats['notifications_sent'] == 6
        assert stats['errors'] == 0
    
    @pytest.mark.asyncio
    async def test_database_error_handling(
        self, db_session, mock_crud_urgence
    ):
        """Verify proper error handling for database failures."""
        # Arrange
        mock_crud_urgence.get_urgences_for_escalation = AsyncMock(
            side_effect=SQLAlchemyError("Connection error")
        )
        service = UrgenceEscalationService(crud_urgence=mock_crud_urgence)
        
        # Act & Assert
        with pytest.raises(UrgenceEscalationError) as exc_info:
            await service.check_and_escalate_all_urgences(db_session)
        
        assert "Erreur base de données" in str(exc_info.value)


class TestIndividualEscalation:
    """Tests for individual urgency escalation logic."""
    
    @pytest.mark.asyncio
    async def test_escalation_not_yet_due(
        self, db_session, mock_crud_urgence, mock_urgence_projet
    ):
        """Verify no escalation when time threshold not met."""
        # Arrange
        mock_urgence_projet.escalation_level = 0
        mock_urgence_projet.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        
        service = UrgenceEscalationService(crud_urgence=mock_crud_urgence)
        
        # Act
        result = await service.escalate_urgence(db_session, mock_urgence_projet)
        
        # Assert
        assert result['escalated'] is False
        assert result['new_level'] == 0
        assert result['next_escalation'] is not None
        mock_crud_urgence.update_escalation_status.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_max_escalation_level_reached(
        self, db_session, mock_crud_urgence, mock_urgence_critique
    ):
        """Verify no escalation when max level reached."""
        # Arrange
        mock_urgence_critique.escalation_level = 3  # Max for critical
        mock_urgence_critique.last_escalation_at = datetime.now(timezone.utc) - timedelta(hours=3)
        
        service = UrgenceEscalationService(crud_urgence=mock_crud_urgence)
        
        # Act
        result = await service.escalate_urgence(db_session, mock_urgence_critique)
        
        # Assert
        assert result['escalated'] is False
        assert result['new_level'] == 3
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "niveau,level,minutes_elapsed,should_escalate",
        [
            (NiveauUrgence.CRITIQUE, 0, 35, True),   # Past threshold
            (NiveauUrgence.CRITIQUE, 0, 25, False),  # Before threshold
            (NiveauUrgence.ELEVE, 0, 130, True),     # High urgency escalation
            (NiveauUrgence.MOYEN, 0, 370, True),     # Medium urgency escalation
        ]
    )
    async def test_escalation_timing_rules(
        self, db_session, mock_crud_urgence, mock_urgence_projet,
        niveau, level, minutes_elapsed, should_escalate
    ):
        """Test escalation timing rules for different urgency levels."""
        # Arrange
        mock_urgence_projet.niveau_urgence = niveau
        mock_urgence_projet.escalation_level = level
        mock_urgence_projet.created_at = (
            datetime.now(timezone.utc) - timedelta(minutes=minutes_elapsed)
        )
        
        mock_crud_urgence.update_escalation_status = AsyncMock(return_value=True)
        service = UrgenceEscalationService(crud_urgence=mock_crud_urgence)
        
        with patch.object(service, '_find_new_artisans', 
                         return_value=[(MagicMock(), 5.0, 75.0)] * 5 if should_escalate else []):
            with patch.object(service, '_create_escalation_notifications', 
                            return_value=5 if should_escalate else 0):
                # Act
                result = await service.escalate_urgence(db_session, mock_urgence_projet)
        
        # Assert
        assert result['escalated'] == should_escalate
        if should_escalate:
            assert result['new_level'] == level + 1
            mock_crud_urgence.update_escalation_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_admin_notification_trigger(
        self, db_session, mock_crud_urgence, mock_urgence_critique
    ):
        """Verify admin notification at configured escalation level."""
        # Arrange
        mock_urgence_critique.escalation_level = 2  # Next level triggers admin
        mock_urgence_critique.last_escalation_at = (
            datetime.now(timezone.utc) - timedelta(hours=3)
        )
        
        mock_crud_urgence.update_escalation_status = AsyncMock(return_value=True)
        service = UrgenceEscalationService(crud_urgence=mock_crud_urgence)
        
        with patch.object(service, '_find_new_artisans', 
                         return_value=[(MagicMock(), 5.0, 75.0)] * 3):
            with patch.object(service, '_create_escalation_notifications', 
                            return_value=3):
                with patch.object(service, '_schedule_admin_notification', 
                                new_callable=AsyncMock) as mock_admin:
                    # Act
                    result = await service.escalate_urgence(
                        db_session, mock_urgence_critique
                    )
        
        # Assert
        assert result['escalated'] is True
        assert result['new_level'] == 3
        mock_admin.assert_called_once()


class TestArtisanDiscovery:
    """Tests for finding new artisans during escalation."""
    
    @pytest.mark.asyncio
    async def test_no_artisans_found(
        self, db_session, mock_urgence_projet, mock_crud_artisan_for_urgence
    ):
        """Verify handling when no artisans are available."""
        # Arrange
        mock_crud_artisan_for_urgence.find_artisans_in_radius = AsyncMock(
            return_value=[]
        )
        
        service = UrgenceEscalationService()
        service.crud_artisan_for_urgence = mock_crud_artisan_for_urgence
        
        # Act
        artisans = await service._find_new_artisans(
            db_session, mock_urgence_projet, radius_km=30, limit=10
        )
        
        # Assert
        assert artisans == []
    
    @pytest.mark.asyncio
    async def test_exclude_previously_notified_artisans(
        self, db_session, mock_urgence_projet, mock_crud_artisan_for_urgence,
        mock_urgence_notification, create_mock_artisans_with_distance
    ):
        """Verify previously notified artisans are excluded."""
        # Arrange
        mock_urgence_projet.notifications = [mock_urgence_notification]
        artisans = create_mock_artisans_with_distance(5)
        
        mock_crud_artisan_for_urgence.find_artisans_in_radius = AsyncMock(
            return_value=artisans
        )
        
        service = UrgenceEscalationService()
        service.crud_artisan_for_urgence = mock_crud_artisan_for_urgence
        
        # Act
        result = await service._find_new_artisans(
            db_session, mock_urgence_projet, radius_km=50, limit=3
        )
        
        # Assert
        assert len(result) == 3
        call_args = mock_crud_artisan_for_urgence.find_artisans_in_radius.call_args
        assert mock_urgence_notification.artisan_id in call_args.kwargs['excluded_artisan_ids']


class TestMatchScoring:
    """Tests for artisan-urgency match scoring algorithm."""
    
    @pytest.mark.parametrize(
        "distance,trust_score,niveau,is_premium,expected",
        [
            (3.0, 100.0, NiveauUrgence.CRITIQUE, True, 100.0),   # Perfect score
            (3.0, 100.0, NiveauUrgence.CRITIQUE, False, 80.0),   # No premium bonus
            (7.0, 80.0, NiveauUrgence.ELEVE, False, 54.0),       # Medium distance
            (15.0, 60.0, NiveauUrgence.MOYEN, False, 38.0),      # High distance
            (25.0, 40.0, NiveauUrgence.FAIBLE, False, 22.0),     # Very high distance
        ]
    )
    def test_match_score_calculation(
        self, mock_urgence_projet, mock_artisan_profile,
        distance, trust_score, niveau, is_premium, expected
    ):
        """Test match score calculation with various parameters."""
        # Arrange
        mock_urgence_projet.niveau_urgence = niveau
        mock_artisan_profile.trust_score = trust_score
        
        service = UrgenceEscalationService()
        
        with patch.object(service, '_is_premium_artisan', return_value=is_premium):
            # Act
            score = service._calculate_match_score(
                mock_urgence_projet, mock_artisan_profile, distance
            )
        
        # Assert
        assert score == expected


class TestNotificationManagement:
    """Tests for notification creation and management."""
    
    @pytest.mark.asyncio
    async def test_batch_notification_creation(
        self, db_session, mock_urgence_projet, mock_crud_notification,
        create_mock_artisans_with_distance
    ):
        """Verify batch notification creation for escalation."""
        # Arrange
        artisans = create_mock_artisans_with_distance(3)
        artisans_with_score = [(a[0], a[1], 75.0) for a in artisans]
        
        mock_crud_notification.create_batch_notifications = AsyncMock(
            return_value=[MagicMock() for _ in range(3)]
        )
        
        service = UrgenceEscalationService()
        service.crud_notification = mock_crud_notification
        
        # Act
        count = await service._create_escalation_notifications(
            db_session, mock_urgence_projet, artisans_with_score, escalation_level=2
        )
        
        # Assert
        assert count == 3
        call_args = mock_crud_notification.create_batch_notifications.call_args
        assert call_args.kwargs['urgence_id'] == mock_urgence_projet.id
        assert call_args.kwargs['is_escalation'] is True
        assert call_args.kwargs['escalation_level'] == 2


class TestUtilityMethods:
    """Tests for utility and helper methods."""
    
    @pytest.mark.parametrize(
        "type_urgence,expected",
        [
            ("fuite_eau", ["plombier"]),
            ("panne_electrique", ["electricien"]),
            ("porte_claquee", ["serrurier"]),
            ("volet_bloque", ["menuisier", "serrurier"]),
            ("unknown", []),
        ]
    )
    def test_activity_type_mapping(self, type_urgence, expected):
        """Test mapping of urgency types to activity types."""
        # Arrange
        service = UrgenceEscalationService()
        
        # Act
        activities = service._get_activity_types_for_urgence(type_urgence)
        
        # Assert
        assert activities == expected
    
    def test_next_escalation_time_calculation(self):
        """Test calculation of next escalation time."""
        # Arrange
        service = UrgenceEscalationService()
        reference_time = datetime.now(timezone.utc)
        
        # Act
        next_time = service.calculate_next_escalation_time(
            NiveauUrgence.CRITIQUE, current_level=0, reference_time=reference_time
        )
        
        # Assert
        assert next_time == reference_time + timedelta(minutes=30)
    
    def test_premium_artisan_detection(self, mock_artisan_profile):
        """Test premium status detection based on trust score."""
        # Arrange
        service = UrgenceEscalationService()
        
        # Act & Assert - High trust score
        mock_artisan_profile.trust_score = 85.0
        assert service._is_premium_artisan(mock_artisan_profile) is True
        
        # Act & Assert - Low trust score
        mock_artisan_profile.trust_score = 50.0
        assert service._is_premium_artisan(mock_artisan_profile) is False
