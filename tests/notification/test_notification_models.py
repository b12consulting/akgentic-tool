"""The delivery-class resolver and the notification models (AC1, AC10).

The resolver is the whole of the module-boundary trick: ``akgentic-tool`` never
imports the package that owns the delivery class, so every way that string can be
wrong has to fail loudly at wiring time rather than 300 seconds later.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.messages.message import Message, UserMessage
from akgentic.tool.notification.models import (
    DEFAULT_MESSAGE_CLASS,
    NotificationState,
    PendingNotification,
    resolve_message_class,
)
from pydantic import ValidationError

from tests.notification.conftest import FAKE_MESSAGE_PATH, FakeNotificationMessage


class TestResolveMessageClass:
    def test_resolves_a_class_carrying_content_and_type(self) -> None:
        assert resolve_message_class(FAKE_MESSAGE_PATH) is FakeNotificationMessage

    def test_the_shipped_default_resolves(self) -> None:
        """The default names ``akgentic-agent``, which this package does not depend on.

        Skipped rather than dropped: standalone CI installs this package alone, so
        the default cannot resolve there — but where the framework is installed
        together, the default must be genuinely usable, and only this asserts it.
        """
        pytest.importorskip(
            DEFAULT_MESSAGE_CLASS.rsplit(".", 1)[0],
            reason="akgentic-agent is not installed alongside this package",
        )
        resolved = resolve_message_class(DEFAULT_MESSAGE_CLASS)
        assert issubclass(resolved, Message)
        assert {"content", "type"} <= set(resolved.model_fields)

    def test_a_non_importable_module_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="not importable"):
            resolve_message_class("akgentic.tool.notification.no_such_module.Thing")

    def test_a_missing_attribute_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="no attribute 'NoSuchClass'"):
            resolve_message_class("tests.notification.conftest.NoSuchClass")

    def test_a_bare_name_without_a_module_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="not a dotted path"):
            resolve_message_class("FakeNotificationMessage")

    def test_a_non_message_class_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="not a Message subclass"):
            resolve_message_class("tests.notification.conftest.NotAMessage")

    def test_a_message_without_content_or_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"field\(s\): content, type"):
            resolve_message_class("akgentic.core.messages.message.Message")

    def test_a_message_missing_only_type_names_only_that_field(self) -> None:
        """``UserMessage`` carries ``content`` but no ``type`` — say exactly that."""
        assert issubclass(UserMessage, Message)
        with pytest.raises(ValueError, match=r"field\(s\): type\."):
            resolve_message_class("akgentic.core.messages.message.UserMessage")

    def test_a_type_that_cannot_hold_notification_raises_value_error(self) -> None:
        """Declaring ``type`` is not accepting ``"notification"`` into it.

        Without this check the class passes wiring and fails inside ``_deliver``,
        where the broad except logs one line per fire and drops the entry — a
        configuration defect surfacing as silently missing notifications.
        """
        with pytest.raises(ValueError, match="cannot carry type='notification'"):
            resolve_message_class("tests.notification.conftest.NarrowTypeMessage")

    def test_a_class_that_rejects_empty_content_still_resolves(self) -> None:
        """The probe carries a delivery-shaped content, not a minimal one.

        ``_deliver`` writes the scheduled text, never ``""``. A class whose
        ``content`` is constrained away from the empty string delivers fine, so
        refusing it at ``observer()`` bind time would break a working card over a
        payload the tool never builds.
        """
        resolved = resolve_message_class("tests.notification.conftest.NonEmptyContentMessage")

        assert resolved.__name__ == "NonEmptyContentMessage"


class TestModels:
    def test_pending_notification_round_trips(self, owner_address: ActorAddress) -> None:
        now = datetime.now(UTC)
        entry = PendingNotification(
            notification_id=7,
            owner=owner_address,
            content="check the build",
            created_at=now,
            fire_at=now + timedelta(seconds=30),
        )
        restored = PendingNotification.model_validate(entry.model_dump())
        assert restored.notification_id == 7
        assert restored.content == "check the build"
        assert restored.fire_at == entry.fire_at
        assert restored.owner.agent_id == owner_address.agent_id

    def test_pending_notification_rejects_a_naive_datetime(
        self, owner_address: ActorAddress
    ) -> None:
        """Due times are compared against ``now(UTC)``; a naive one raises there."""
        with pytest.raises(ValidationError, match="timezone"):
            PendingNotification(
                notification_id=1,
                owner=owner_address,
                content="x",
                created_at=datetime.now(),
                fire_at=datetime.now(),
            )

    def test_state_defaults_are_empty_and_start_at_one(self) -> None:
        state = NotificationState()
        assert state.pending == {}
        assert state.next_id == 1

    def test_state_round_trips_with_its_pending_entries(self, owner_address: ActorAddress) -> None:
        """Resume depends on this: entries and the counter both come back."""
        now = datetime.now(UTC)
        state = NotificationState(
            next_id=4,
            pending={
                3: PendingNotification(
                    notification_id=3,
                    owner=owner_address,
                    content="ping",
                    created_at=now,
                    fire_at=now + timedelta(seconds=5),
                )
            },
        )
        restored = NotificationState.model_validate(state.model_dump())
        assert restored.next_id == 4
        assert list(restored.pending) == [3]
        assert restored.pending[3].content == "ping"
        assert isinstance(restored.pending[3].owner.agent_id, uuid.UUID)
