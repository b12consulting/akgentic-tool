"""The message base a mailbox-deliverable class extends.

``MailboxMessage`` adds the two renderings the mailbox needs to an ordinary
``akgentic.core`` message: the prose a message reads as once it has been taken
on, and the single line it is advertised with while another run is in flight.

**Why a base class here, and not a Protocol or a core method.** The two
renderings have two consumers in two packages —
:class:`~akgentic.tool.mailbox.capability.MailboxCapability` for the mid-run
absorb, and ``BaseAgent.act()`` for the ordinary turn — and neither
``akgentic-tool`` nor ``akgentic-agent`` may import the other. Declared as
Protocols, that forced the *same* one-method contract to be written twice, once
per package, kept in step by nothing but care. Declared on
``akgentic.core.messages.Message``, it put prose rendering into the actor
framework, which knows nothing about readers and should not learn.

This package is the one both consumers already depend on, so the contract sits
here and is inherited rather than duplicated or pushed down. ``akgentic-tool``
depends on ``akgentic-core``, so extending ``Message`` costs nothing.

Nominal, not structural, and that is the deliberate part: a class is
mailbox-deliverable because it *says so* by extending this, not because it
happens to have two methods with the right names. The capability can then ask
``isinstance`` and get a real answer.
"""

from akgentic.core.messages import Message


class MailboxMessage(Message):
    """A message that can render itself for a reader, and be previewed in a list.

    **Both are promises, and both raise if a subclass does not answer them.**
    Extending this class is the declaration that a message can travel through a
    mailbox, and a message that can be delivered can be listed: there is no
    coherent class that renders but cannot summarise itself.

    Opting *out* of mid-run reads is therefore not done by declining a method —
    it is done by not extending this class at all, and by the offer filter's
    exact-class match. A pending message of a different class than the one being
    handled is listed without an id whatever it declares, so a run never absorbs
    mail it could not route.
    """

    def rendering(self) -> str:
        """This message as prose. **Subclasses must override.**

        **Rendering a message is the message's job.** The object that knows how a
        message reads is the message — not the handler that happens to be
        running, and not whatever consumes it. This is a *method* and never a
        ``content`` field: keying it on a field re-encodes the assumption that
        produced the defect this design removes, where one consumer rendered
        every message as ``getattr(message, "content", "")`` and silently
        emptied every class that declares its own fields instead.

        The rendering is self-contained and carries whatever framing the message
        needs to stand on its own — who sent it, what kind of message it is,
        what is expected in return. A consumer concatenates it and never
        composes framing of its own, because only the class knows what its
        fields mean.

        **It raises rather than returning ``None``, and that is the safety
        property.** A consumer that takes a message on has already removed it
        from the mailbox; if rendering could quietly yield nothing, that message
        would be gone from the queue and never delivered — lost outright, with no
        log line a user would ever see. Raising turns a class that forgot to
        override into an immediate, obvious failure in development instead of
        silent data loss in production.

        Returns:
            The prose form. Never ``None``.

        Raises:
            NotImplementedError: Always, unless the subclass overrides. Extending
                ``MailboxMessage`` is a declaration that this method is answered.
        """
        raise NotImplementedError(
            f"{type(self).__name__} extends MailboxMessage but does not override rendering(); "
            f"a deliverable message must say how it reads."
        )

    def rendering_preview(self) -> str:
        """This message as a single summary line. **Subclasses must override.**

        The second half of the same bargain as :meth:`rendering`, and a distinct
        question: how a message reads once it is being dealt with is not how it
        is *listed* while something else is in progress. One is a prompt, the
        other is one line in a notice — but a class that can answer the first can
        always answer the second, which is why both are required rather than one
        being optional.

        It **raises rather than returning ``None``** for the same reason
        :meth:`rendering` does. A ``None`` here would be a second way of saying
        "not offerable", competing with the one that already decides it — the
        offer filter's exact-class match — and two mechanisms for one rule is how
        a message ends up listed with an id it cannot honour.

        Returns:
            The one-line summary. Never ``None``.

        Raises:
            NotImplementedError: Always, unless the subclass overrides.
        """
        raise NotImplementedError(
            f"{type(self).__name__} extends MailboxMessage but does not override "
            f"rendering_preview(); a message that can be delivered can be listed."
        )
