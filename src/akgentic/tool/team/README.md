# TeamTool

Team self-management: who is on the team, what roles could be hired, hiring and firing members,
and — from the team's own telemetry — who is busy right now and on what.

```python
from akgentic.tool.team import TeamTool
```

| | |
|---|---|
| Module | `akgentic.tool.team.team` |
| Actor | `TeamActivityActor` (`#TeamActivity`) — **created only when a summarizer is configured** |
| Channels used | `SYSTEM_PROMPT`, `TOOL_CALL`, `COMMAND` |
| Observer required | `TeamManagementToolObserver` (provided by `BaseAgent` in `akgentic-agent`) |
| Optional extras | none |

---

## The ToolCard

```python
class TeamTool(ToolCard):
    hire_team_members: HireTeamMember | bool = True
    fire_team_members: FireTeamMember | bool = True
    get_role_profiles: GetRoleProfiles | bool = True
    get_team_roster: GetTeamRoster | bool = True
    get_team_activity: GetTeamActivity | bool = True

    _activity_proxy: TeamActivityActor | None = PrivateAttr(default=None)
```

**This card needs more than the base observer.** `observer()` keeps the base `ToolObserver`
parameter type — `ToolFactory` attaches one observer to every card uniformly, so narrowing the
signature would break substitutability — and applies the narrower `TeamManagementToolObserver`
type internally. That protocol adds `createActor`, `on_hire` and `on_fire` on top of the actor
observer, which is what lets a tool call bring a new agent into the team. Conformance is a
documented precondition, not a runtime check: a non-conforming observer fails at first use.

**Hiring goes through the observer, never through Pykka.** `createActor` handles context
propagation (user id, team id, orchestrator, parent, children tracking) and notifies the
orchestrator. Calling `agent_class.start()` directly would produce an actor the team does not
know about.

---

## ToolCard fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `hire_team_members` | `HireTeamMember \| bool` | `True` | `hire_members(roles)` on `TOOL_CALL`, `hire_member(role, name=None)` on `COMMAND`. |
| `fire_team_members` | `FireTeamMember \| bool` | `True` | `fire_members(names)` on `TOOL_CALL`, `fire_member(name)` on `COMMAND`. |
| `get_team_roster` | `GetTeamRoster \| bool` | `True` | The current roster as a system prompt. |
| `get_role_profiles` | `GetRoleProfiles \| bool` | `True` | The hireable role catalog as a system prompt. |
| `get_team_activity` | `GetTeamActivity \| bool` | `True` | `team_activity()` — who is mid-handler. Free by default. |

---

## Capability parameters

### `HireTeamMember` / `FireTeamMember`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL, COMMAND}` | The two channels carry **different callables**, not the same one twice. |

| Channel | Hire | Fire |
|---|---|---|
| `TOOL_CALL` | `hire_members(roles: list[str]) -> str` — batch, auto-named | `fire_members(names: list[str]) -> str` — batch |
| `COMMAND` | `hire_member(role: str, name: str \| None = None) -> ActorAddress` | `fire_member(name: str) -> str` |

The batch variants are what the LLM sees; the singular variants exist for programmatic
orchestration and for the human `/`-command surface, where naming the new member matters.

**Naming.** With no explicit name, a hire is named `@<Role><random 100–999>`, retried upward until
unique. An explicit name is stripped, must be non-empty, and must not collide — each failure is a
`RetriableError` the model can correct. The role must exist in the orchestrator's agent catalog;
otherwise the error lists the available roles.

**Partial success is reported, not hidden.** `hire_members(["Dev", "Nonexistent"])` hires the Dev,
then raises `RetriableError` whose message begins `Partial success - Members hired: [...]` and
goes on to list the failures and the available roles. `fire_members` behaves the same way. The
model therefore learns both what happened and what to fix.

Both tool docstrings carry an explicit note that these should only be used when the user asks —
hiring is the one capability in the package with an unbounded blast radius.

#### Hire and fire are not symmetric

They look like a pair, and they are not. Hire creates through the observer; fire stops the member
directly and lets the member tell the orchestrator.

```python
# hire — the orchestrator proxy answers "which class, and is the name free?"
child_address = observer.createActor(actor_class, config=agent_card_config)
observer.on_hire(child_address)

# fire — the orchestrator proxy only resolves the name
address = orchestrator_proxy.get_team_member(name)
observer.proxy_ask(address, Akgent).stop()
observer.on_fire(address)
```

**The orchestrator is a directory on the fire path, not the executor.** It is asked for
`get_team_member(name)` — and for `get_team()` when that lookup fails, to list the current members
in the error — and nothing else. Nobody asks it to remove the member. The stop is delivered to the
**member itself** through `proxy_ask(address, Akgent)`, and the orchestrator finds out afterwards
because `Akgent.on_stop()` sends it a `StopMessage`. That is also why a member stopped by any
other route leaves the roster correctly: the notification is the actor's, not the tool's.

**`proxy_ask` is the public route to another actor, and it blocks.** Reaching through
`ActorAddressImpl._actor_ref._actor` to call Pykka's `stop()` on the raw actor would bypass the
mailbox and the teardown chain; `proxy_ask(address, Akgent).stop()` delivers `stop` as a message
and waits for it. Firing is therefore synchronous from the caller's point of view.

**Firing one member fires its subtree.** `Akgent.stop()` publishes the agent's state, calls
`stop_children(blocking=True)`, and only then stops itself — so a fired coordinator takes every
agent it hired with it, depth-first, and the call returns once the whole subtree is down. State is
published *before* the child teardown on purpose: a child that hangs or raises means `on_stop()`
is never reached, and the state would otherwise die with the actor.

**The roster catches up asynchronously.** `on_stop` reaches the orchestrator through a *tell*, so
the fire call returns before the orchestrator has necessarily processed the `StopMessage` and
dropped the member. Between those two moments `get_team_member(name)` can still hand back the
address of an actor that is already down.

Two consequences for `fire_members(names)`: members are stopped **one at a time**, each waiting
for its own subtree, so firing a deep team is not instantaneous; and firing a coordinator and one
of the agents it hired in the same call is a genuine race — the descendant is already stopped by
the ancestor's teardown, but the lookup that guards the second stop may not know it yet. Nothing
on the fire path re-checks liveness before building the proxy, and the per-name handling in
`fire_members` catches only `RetriableError`. Fire ancestors last, or fire them alone.

### `GetTeamRoster` — `team_members()`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{SYSTEM_PROMPT, COMMAND}` | Context, not a tool call. |

Renders `name (role: X)` per member, marking the calling agent with `- [you]`. **Tool actors are
excluded** — any member whose name starts with `#`. Returns `""` when the team is empty or the
calling agent has already stopped, so an empty roster adds nothing to the prompt. Any failure is
logged and rendered as `"Cannot get team roster..."` rather than raised.

### `GetRoleProfiles` — `team_roles()`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{SYSTEM_PROMPT, COMMAND}` | |

Renders `role: description (Skills: a, b)` for every `AgentCard` in the orchestrator's catalog —
the menu `hire_members` draws from. Same soft-failure behaviour as the roster.

### `GetTeamActivity` — `team_activity()`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL, COMMAND}` | |
| `summarizer` | `ActivitySummarizer \| None` | `None` | Configuring one enables on-demand summaries **and** brings the `#TeamActivity` actor into existence. `None` ⇒ truncation only, no actor, no model call, ever. |
| `stale_after_seconds` | `float` | `300.0` | Handlers open longer than this are treated as replayed history and dropped. |
| `max_task_chars` | `int` | `200` | Character budget for reported task text — the truncation length, and the summary length when summarizing. |

#### `ActivitySummarizer`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model` | `str` | `"openai:gpt-5.2-mini"` | A **pydantic-ai model spec string**, not a framework `ModelConfig` — `akgentic-tool` must not depend on `akgentic-llm`. |
| `poll_attempts` | `int` | `5` | Attempts spent waiting for an in-flight summary. |
| `poll_delay_seconds` | `float` | `0.4` | Sleep between attempts. |

Because the model spec bypasses `ReactAgent`, **those tokens are counted by neither its cost
accounting nor its usage limits**. Budget for them separately.

---

## What `team_activity` costs

Two independent gates, and conflating them is the mistake the design exists to prevent:

| Configuration | `team_activity` | `#TeamActivity` actor | model call | `summarize_over` in the schema |
|---|---|---|---|---|
| `get_team_activity=False` | not exposed | not created | never | n/a |
| `get_team_activity=True`, `summarizer=None` *(default)* | exposed, truncates | **not created** | never | **absent** |
| `summarizer=ActivitySummarizer(...)` | exposed, summarizes | created | on demand | present |

The default is on because the truncate-only report is derived from telemetry the orchestrator
already keeps: it costs nothing, so it is safe to enable everywhere. A card persisted with an
explicit `False` keeps it off.

**The signature follows the configuration.** Without a summarizer the callable is
`team_activity() -> TeamActivityReport` and `summarize_over` is *absent from the tool schema*, not
merely defaulted off — the model cannot request a summary nothing could produce. With one
configured it becomes `team_activity(summarize_over: int | None = None)`, and passing `None` still
performs zero model calls. Passing an integer is the opt-in: only tasks longer than that many
characters are summarized, and each summary is cached by `message_id`, so asking again is free.
The threshold *is* the consent; there is no eager warming.

A summary that has not arrived within `poll_attempts × poll_delay_seconds` comes back truncated
and is counted in `pending_summaries`.

### How "busy" is derived

An agent with a `ReceivedMessage` and no matching `ProcessedMessage` is mid-handler; the task text
comes from the corresponding `SentMessage`. Three consequences:

- **Busy means exactly one open message.** Actors are sequential, so the open count is
  structurally 0 or 1. A higher count is reported with `suspect=True` rather than as plain
  "working", and never silently dropped.
- **Stale entries are dropped.** A resumed team replays telemetry that can be permanently
  unbalanced — received before the stop, processed never. Anything open longer than
  `stale_after_seconds` is excluded rather than reported as a phantom worker.
- **The caller, tool actors and the user proxy never appear.** The caller is excluded by
  `agent_id`, so renaming cannot slip it through; a human proxy waiting on input is not working.

### The report is lean on purpose

It is read back by the calling model on every invocation, so every field is prompt cost. A member
row carries `name`, `role`, `task`, `summarized`, `started_at` and `suspect` — nothing else. The
derivation keys stay internal: grouping is by `agent_id`, the summary cache is keyed by
`message_id`, and neither reaches the wire. Busy duration is simply
`generated_at − started_at`.

---

## Configuration

### Recipes

```python
from akgentic.tool.team import ActivitySummarizer, GetTeamActivity, GetTeamRoster, TeamTool

TeamTool()                                     # hire/fire/roster/profiles + free team_activity

TeamTool(get_team_activity=False)              # team management only

TeamTool(                                      # summaries on demand; #TeamActivity is created
    get_team_activity=GetTeamActivity(
        summarizer=ActivitySummarizer(model="openai:gpt-5.2-mini"),
        max_task_chars=400,
    ),
)

TeamTool(                                      # awareness without authority
    hire_team_members=False,
    fire_team_members=False,
)

TeamTool(                                      # hiring only via the human /-command surface
    hire_team_members=HireTeamMember(expose={COMMAND}),
    fire_team_members=FireTeamMember(expose={COMMAND}),
)

TeamTool(get_team_roster=False)                # drop the roster from the prompt
```

### Failure modes worth knowing

- `observer()` raises `ValueError` when the observer has no orchestrator.
- Building the summarizing `team_activity` without a bound `#TeamActivity` proxy raises
  `ValueError` — it means `observer()` never ran.
- Every capability captures a **weak** reference to its owning agent. Once that agent stops, hire
  and fire raise `RetriableError("Team is shutting down; …")` and the roster prompt returns `""`,
  so a stopped agent's tools can never pin it in memory.

### Import paths

```python
from akgentic.tool.team import (
    TeamTool, TeamManagementToolObserver,
    HireTeamMember, FireTeamMember, GetTeamRoster, GetRoleProfiles,
    GetTeamActivity, ActivitySummarizer, AgentActivity, TeamActivityReport,
)
```

---

See the [package README](../../../../README.md) for the observer hierarchy, the channel system,
and the deferred-result mechanism the summary cache is built on.
