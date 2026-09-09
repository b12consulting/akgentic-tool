"""DockerBackend — an ephemeral Docker container as compute for one tree.

``DockerSandboxActor`` is kept beside it as a thin delegator: it owns the
actor lifecycle and the state it publishes, and nothing else.
"""

from __future__ import annotations

import importlib.resources
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from akgentic.tool.sandbox.actor import SandboxActor
from akgentic.tool.sandbox.backend import (
    DEFAULT_BACKEND_TIMEOUT_S,
    ExecResult,
    ProcessBackend,
    validate_command,
)

logger = logging.getLogger(__name__)

SANDBOX_IMAGE: str = "akgentic-sandbox:v2"
"""The bundled image, tagged by revision rather than by ``latest``.

The tag moved with the Dockerfile that relocated ``uv`` off ``/root``.
:meth:`DockerBackend._ensure_image` skips the build whenever *any* image carries
the tag, so a host holding an image built from the previous file would have kept
it for ever — and under ``--user <non-root>`` that image's ``uv`` sits behind
Debian's mode-0700 ``/root`` and is unreachable. Moving the tag makes the check
miss and the correct image get built. The cost is one build per host that had the
old image, paid once; the old tag is left on disk, because this code must never
remove an image it did not create.
"""

SANDBOX_IMAGE_BUILD_TIMEOUT_S: float = 600.0
"""Wall-clock budget for building :data:`SANDBOX_IMAGE` from the bundled file.

**Ten minutes, and the arithmetic that picks it.** A first-time build pulls a
Debian base and installs LibreOffice, the PDF tooling, Node 18 and a Python
scientific stack, so *minutes* is the honest figure; ten of them is a ceiling
rather than an expectation. It is deliberately far above any exec budget: the
run that triggers a build has long since handed its caller a run id, and the
alternative to waiting is a deployment where docker mode never works at all.

**A bound is owed because the build is on the worker thread.** It used to run
inside the sandbox actor's ``on_start`` where an unbounded 78-second build was
observed on a team whose command then took 0.07 s. The run now blocks only
itself, which is what makes a slow build tolerable — but *unbounded* on a worker
thread is the hang class this design removes everywhere else, so the build gets a
budget and a failure gets a reason.
"""

DOCKER_EXEC_TIMEOUT: float = DEFAULT_BACKEND_TIMEOUT_S
"""Default budget for one ``docker exec``, when the caller names none.

This used to be 60 s — twice the orchestrator's 30 s stop backstop — which made
docker the one backend able to hold a team's teardown open past the point that
teardown gives up. A Python thread cannot be cancelled, so the difference is real
wall clock, not a formality. Docker is no longer the exception.
"""

DOCKER_RM_TIMEOUT_S: float = 10.0
"""Budget for the ``docker rm -f`` that :meth:`DockerBackend.stop` issues.

**This runs on the actor's thread, inside ``on_stop``**, so an unbounded call
here is a teardown hang — the exact class the executor's ordered shutdown was
built to remove. A daemon that answers removes a container in well under a
second; a daemon that has wedged never answers, and without a bound the actor
would sit in ``on_stop`` until the orchestrator's 30 s backstop gave up on it.

**Ten seconds, and the arithmetic.** Teardown's bounded drain is
``EXEC_SHUTDOWN_GRACE_S`` (3 s) and this follows it, so the whole of exec
teardown is ~13 s worst case — under the 30 s backstop, and the figure the grace
was originally sized for. If either number changes, state the sum again.
"""

CONTAINER_NAME_PREFIX: str = "akgentic-sandbox-"
"""What every container this backend creates is named, before its random half."""

WORKSPACE_PATH_LABEL: str = "akgentic.workspace_path"
"""The label carrying the tree a container was started on.

The container **name** carries nothing, so this is the only thing a host-side
reaper can key on — and keying on a label is deliberate rather than incidental:
a workspace path can end in a metadata leaf, and a leaf belongs in
``docker inspect`` rather than in every ``docker ps`` line on the host.
"""

SANDBOX_HOME: str = "/home/agent"
"""``$HOME`` inside the container, chosen here rather than inherited.

**Docker gives a numeric ``--user`` with no ``/etc/passwd`` entry ``HOME=/``**,
which is on the read-only root. So the backend has to decide what ``$HOME`` is
before it can mount a writable one, and :meth:`DockerBackend._run_argv` sets
``-e HOME=`` and a ``--tmpfs`` from this one constant so the two cannot drift.
"""

TMPFS_MOUNT_OPTIONS: str = "rw,exec,mode=1777,size=512m"
"""What makes a tmpfs usable under ``--user <non-root>`` on a read-only root.

**What a bare ``--tmpfs /path`` actually gives, observed on the daemon rather
than assumed.** It mounts ``rw,nosuid,nodev,noexec`` with **no** ``size=`` — the
kernel's default, which is half of the machine's RAM — at the kernel's default
tmpfs mode, ``1777`` owned by root. So a bare tmpfs *is* writable by a non-root
uid; the wall is elsewhere. ``noexec`` is it: ``pip`` builds wheels and runs
build backends out of ``TMPDIR``, and a script on a ``noexec`` mount fails with
*Permission denied* however it is chmod'ed — so ``exec`` is what turns the mount
into a fix. The size is stated because the default is *unbounded* for practical
purposes: a runaway install would eat the daemon's memory rather than fail. And
``mode=1777`` restates the kernel default explicitly, so the spec can pin it and
nobody has to know what the default was.
"""

SANDBOX_TMPDIR: str = "/tmp"
"""The second writable tmpfs — a path inside the container, never on the host.

Separate from ``$HOME`` rather than folded into it, so two concerns that are
separately sized and separately inspectable are two mounts, and ``docker
inspect`` reads as the design intended rather than as a coincidence.
"""

SANDBOX_GIT_CONFIG: tuple[tuple[str, str], ...] = (
    ("safe.directory", "*"),
    ("user.name", "akgentic-sandbox"),
    ("user.email", "sandbox@akgentic"),
)
"""Git's *command* scope, passed by environment because nothing may be written.

**``safe.directory`` is honoured only in git's protected configuration**, which
is system, global **and command** scope — and ``GIT_CONFIG_COUNT`` /
``GIT_CONFIG_KEY_n`` / ``GIT_CONFIG_VALUE_n`` *is* command scope. That is why it
works here and why a repository-local ``git config`` cannot do it. A reader who
assumes ``safe.directory`` is global-only will "fix" this into a
``git config --global`` that cannot run: the home directory is a tmpfs on a
read-only root and nothing should be writing configuration into it. The ``*``
wildcard needs git >= 2.35.3; the bundled image is Debian bookworm with 2.39.

The identity is the **container's**, not any agent's — :meth:`DockerBackend.start`
has no agent and never will, because the backend is per tree and the agent is per
run. The domain matches the journal's synthetic one; it is written out rather
than imported, because ``sandbox/`` importing ``workspace/`` would reverse the
one-directional edge between them. **This identity governs only an agent's own
``git commit`` under ``/workspace/.git``** — the journal lives at the sibling
``<root>.git`` on the host, outside every mount.
"""

_BUILD_STDERR_TAIL_LINES: int = 20
"""How much of a failed build's stderr goes into the raised message."""


def _stderr_tail(stderr: str | bytes | None) -> str:
    """Last few lines of a build's captured stderr, ready to paste into a message."""
    if stderr is None:
        return ""
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
    lines = text.strip().splitlines()
    return "\n".join(lines[-_BUILD_STDERR_TAIL_LINES:])


def _build_failure(reason: str, stderr: str | bytes | None) -> RuntimeError:
    """Build the error a failed image build raises, naming both remedies."""
    tail = _stderr_tail(stderr)
    message = (
        f"Failed to build {SANDBOX_IMAGE}: {reason}. Either pre-build the image on this "
        f"host, or set AKGENTIC_SANDBOX_IMAGE to an image that is already available."
    )
    return RuntimeError(f"{message}\nLast build output:\n{tail}" if tail else message)


class DockerBackend(ProcessBackend):
    """An ephemeral Docker container as compute for one workspace tree.

    A container is created by :meth:`start` — which the worker thread calls
    lazily, before the first command — and **removed** by :meth:`stop`. It holds
    nothing worth keeping between runs: the only mount whose writes outlive it is
    the bind mount at ``/workspace``, which is
    ``{AKGENTIC_WORKSPACES_ROOT}/{workspace_path}`` — the path the card already
    resolved, joined and never re-derived, so the mounted directory is by
    construction the one the write gate and the journal are working on.

    **The root is read only, and three further things are what make that
    usable.** ``--user`` matched to the host, a writable ``$HOME`` and ``/tmp``
    on tmpfs, and git configured by environment. Two of the three is a wall
    rather than a partial fix, so :meth:`_run_argv` builds them as one unit.

    **The name is opaque and per lifetime.** Nothing parses it and nothing
    persists it; a reaper keys on :data:`WORKSPACE_PATH_LABEL` instead. The
    randomness is forced rather than stylistic — :meth:`stop` removes the
    container, so any name that is a function of the team or the tree would
    collide with its own predecessor on the next ``start()`` and ``docker run
    --name`` refuses that outright.

    Args:
        team_id: Unused here, and kept. It is
            :class:`~akgentic.tool.sandbox.backend.ProcessBackend`'s uniform
            constructor and ``resolve_mode``'s keyword; this backend was its last
            reader until the container name stopped carrying it.
    """

    def __init__(self, team_id: str = "") -> None:
        super().__init__(team_id)
        self.container_name: str | None = None
        """The container this backend runs in, set by :meth:`start`."""

    def _resolved_image(self) -> str:
        """Image name for docker run: AKGENTIC_SANDBOX_IMAGE override or the default."""
        return os.environ.get("AKGENTIC_SANDBOX_IMAGE", SANDBOX_IMAGE)

    def _ensure_image(self) -> None:
        """Build SANDBOX_IMAGE from the bundled Dockerfile if not present locally.

        Skipped entirely when AKGENTIC_SANDBOX_IMAGE is set — the caller owns the
        image. **Nothing is ever pulled.** :data:`SANDBOX_IMAGE` is published to
        no registry, so a ``docker pull`` would reach Docker Hub for a name this
        project chose and either 404 or fetch a stranger's image. That is a
        supply-chain hazard rather than a fallback.
        """
        if os.environ.get("AKGENTIC_SANDBOX_IMAGE"):
            return
        check = subprocess.run(
            ["docker", "images", "-q", SANDBOX_IMAGE], capture_output=True, text=True
        )
        if check.stdout.strip():
            return
        self._build_image()

    def _build_image(self) -> None:
        """Build the bundled Dockerfile under a budget, with its output captured.

        Raises:
            RuntimeError: The build failed or outlived
                :data:`SANDBOX_IMAGE_BUILD_TIMEOUT_S`. Both carry the tail of the
                captured output and name both remedies; a raw ``TimeoutExpired``
                would reach the run's report naming neither.
        """
        logger.warning(
            "Building %s from the bundled Dockerfile — first use on this host, "
            "and it may take several minutes.",
            SANDBOX_IMAGE,
        )
        dockerfile_text = (
            importlib.resources.files("akgentic.tool.sandbox")
            .joinpath("sandbox.Dockerfile")
            .read_text(encoding="utf-8")
        )
        started = time.monotonic()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")
            try:
                result = subprocess.run(
                    ["docker", "build", "-t", SANDBOX_IMAGE, tmpdir],
                    capture_output=True,
                    text=True,
                    timeout=SANDBOX_IMAGE_BUILD_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired as exc:
                raise _build_failure(
                    f"it outlived the {SANDBOX_IMAGE_BUILD_TIMEOUT_S:.0f}s build budget",
                    exc.stderr,
                ) from exc
        if result.returncode != 0:
            raise _build_failure(f"docker build exited {result.returncode}", result.stderr)
        logger.info("Built %s in %.1fs.", SANDBOX_IMAGE, time.monotonic() - started)

    def _run_argv(self, container_name: str, workspace_path: str, volume: str) -> list[str]:
        """The whole ``docker run`` command line, in the order docker reads it.

        **``docker run`` is positional in the way that matters**: everything
        before the image name is an option to ``docker run`` and everything after
        it is the container's own command. A flag that moves after the image
        silently becomes an argument to ``sleep``, which is why the specs assert
        this vector in order rather than for the presence of a flag.
        """
        argv = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--label",
            f"{WORKSPACE_PATH_LABEL}={workspace_path}",
            "--read-only",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            f"HOME={SANDBOX_HOME}",
            "--tmpfs",
            f"{SANDBOX_TMPDIR}:{TMPFS_MOUNT_OPTIONS}",
            "--tmpfs",
            f"{SANDBOX_HOME}:{TMPFS_MOUNT_OPTIONS}",
        ]
        # Git's *command* scope — see SANDBOX_GIT_CONFIG for why ``safe.directory``
        # is honoured here and cannot be set from a repository-local config. The
        # count is derived from the tuple rather than written out: git reads only
        # the first COUNT pairs, so a hand-written count that was not incremented
        # drops the settings after it with no error anywhere.
        argv += ["-e", f"GIT_CONFIG_COUNT={len(SANDBOX_GIT_CONFIG)}"]
        for index, (key, value) in enumerate(SANDBOX_GIT_CONFIG):
            argv += [
                "-e",
                f"GIT_CONFIG_KEY_{index}={key}",
                "-e",
                f"GIT_CONFIG_VALUE_{index}={value}",
            ]
        argv += ["-v", volume, "-w", "/workspace", self._resolved_image(), "sleep", "infinity"]
        return argv

    def start(self, workspace_path: str) -> None:
        """Create the container this backend runs in, mounting *workspace_path*.

        One ``docker run`` and nothing else. **There is no reuse branch**: a
        container created on the first command and removed by :meth:`stop` is
        never there to be found again, so the ``docker ps -a`` probe that used to
        precede this was a branch whose precondition can no longer hold.

        The name is generated here and assigned **only after the run succeeds**,
        so a failed start leaves ``container_name`` at ``None`` and every guard
        that reads it stays reachable.

        Raises:
            RuntimeError: ``docker`` is not on ``PATH``, the image could not be
                built, or ``docker run`` exited non-zero.
        """
        if shutil.which("docker") is None:
            raise RuntimeError("docker CLI not found on PATH — cannot start DockerSandboxActor")
        self._ensure_image()
        base = os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces")
        volume = f"{(Path(base) / workspace_path).resolve()}:/workspace"
        container_name = f"{CONTAINER_NAME_PREFIX}{uuid4().hex[:12]}"
        result = subprocess.run(
            self._run_argv(container_name, workspace_path, volume),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"docker run failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        self.container_name = container_name

    def exec(self, cmd: str, cwd: str = "", timeout: float | None = None) -> ExecResult:
        """Execute a command in this backend's container.

        Only ``<root>:/workspace`` is mounted (see :meth:`start`), so the
        sibling journal at ``<root>.git`` is not visible inside the container.

        Raises:
            RuntimeError: If :meth:`start` has not run, so there is no container
                to exec in. An explicit raise rather than an ``assert``: under
                ``python -O`` the assert is stripped and the argv would carry a
                literal ``None`` straight to the docker CLI.
        """
        if self.container_name is None:
            raise RuntimeError(
                "DockerBackend.exec was called before start() — there is no container to run in."
            )
        effective_workdir = f"/workspace/{cwd}" if cwd else "/workspace"
        docker_cmd = [
            "docker",
            "exec",
            "-w",
            effective_workdir,
            self.container_name,
        ] + validate_command(cmd)
        return self._run(
            docker_cmd,
            timeout=DOCKER_EXEC_TIMEOUT if timeout is None else timeout,
        )

    def kill(self) -> None:
        """End the local ``docker exec`` client — **not**, necessarily, what runs inside.

        The signal reaches the ``docker exec`` process on this host. The command
        it started inside the container is a child of the container's own init,
        not of this process, so it may keep running after this returns — and it
        keeps writing to ``/workspace`` until the container is gone.

        :meth:`stop` is the authoritative end: it **removes** the container,
        which does end everything inside it. Use ``kill`` to abandon a run,
        ``stop`` to be certain it is over. That the container is ephemeral is
        what bounds this: the process outlives the ``kill`` but not the teardown.
        """
        super().kill()

    def _release(self) -> None:
        """Remove the container. **Nothing here raises**, on any path.

        This runs from ``stop()``, which runs from ``#Workspace``'s teardown and
        from ``configure_exec`` replacing a runner — neither is a place an
        exception belongs.

        ``docker rm -f`` rather than ``docker stop`` then ``docker rm``: one round
        trip, and it does not spend ``docker stop``'s ten-second SIGTERM grace on
        a container whose only purpose was to hold a process the caller has
        already given up on.

        The five cases, and **a container that is already gone is the outcome
        that was asked for** rather than a problem to report: an unstarted
        backend issues no docker command at all; a successful removal logs at
        debug; a *no such container* failure logs at debug; anything else — a
        daemon that has gone away, ``docker`` no longer on ``PATH``, a removal
        that outlived :data:`DOCKER_RM_TIMEOUT_S` — logs at warning. The name is
        cleared first, so a second ``stop()`` is a no-op and a later ``exec``
        cannot address a container that is no longer there.

        **The removal is bounded because this is the actor's thread.** A daemon
        that never answers would otherwise hold ``on_stop`` open until the
        orchestrator's backstop gave up on it. A container the bound abandoned
        may still be running; it carries :data:`WORKSPACE_PATH_LABEL`, which is
        what a host-side reaper keys on.
        """
        container_name = self.container_name
        if container_name is None:
            return
        self.container_name = None
        try:
            result = subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=DOCKER_RM_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "Removing sandbox container %s outlived its %.0fs budget — abandoning it "
                "to the reaper.",
                container_name,
                DOCKER_RM_TIMEOUT_S,
            )
            return
        except OSError as exc:
            logger.warning("Could not remove sandbox container %s: %s", container_name, exc)
            return
        if result.returncode == 0:
            logger.debug("Removed sandbox container %s.", container_name)
        elif "no such container" in result.stderr.lower():
            logger.debug("Sandbox container %s was already gone.", container_name)
        else:
            logger.warning(
                "Removing sandbox container %s failed (exit %d): %s",
                container_name,
                result.returncode,
                result.stderr.strip(),
            )


class DockerSandboxActor(SandboxActor):
    """Docker sandbox actor for one tree.

    A thin delegator over :class:`DockerBackend`, which holds the implementation.
    """

    _backend: DockerBackend | None = None

    def _start_sandbox(self) -> None:
        """Build the backend and start its container. **Nothing is published.**

        The container name used to be written to ``SandboxState`` and notified,
        which put it in the team's checkpointed event stream. Nothing needs to
        find the container again after it is gone, and a restored team carrying
        the name of a removed container is worse than one carrying nothing: it
        reads as a handle. With no state left to change there is nothing to
        notify about either.
        """
        self._backend = DockerBackend(self.config.team_id)
        self._backend.start(self.config.workspace_path)

    def _stop_sandbox(self) -> None:
        if self._backend is not None:
            self._backend.stop()

    def _exec(self, cmd: str, cwd: str, timeout: float | None = None) -> ExecResult:
        assert self._backend is not None
        return self._backend.exec(cmd, cwd, timeout)
