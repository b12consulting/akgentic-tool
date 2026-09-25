"""The sandbox backend slot: the one class ``workspace_exec`` runs on.

There is no mode, no host probe and no fallback. ``SANDBOX_BACKEND`` names the
single :class:`SandboxBackend` every exec-capable card runs its commands on, and
it ships as :class:`DockerBackend`. On a host without Docker the first command
fails, and nothing picks a weaker backend instead.

The slot is reached through ``akgentic.tool.sandbox``, never through this module
directly: the package attribute is the one a deployment assigns and the one
``#Workspace`` reads. This file is where the default happens to be defined.
"""

from __future__ import annotations

from akgentic.tool.sandbox.backend import SandboxBackend
from akgentic.tool.sandbox.docker import DockerBackend

SANDBOX_BACKEND: type[SandboxBackend] = DockerBackend
"""The backend class every exec-capable card runs on.

A deployment that needs a different executor assigns its own class to the
package attribute, in its own wiring, before any card binds:
``akgentic.tool.sandbox.SANDBOX_BACKEND = MyBackend``. ``#Workspace`` reads the
attribute through the package at call time, when it builds its runner, so an
assignment made after every module was imported is still seen. The class is
constructed with no arguments.
"""
