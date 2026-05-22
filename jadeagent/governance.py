"""
Executable governance primitives for JadeAgent nodes and tasks.

These rules are enforced by runtime code before a tool, memory operation, or
delegation is executed. They are not prompt instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.tools import Tool


TRUST_TIER_RANK = {
    "isolated": 0,
    "standard": 1,
    "trusted": 2,
    "privileged": 3,
}


def _as_tuple(values: list[str] | tuple[str, ...] | set[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(v) for v in values if str(v).strip())


def trust_tier_allows(actual: str, minimum: str | None) -> bool:
    if not minimum:
        return True
    return TRUST_TIER_RANK.get(actual, 0) >= TRUST_TIER_RANK.get(minimum, 0)


@dataclass(frozen=True)
class FilesystemPolicy:
    allow_read_all: bool = True
    allow_write_all: bool = True
    allow_read_roots: tuple[str, ...] = ()
    allow_write_roots: tuple[str, ...] = ()
    deny_roots: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_read_all": self.allow_read_all,
            "allow_write_all": self.allow_write_all,
            "allow_read_roots": list(self.allow_read_roots),
            "allow_write_roots": list(self.allow_write_roots),
            "deny_roots": list(self.deny_roots),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FilesystemPolicy | None:
        if not data:
            return None
        return cls(
            allow_read_all=bool(data.get("allow_read_all", True)),
            allow_write_all=bool(data.get("allow_write_all", True)),
            allow_read_roots=_as_tuple(data.get("allow_read_roots")),
            allow_write_roots=_as_tuple(data.get("allow_write_roots")),
            deny_roots=_as_tuple(data.get("deny_roots")),
        )


@dataclass(frozen=True)
class PolicyBundle:
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    allow_dynamic_tool_creation: bool = True
    allow_network: bool = True
    allow_shell: bool = True
    allow_delegate: bool = True
    read_only: bool = False
    enforce_declared_effects: bool = False
    filesystem: FilesystemPolicy = field(default_factory=FilesystemPolicy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools),
            "denied_tools": list(self.denied_tools),
            "allow_dynamic_tool_creation": self.allow_dynamic_tool_creation,
            "allow_network": self.allow_network,
            "allow_shell": self.allow_shell,
            "allow_delegate": self.allow_delegate,
            "read_only": self.read_only,
            "enforce_declared_effects": self.enforce_declared_effects,
            "filesystem": self.filesystem.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PolicyBundle | None:
        if not data:
            return None
        return cls(
            allowed_tools=_as_tuple(data.get("allowed_tools")),
            denied_tools=_as_tuple(data.get("denied_tools")),
            allow_dynamic_tool_creation=bool(data.get("allow_dynamic_tool_creation", True)),
            allow_network=bool(data.get("allow_network", True)),
            allow_shell=bool(data.get("allow_shell", True)),
            allow_delegate=bool(data.get("allow_delegate", True)),
            read_only=bool(data.get("read_only", False)),
            enforce_declared_effects=bool(data.get("enforce_declared_effects", False)),
            filesystem=FilesystemPolicy.from_dict(data.get("filesystem")) or FilesystemPolicy(),
        )


@dataclass(frozen=True)
class TaskPolicy:
    read_only: bool = False
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    allowed_memory_mounts: tuple[str, ...] = ()
    denied_memory_mounts: tuple[str, ...] = ()
    allow_dynamic_tool_creation: bool | None = None
    allow_network: bool | None = None
    allow_shell: bool | None = None
    allow_delegate: bool | None = None
    filesystem: FilesystemPolicy | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "read_only": self.read_only,
            "allowed_tools": list(self.allowed_tools),
            "denied_tools": list(self.denied_tools),
            "allowed_memory_mounts": list(self.allowed_memory_mounts),
            "denied_memory_mounts": list(self.denied_memory_mounts),
        }
        if self.allow_dynamic_tool_creation is not None:
            data["allow_dynamic_tool_creation"] = self.allow_dynamic_tool_creation
        if self.allow_network is not None:
            data["allow_network"] = self.allow_network
        if self.allow_shell is not None:
            data["allow_shell"] = self.allow_shell
        if self.allow_delegate is not None:
            data["allow_delegate"] = self.allow_delegate
        if self.filesystem is not None:
            data["filesystem"] = self.filesystem.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TaskPolicy | None:
        if not data:
            return None
        return cls(
            read_only=bool(data.get("read_only", False)),
            allowed_tools=_as_tuple(data.get("allowed_tools")),
            denied_tools=_as_tuple(data.get("denied_tools")),
            allowed_memory_mounts=_as_tuple(data.get("allowed_memory_mounts")),
            denied_memory_mounts=_as_tuple(data.get("denied_memory_mounts")),
            allow_dynamic_tool_creation=data.get("allow_dynamic_tool_creation"),
            allow_network=data.get("allow_network"),
            allow_shell=data.get("allow_shell"),
            allow_delegate=data.get("allow_delegate"),
            filesystem=FilesystemPolicy.from_dict(data.get("filesystem")),
        )


@dataclass(frozen=True)
class AccessGrant:
    """Grant a resource pattern and optional scope to a node."""

    resource: str
    actions: tuple[str, ...] = ()
    scope: str = ""

    def matches(self, resource: str, action: str = "*", scope: str = "") -> bool:
        if not fnmatch(resource, self.resource):
            return False

        if self.actions and "*" not in self.actions and action not in self.actions:
            return False

        if self.scope:
            if not scope:
                return False
            if not fnmatch(scope, self.scope):
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "actions": list(self.actions),
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AccessGrant | None:
        if not data:
            return None
        return cls(
            resource=str(data.get("resource", "")),
            actions=_as_tuple(data.get("actions")),
            scope=str(data.get("scope", "")),
        )


@dataclass(frozen=True)
class MemoryMount:
    """Memory surface attached to a node manifest."""

    name: str
    backend: str = "local"
    mode: str = "rw"
    shared: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "mode": self.mode,
            "shared": self.shared,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MemoryMount | None:
        if not data:
            return None
        return cls(
            name=str(data.get("name", "")),
            backend=str(data.get("backend", "local")),
            mode=str(data.get("mode", "rw")),
            shared=bool(data.get("shared", False)),
        )


@dataclass(frozen=True)
class NodeManifest:
    """Canonical runtime identity for a node."""

    node_id: str
    role: str = "agent"
    capabilities: tuple[str, ...] = ()
    description: str = ""
    constitution: PolicyBundle = field(default_factory=PolicyBundle)
    access: tuple[AccessGrant, ...] = ()
    memory_mounts: tuple[MemoryMount, ...] = ()
    trust_tier: str = "standard"
    labels: tuple[str, ...] = ()
    tenant_id: str = ""
    delegation_allowlist: tuple[str, ...] = ()
    protocol_version: str = "jade.mesh.v1"
    auth_schemes: tuple[str, ...] = ("hmac-sha256",)
    agent_card_url: str = ""

    def routing_metadata(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "trust_tier": self.trust_tier,
            "labels": list(self.labels),
            "tenant_id": self.tenant_id,
            "delegation_allowlist": list(self.delegation_allowlist),
            "protocol_version": self.protocol_version,
            "auth_schemes": list(self.auth_schemes),
            "agent_card_url": self.agent_card_url,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "capabilities": list(self.capabilities),
            "description": self.description,
            "constitution": self.constitution.to_dict(),
            "access": [grant.to_dict() for grant in self.access],
            "memory_mounts": [mount.to_dict() for mount in self.memory_mounts],
            "trust_tier": self.trust_tier,
            "labels": list(self.labels),
            "tenant_id": self.tenant_id,
            "delegation_allowlist": list(self.delegation_allowlist),
            "protocol_version": self.protocol_version,
            "auth_schemes": list(self.auth_schemes),
            "agent_card_url": self.agent_card_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NodeManifest | None:
        if not data:
            return None
        return cls(
            node_id=str(data.get("node_id", "")),
            role=str(data.get("role", "agent")),
            capabilities=_as_tuple(data.get("capabilities")),
            description=str(data.get("description", "")),
            constitution=PolicyBundle.from_dict(data.get("constitution")) or PolicyBundle(),
            access=tuple(
                grant
                for grant in (AccessGrant.from_dict(item) for item in data.get("access", []))
                if grant is not None
            ),
            memory_mounts=tuple(
                mount
                for mount in (MemoryMount.from_dict(item) for item in data.get("memory_mounts", []))
                if mount is not None
            ),
            trust_tier=str(data.get("trust_tier", "standard")),
            labels=_as_tuple(data.get("labels")),
            tenant_id=str(data.get("tenant_id", "")),
            delegation_allowlist=_as_tuple(data.get("delegation_allowlist")),
            protocol_version=str(data.get("protocol_version", "jade.mesh.v1")),
            auth_schemes=_as_tuple(data.get("auth_schemes")) or ("hmac-sha256",),
            agent_card_url=str(data.get("agent_card_url", "")),
        )


@dataclass(frozen=True)
class ResourceRequirement:
    resource: str
    action: str = "execute"
    scope: str = ""


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str = "ok"
    resource: str = ""
    action: str = ""
    scope: str = ""


def _resolve_path(raw_path: str, cwd: str | None) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        base = Path(cwd or Path.cwd())
        path = base / path
    return path.resolve(strict=False)


def _path_within(candidate: Path, root: str, cwd: str | None) -> bool:
    resolved_root = _resolve_path(root, cwd)
    try:
        candidate.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def _iter_paths(value: Any) -> list[str]:
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, (list, tuple, set)):
        results: list[str] = []
        for item in value:
            results.extend(_iter_paths(item))
        return results
    return []


def _collect_paths(arguments: dict[str, Any], arg_names: tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    for arg_name in arg_names:
        if arg_name in arguments:
            paths.extend(_iter_paths(arguments[arg_name]))
    return paths


def _check_filesystem_policy(
    fs_policy: FilesystemPolicy,
    raw_path: str,
    *,
    mode: str,
    cwd: str | None,
) -> PolicyDecision:
    candidate = _resolve_path(raw_path, cwd)

    for deny_root in fs_policy.deny_roots:
        if _path_within(candidate, deny_root, cwd):
            return PolicyDecision(
                False,
                f"{mode} access to '{candidate}' is denied by root policy.",
                resource=f"fs.{mode}",
                action=mode,
                scope=str(candidate),
            )

    if mode == "read":
        allow_all = fs_policy.allow_read_all
        allow_roots = fs_policy.allow_read_roots
    else:
        allow_all = fs_policy.allow_write_all
        allow_roots = fs_policy.allow_write_roots

    if allow_all:
        return PolicyDecision(True)

    if allow_roots:
        if any(_path_within(candidate, root, cwd) for root in allow_roots):
            return PolicyDecision(True)
        return PolicyDecision(
            False,
            f"{mode} access to '{candidate}' is outside allowed roots.",
            resource=f"fs.{mode}",
            action=mode,
            scope=str(candidate),
        )

    return PolicyDecision(
        False,
        f"{mode} access to '{candidate}' is blocked by default-deny policy.",
        resource=f"fs.{mode}",
        action=mode,
        scope=str(candidate),
    )


def _bool_allowed(base_allowed: bool, task_override: bool | None) -> bool:
    if not base_allowed:
        return False
    if task_override is None:
        return base_allowed
    return bool(task_override)


def check_access(
    node_manifest: NodeManifest | None,
    requirement: ResourceRequirement,
) -> PolicyDecision:
    if node_manifest is None or not node_manifest.access:
        return PolicyDecision(True)

    for grant in node_manifest.access:
        if grant.matches(requirement.resource, requirement.action, requirement.scope):
            return PolicyDecision(True)

    return PolicyDecision(
        False,
        (
            f"resource '{requirement.resource}' action '{requirement.action}'"
            " is not granted for this node."
        ),
        resource=requirement.resource,
        action=requirement.action,
        scope=requirement.scope,
    )


def _infer_action_for_resource(resource: str) -> str:
    if resource.startswith("tool.execute:"):
        return "execute"
    if resource == "fs.read" or resource.startswith("memory.read:"):
        return "read"
    if resource == "fs.write" or resource.startswith("memory.write:"):
        return "write"
    if resource == "network.outbound":
        return "network"
    if resource == "shell.execute":
        return "execute"
    if resource.startswith("delegate.capability:"):
        return "delegate"
    return "execute"


def derive_tool_resource_requirements(tool: Tool, arguments: dict[str, Any]) -> list[ResourceRequirement]:
    requirements: list[ResourceRequirement] = [
        ResourceRequirement(resource=f"tool.execute:{tool.name}", action="execute")
    ]

    raw_refs = tuple(getattr(tool, "resource_refs", ()))
    for ref in raw_refs:
        if isinstance(ref, str):
            requirements.append(ResourceRequirement(
                resource=ref,
                action=_infer_action_for_resource(ref),
            ))
        elif isinstance(ref, dict):
            resource = str(ref.get("resource", ""))
            requirements.append(ResourceRequirement(
                resource=resource,
                action=str(ref.get("action", _infer_action_for_resource(resource))),
                scope=str(ref.get("scope", "")),
            ))
        elif isinstance(ref, (list, tuple)):
            resource = str(ref[0]) if len(ref) > 0 else ""
            action = str(ref[1]) if len(ref) > 1 else _infer_action_for_resource(resource)
            scope = str(ref[2]) if len(ref) > 2 else ""
            requirements.append(ResourceRequirement(resource=resource, action=action, scope=scope))

    effects = set(getattr(tool, "effects", ()))
    read_path_args = tuple(getattr(tool, "read_path_args", ()))
    write_path_args = tuple(getattr(tool, "write_path_args", ()))
    metadata = dict(getattr(tool, "metadata", {}))

    if read_path_args:
        requirements.append(ResourceRequirement(resource="fs.read", action="read"))
    if write_path_args:
        requirements.append(ResourceRequirement(resource="fs.write", action="write"))
    if "network" in effects:
        requirements.append(ResourceRequirement(resource="network.outbound", action="network"))
    if "shell" in effects or "execute" in effects:
        requirements.append(ResourceRequirement(resource="shell.execute", action="execute"))
    if "delegate" in effects:
        delegate_cap = str(metadata.get("delegate_capability", "*"))
        requirements.append(ResourceRequirement(
            resource=f"delegate.capability:{delegate_cap}",
            action="delegate",
        ))

    mount_name = str(metadata.get("memory_mount", ""))
    if mount_name:
        if "memory_read" in effects:
            requirements.append(ResourceRequirement(
                resource=f"memory.read:{mount_name}",
                action="read",
                scope=mount_name,
            ))
        if "memory_write" in effects:
            requirements.append(ResourceRequirement(
                resource=f"memory.write:{mount_name}",
                action="write",
                scope=mount_name,
            ))

    seen: set[tuple[str, str, str]] = set()
    deduped: list[ResourceRequirement] = []
    for requirement in requirements:
        key = (requirement.resource, requirement.action, requirement.scope)
        if key in seen or not requirement.resource:
            continue
        seen.add(key)
        deduped.append(requirement)
    return deduped


def memory_mount_allowed(
    mount_name: str,
    *,
    task_policy: TaskPolicy | None = None,
) -> PolicyDecision:
    task_policy = task_policy or TaskPolicy()
    if task_policy.denied_memory_mounts and mount_name in task_policy.denied_memory_mounts:
        return PolicyDecision(False, f"memory mount '{mount_name}' is denied by task policy.")
    if task_policy.allowed_memory_mounts and mount_name not in task_policy.allowed_memory_mounts:
        return PolicyDecision(False, f"memory mount '{mount_name}' is not allowed by task policy.")
    return PolicyDecision(True)


def evaluate_tool_call(
    tool: Tool,
    arguments: dict[str, Any],
    *,
    node_manifest: NodeManifest | None = None,
    task_policy: TaskPolicy | None = None,
    cwd: str | None = None,
) -> PolicyDecision:
    constitution = node_manifest.constitution if node_manifest is not None else PolicyBundle()
    task_policy = task_policy or TaskPolicy()
    effects = set(getattr(tool, "effects", ()))
    read_path_args = tuple(getattr(tool, "read_path_args", ()))
    write_path_args = tuple(getattr(tool, "write_path_args", ()))

    if tool.name in constitution.denied_tools or tool.name in task_policy.denied_tools:
        return PolicyDecision(False, f"tool '{tool.name}' is explicitly denied.")

    if constitution.allowed_tools and tool.name not in constitution.allowed_tools:
        return PolicyDecision(False, f"tool '{tool.name}' is not in the node allowlist.")

    if task_policy.allowed_tools and tool.name not in task_policy.allowed_tools:
        return PolicyDecision(False, f"tool '{tool.name}' is not in the task allowlist.")

    dynamic_allowed = _bool_allowed(
        constitution.allow_dynamic_tool_creation,
        task_policy.allow_dynamic_tool_creation,
    )
    if tool.name == "create_and_use_tool" and not dynamic_allowed:
        return PolicyDecision(False, "dynamic tool creation is disabled for this node/task.")

    read_only = constitution.read_only or task_policy.read_only
    if read_only and ("write" in effects or "delete" in effects or write_path_args):
        return PolicyDecision(False, f"tool '{tool.name}' performs write effects under a read-only policy.")

    if not _bool_allowed(constitution.allow_network, task_policy.allow_network) and "network" in effects:
        return PolicyDecision(False, f"tool '{tool.name}' requires network access, which is disabled.")

    if not _bool_allowed(constitution.allow_shell, task_policy.allow_shell) and (
        "shell" in effects or "execute" in effects
    ):
        return PolicyDecision(False, f"tool '{tool.name}' requires shell/execute access, which is disabled.")

    if not _bool_allowed(constitution.allow_delegate, task_policy.allow_delegate) and "delegate" in effects:
        return PolicyDecision(False, f"tool '{tool.name}' performs delegation, which is disabled.")

    has_declared_metadata = bool(
        effects
        or read_path_args
        or write_path_args
        or getattr(tool, "resource_refs", ())
    )
    if constitution.enforce_declared_effects and not has_declared_metadata:
        return PolicyDecision(False, f"tool '{tool.name}' has no declared effects metadata under strict policy.")

    for requirement in derive_tool_resource_requirements(tool, arguments):
        decision = check_access(node_manifest, requirement)
        if not decision.allowed:
            return decision

    for raw_path in _collect_paths(arguments, read_path_args):
        decision = _check_filesystem_policy(constitution.filesystem, raw_path, mode="read", cwd=cwd)
        if not decision.allowed:
            return decision
        if task_policy.filesystem is not None:
            decision = _check_filesystem_policy(task_policy.filesystem, raw_path, mode="read", cwd=cwd)
            if not decision.allowed:
                return decision

    for raw_path in _collect_paths(arguments, write_path_args):
        decision = _check_filesystem_policy(constitution.filesystem, raw_path, mode="write", cwd=cwd)
        if not decision.allowed:
            return decision
        if task_policy.filesystem is not None:
            decision = _check_filesystem_policy(task_policy.filesystem, raw_path, mode="write", cwd=cwd)
            if not decision.allowed:
                return decision

    return PolicyDecision(True)
