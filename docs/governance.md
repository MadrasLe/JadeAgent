# Governance

JadeAgent governance is executable runtime policy. It is not prompt text.

## Main Types

- `PolicyBundle`: node-level constitution.
- `TaskPolicy`: per-task restrictions.
- `NodeManifest`: identity, capabilities, trust, tenant, memory mounts.
- `AccessGrant`: resource/action/scope grant.
- `FilesystemPolicy`: read/write root policy.
- `PolicyDecision`: result of a policy check.

## Tool Enforcement

Every tool execution passes through `evaluate_tool_call()` before the Python
function is called. The policy layer can block:

- denied tools;
- tools outside an allowlist;
- writes under read-only policy;
- network effects;
- shell/execute effects;
- delegation effects;
- filesystem reads/writes outside allowed roots;
- memory mount access.

## Governance In JGX

A JGX manifest stores compatibility fingerprints:

- policy hash;
- tool registry hash;
- memory scope hash;
- backend/model fingerprint;
- tenant id;
- capability.

Restore can reject a run if the current runtime no longer matches the saved
policy or tool surface.

## Design Principle

The LLM can request a tool call, but the runtime decides whether it executes.
This means model behavior is advisory; runtime policy is authoritative.

## Next Governance Work

- Compare old and new policies and allow restore only if the new policy is equal
  or stricter.
- Add signed policy bundles.
- Emit policy explain records in the CLI.
- Include idempotency decisions in audit output.

