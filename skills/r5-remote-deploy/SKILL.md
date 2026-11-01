---
name: r5-remote-deploy
description: Deploy and operate this R5 benchmark on a user-selected Linux SSH host, optionally using a local Codex bridge. Use for remote setup, synchronization, status, collection, or explicitly requested task launches. Local experiments do not need this skill.
---

# Optional remote deployment

The normal benchmark runs on the reader's own computer through `tools/launch.py`,
`tools/reproduce.py`, and `tools/batch_run.py`. This skill is a deployment guide
with optional helpers; it is not a manipulation skill, model prompt, or source
of task-specific action plans. Nothing in `src/` automatically loads it.

Use the copy inside the selected project checkout so the helpers resolve that
checkout's code, task configurations, and assets. A coding agent can read this
file by path; a human can follow the same instructions without an agent.

Read [the deployment reference](references/deployment.md) for configuration,
setup, SSH commands, and optional local-model forwarding. Use the supplied
[sample configuration](assets/fleet.example.json) as a template, never as a real
host. Keep account credentials and host-specific paths in ignored local files.

For an authorized deployment:

1. Identify the user's target host, project directory, installed Isaac/cuRobo
   runtimes, and where model access will happen. Reuse existing configuration
   when available. Do not assume a team cluster, username, drive, or GPU count.
2. Check running jobs before transferring files. The helper's `sync` checks
   processes using the destination checkout; also inspect jobs outside that
   checkout if they share the selected GPU. Deploy into an idle checkout or a
   separate explicitly selected destination. Do not stop experiments to sync.
3. Transfer the runnable project and assets using the supplied exclusions. Never
   copy private account sessions, local configuration, environments, or `runs/`.
   Check hashes on the destination and run the frozen task-contract check. Keep
   original and `_variant01` task definitions unchanged.
4. Configure remote runtime wrappers. Use remote model access where available;
   enable the loopback SSH bridge only if the user needs the local Codex login.
   A status or hash check must not generate a model request.
5. Start experiments only when requested, with explicit task IDs and budgets.
   Report which configuration and output directory were used. Stop only the
   owned job when the user requests it; preserve recordings and original logs.

The helpers use one owned batch per host/check-out. They are not a multi-GPU
scheduler. Use separate explicitly configured workers or an existing scheduler
when concurrency is needed. Avoid silently selecting GPU 0 on a shared GPU host.

`tools/remote/fleet.py` remains a compatibility entrypoint for existing users;
new instructions use this skill's `scripts/fleet.py`. Local maintenance rules in
`AGENTS.local.md`, if present, apply only to that checkout and are not public
requirements for readers.
