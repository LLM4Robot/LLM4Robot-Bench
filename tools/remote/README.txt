Optional remote deployment has moved to skills/r5-remote-deploy/SKILL.md.
Local experiments use tools/reproduce.py or tools/batch_run.py and need no SSH.

This directory retains only compatibility launchers for existing commands:
- fleet.py forwards to the skill helper.
- remote_entry.sh and recovery_planner.sh forward existing runtime/recovery calls.
- fleet.local.json, when present, is private and ignored; it remains supported.

New deployments configure configs/runtime/remote.local.json from the skill's
assets/fleet.example.json. The public skill contains no team host or credentials.
