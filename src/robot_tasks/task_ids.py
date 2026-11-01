"""Public names; legacy identities remain stable in frozen experiment records."""


def internal_task_id(task_id: str) -> str:
    return task_id.removesuffix("_variant01") + "_v3" if task_id.endswith("_variant01") else task_id


def public_task_id(task_id: str) -> str:
    return task_id.removesuffix("_v3") + "_variant01" if task_id.endswith("_v3") else task_id
