"""Prevent reference runners from executing an unimplemented study contract."""
def require_ready(config):
    if config.get("execution_ready") is not True:
        raise RuntimeError("Study execution is not enabled by this frozen config.")

def require_training_ready(config,stage):
    if stage=="smoke":return
    if stage in ("development","pilot") and config.get("development_training_ready") is True:return
    if stage=="full" and config.get("full_training_ready") is True:return
    raise RuntimeError("Requested training stage is not ready or enabled by this frozen config.")
