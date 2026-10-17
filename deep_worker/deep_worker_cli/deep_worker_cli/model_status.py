"""Select model observations belonging to the current agent launch."""

from datetime import datetime

from deep_worker_cli.model_predicate import model_can_belong_to_provider


def current_record_model(
    record: object, provider: str, started_at: datetime | None
) -> str | None:
    """Reject statistics from another provider, launch, or TaskManager.

    Returns:
        The observed model, or None when the record is not for this launch.
    """
    if not isinstance(record, dict) or started_at is None:
        return None
    if record.get("provider") != provider or record.get("kind") != "agent":
        return None
    stamp = record.get("ts")
    if not isinstance(stamp, str):
        return None
    try:
        observed = datetime.fromisoformat(stamp)
        if observed.astimezone() < started_at.astimezone():
            return None
    except (ValueError, TypeError, OverflowError):
        return None
    model = record.get("model")
    if not isinstance(model, str) or not model_can_belong_to_provider(provider, model):
        return None
    return model.strip()
