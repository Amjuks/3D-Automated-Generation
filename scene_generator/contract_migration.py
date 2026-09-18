"""Explicit metadata migration. No legacy plan-to-design conversion is defined."""

from .models import Component


def migrate_component(data, target_version=2):
    component = Component.model_validate(data)
    if target_version not in {1, 2} or target_version < component.contract_version:
        raise ValueError("unsupported component migration")
    if component.contract_version == target_version:
        return component
    original_version = component.contract_version
    component.contract_version = 2
    component.local_transform.contract_version = 2
    component.world_transform.contract_version = 2
    component.provenance = {
        **component.provenance,
        "source": "migration",
        "from_version": original_version,
        "reason": "explicit shared component metadata upgrade; geometry and identity unchanged",
    }
    return Component.model_validate(component.model_dump())
