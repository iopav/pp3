"""Keep only numeric training metrics; discard measurement diagnostics."""


METADATA_FIELDS = {
    'energy_method', 'energy_reason', 'gpu_uuid',
    'auxiliary_mem_method', 'auxiliary_mem_reason', 'unit', 'source_unit',
}


def compact_metrics(row):
    """Discard descriptive fields without producing a metadata file."""
    return {('hidden_state_mem_bytes (included)' if key == 'hidden_state_mem_bytes' else key): value
            for key, value in row.items() if key not in METADATA_FIELDS}
