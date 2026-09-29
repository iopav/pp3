import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds.json"


def test_matched_configs_differ_only_in_identity_and_gatezo_switch():
    configs = json.loads(CONFIG.read_text())
    assert len(configs) == 2
    assert [row["selection_mode"] for row in configs] == ["gatezo", "fixed16"]
    assert all(row["proposal_mode"] == "prior" for row in configs)
    assert all(row["trainer_script"] == "train_shd_radlif_residual_proposal.py" for row in configs)
    stripped = []
    for row in configs:
        stripped.append({key: value for key, value in row.items()
                         if key not in {"config_id", "selection_mode"}})
    assert stripped[0] == stripped[1]
    expected = {
        "candidate_pool_size": 512,
        "growth_step": 16,
        "train_channel_jitter_std": 20.0,
        "train_jitter_copies": 1,
        "train_time_scale_min": 0.90,
        "train_time_scale_max": 1.10,
        "heterogeneous_tau_fraction": 0.5,
        "input_branches": 4,
        "input_delay_ms": [0.0, 10.0, 25.0, 50.0, 100.0, 200.0],
        "frequency_bandwidths": [32, 64, 128, 256],
    }
    for key, value in expected.items():
        assert configs[0][key] == value


def test_submission_grid_and_documented_artifacts_are_present():
    script = (ROOT / "run_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds_gpu.sbatch").read_text()
    assert "#SBATCH --array=0-9%8" in script
    assert "--expected-configs 2" in script
    assert "--seeds 57 58 59 60 61" in script
    assert "shd_radlif_r0_gatezo_vs_no_gatezo_5seeds" in script
    required = (
        "run_shd_adlif_gatezo_task.py", "shd_structural_zo_search_space.py",
        "train_shd_radlif_residual_proposal.py", "shd_radlif_residual_proposal_core.py",
        "shd_radlif_advanced_gatezo_core.py", "shd_radlif_delay_gatezo_core.py",
        "shd_adlif_gatezo_core.py", "train_shd_individual_pool_structural_zo.py",
        "shd_gatezo_support.py", "train_shd_bdnn_rsnn_blocks.py",
        "train_shd_bdnn_rsnn_pure.py", "summarize_shd_radlif_r0_gatezo_vs_no_gatezo.py",
    )
    for name in required:
        assert (ROOT / name).is_file(), name
    runbook = (ROOT / "SHD_R0_GATEZO_VS_NO_GATEZO_RUNBOOK_CN.md").read_text(encoding="utf-8")
    for required_text in (
        "R0_fixed_prior_gatezo", "R0_fixed_prior_no_gatezo_fixed16",
        "candidate_pool_size: 512", "array: 0-9%8", "Task ID",
        "latest_structural_zo.pt", "final_structural_zo.pt",
        "summarize_shd_radlif_r0_gatezo_vs_no_gatezo.py",
        "bdnn_arm64.sif", "arm64", "test_used_for_fitting_or_selection",
        "build_shd_radlif_r0_bundle_with_container.sh",
    ):
        assert required_text in runbook
    builder = (ROOT / "build_shd_radlif_r0_bundle_with_container.sh").read_text(encoding="utf-8")
    assert "bdnn_arm64.sif" in builder
    assert "refusing to overwrite" in builder
    assert "tar -cf" in builder
    assert "sha256sum" in builder
