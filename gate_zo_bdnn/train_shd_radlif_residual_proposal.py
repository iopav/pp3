"""Run GateZO with optional residual-adaptive frozen RadLIF proposals."""

try:
    from . import shd_radlif_residual_proposal_core as proposal_core
    from . import train_shd_individual_pool_structural_zo as trainer
except ImportError:
    import shd_radlif_residual_proposal_core as proposal_core
    import train_shd_individual_pool_structural_zo as trainer


def main():
    trainer.core = proposal_core
    trainer.propose_individual_pool = proposal_core.propose_individual_pool
    trainer.load_state = proposal_core.load_state
    trainer.main()


if __name__ == "__main__":
    main()
