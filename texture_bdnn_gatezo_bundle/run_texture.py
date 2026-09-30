"""Run both texture stages; additional arguments override the defaults."""
import sys
from pathlib import Path
from gate_zo_bdnn.run_texture_comparison_task import main

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    supplied = sys.argv[1:]
    sys.argv = [sys.argv[0],
                "--selection-mode", "gatezo",
                "--growth-protocol", "validation_checkpoint",
                "--seed", "22",
                "--data-root", str(root / "data"),
                "--output-root", str(root / "outputs" / "texture_gatezo_protocols"),
                "--device", "auto", *supplied]
    main()
