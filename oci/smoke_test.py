#!/usr/bin/env python3
"""End-to-end inference against the SubCell OCI container."""

import json
import os

os.environ.setdefault("NAHUAL_IPC_TIMEOUT_MS", "900000")

import numpy as np
from nahual.process import dispatch_setup_process


def main() -> None:
    address = os.environ.get("NAHUAL_ADDRESS", "tcp://127.0.0.1:5555")
    setup, process = dispatch_setup_process("subcell")
    info = setup(
        {
            "model_type": "mae_contrast_supcon_model",
            "model_channels": "rybg",
            "device": "cpu",
        },
        address=address,
    )
    pixels = np.random.default_rng(42).random((1, 4, 1, 224, 224), dtype=np.float32)
    result = process(pixels, address=address)
    assert result.shape == (1, 1536), result.shape
    assert np.isfinite(result).all()
    print(json.dumps({"setup": info, "shape": list(result.shape)}))


if __name__ == "__main__":
    main()
