#!/usr/bin/env python3
"""
count_params.py  –  Universal parameter counter for PyTorch checkpoints.

Features
--------
1. Works with    • pure state_dict (.pth / .pt)
                 • dict containing "state_dict"
                 • .safetensors files  (optional)
2. Reports total parameter count and an approximate FP32 size.
3. Verbose mode (-v) shows a few tensor keys.
4. If loading a *full* model object fails because the class definition
   is missing, prints a friendly reminder to export state_dict instead.

Usage
-----
python count_params.py model.pth
python count_params.py model.safetensors -v
"""

from pathlib import Path
import argparse
import sys
import torch


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_state_dict(path: Path):
    """
    Try to retrieve a tensor dictionary from the given file.
    Returns (state_dict, from_model_obj: bool).

    Raises RuntimeError if we can't obtain a tensor dict.
    """
    # ---- safetensors ------------------------------------------------------- #
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as e:
            raise RuntimeError("Install `safetensors` to read .safetensors files") from e
        return load_file(str(path)), False

    # ---- regular torch.load ----------------------------------------------- #
    try:
        obj = torch.load(str(path), map_location="cpu")
    except (ModuleNotFoundError, AttributeError) as e:
        # Likely a full model object but class is missing
        raise RuntimeError(
            "Failed to unpickle the checkpoint – model class not found.\n"
            "If this file was created with `torch.save(model, ...)`, "
            "please export a plain state_dict instead:\n\n"
            "    torch.save(model.state_dict(), 'model_state.pth')\n"
        ) from e

    # Pure state_dict
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj, False

    # Dict wrapping a state_dict
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "network"):
            sd = obj.get(k)
            if isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
                return sd, False

    # Full model object (class available)
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict(), True

    raise RuntimeError("No tensor dictionary found in checkpoint.")


def count_params(sd: dict) -> int:
    """Return total number of scalar parameters."""
    return sum(t.numel() for t in sd.values())


def main():
    parser = argparse.ArgumentParser(description="Count parameters in a PyTorch checkpoint.")
    parser.add_argument("ckpt", type=str, help=".pth / .pt / .safetensors file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print first few tensor names")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        sys.exit(f" File not found: {ckpt_path}")

    try:
        state_dict, from_model = load_state_dict(ckpt_path)
    except RuntimeError as err:
        sys.exit(f" {err}")

    total = count_params(state_dict)
    print(f"Loaded: {ckpt_path.name}")
    if from_model:
        print("(extracted from full model object)")
    print(f"Total parameters : {total:,d}")
    # print(f"Approx. FP32 size: {total * 4 / 1e6:.2f} MB")

    if args.verbose:
        print("\nSample tensor keys:")
        for i, (k, v) in enumerate(state_dict.items()):
            print(f"  {k:50s}  shape={tuple(v.shape)}")
            if i == 4:
                break


if __name__ == "__main__":
    main()
