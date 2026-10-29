import os
import logging

import torch

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_pytorch_device():
    """
    Return the PyTorch device to use. CPU is the default everywhere.

    The model is small (about 683k parameters), so CPU is fine. A GPU is
    opt-in through the environment variable ALTLAS_DEVICE (case-insensitive):

    - unset, ``cpu`` or any other value: CPU. An unrecognised non-empty value
      also logs a warning.
    - ``cuda`` or ``auto``: try the GPU. If torch reports no GPU, or a small
      test kernel fails on it, log a warning and fall back to CPU.

    ROCm note: HSA_OVERRIDE_GFX_VERSION=10.3.0 is a ROCm override that makes a
    build without native kernels try gfx1030 code. It is unreliable on
    gfx1010 (Navi 10, for example the RX 5700 XT); without it the first kernel
    fails with "HIP error: invalid device function".

    On a gfx101x GPU, ``torch.backends.cudnn.enabled`` is set to False (MIOpen
    breaks LSTM training there) unless ALTLAS_MIOPEN=1.

    Returns:
        torch.device: The selected PyTorch device ('cuda' or 'cpu').
    """
    requested = os.environ.get("ALTLAS_DEVICE", "").strip().lower()

    if requested not in ("cuda", "auto"):
        if requested and requested != "cpu":
            logging.warning(
                "Unrecognised ALTLAS_DEVICE=%r (use cpu, cuda or auto). Using CPU.", requested
            )
        else:
            logging.info("Using CPU (set ALTLAS_DEVICE=cuda or auto to try the GPU).")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        logging.info("ALTLAS_DEVICE=%s but no GPU is available. Using CPU.", requested)
        return torch.device("cpu")

    device = torch.device("cuda")
    try:
        torch.ones(8, device=device).mean().item()
        torch.cuda.synchronize()
    except Exception as e:
        logging.warning(
            "GPU kernel test failed (%s: %s). Falling back to CPU.", type(e).__name__, e
        )
        return torch.device("cpu")

    logging.info("GPU kernel test passed. Using %s.", device)

    # gfx101x (Navi 10): MIOpen fails on RNN backward (runtime CK kernel error),
    # so turn it off. ALTLAS_MIOPEN=1 keeps it on, for testing newer stacks.
    # gcnArchName exists only on ROCm builds; do nothing without it.
    try:
        arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    except Exception:
        arch = ""
    if arch.startswith("gfx101") and os.environ.get("ALTLAS_MIOPEN", "").strip() != "1":
        torch.backends.cudnn.enabled = False
        logging.info("MIOpen is off for RNN training on gfx101x (set ALTLAS_MIOPEN=1 to keep it).")
    return device


# Example usage (optional, for direct script execution testing)
if __name__ == '__main__':
    selected_device = get_pytorch_device()
    print(f"Selected device: {selected_device}")
