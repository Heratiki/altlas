import torch
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_pytorch_device():
    """
    Detects and returns the appropriate PyTorch device (GPU if available, else CPU).

    Returns:
        torch.device: The selected PyTorch device ('cuda' or 'cpu').
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Using CPU.")
    return device

# Example usage (optional, for direct script execution testing)
if __name__ == '__main__':
    selected_device = get_pytorch_device()
    print(f"Selected device: {selected_device}")