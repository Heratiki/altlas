import torch
import torch.nn as nn
import torch.nn.init as init # Import init module
import logging # Import logging
from typing import Tuple

# Configure logging - REMOVED basicConfig call
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AltLAS_RNN(nn.Module):
    """
    A simple Recurrent Neural Network (RNN) model using LSTM for AltLAS code generation.
    Takes token IDs as input and outputs logits over the vocabulary for the next token.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        """
        Initializes the RNN model layers.

        Args:
            vocab_size (int): The total number of unique tokens in the vocabulary.
            embedding_dim (int): The dimensionality of the token embeddings.
            hidden_dim (int): The dimensionality of the LSTM hidden state.
            num_layers (int): Number of LSTM layers.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Optional: Enable attention mechanism (default False for stability-first)\n        self.use_attention = True  # Set True to enable attention, or make configurable\n\n        # Positional encoding buffer (fixed sinusoidal)\n        max_seq_len = 512  # or configurable\n        pe = torch.zeros(max_seq_len, self.embedding_dim)\n        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)\n        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-math.log(10000.0) / self.embedding_dim))\n        pe[:, 0::2] = torch.sin(position * div_term)\n        pe[:, 1::2] = torch.cos(position * div_term)\n        self.register_buffer('positional_encoding', pe.unsqueeze(0))  # shape (1, max_seq_len, embedding_dim)\n\n        # Multi-head self-attention layer (query=key=value=LSTM outputs)\n        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)\n
        self.num_layers = num_layers

        # Layer 1: Embedding layer
        # Maps input token IDs to dense vectors.
        # padding_idx=0 assumes PAD token has ID 0, prevents it from affecting gradients.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Layer 2: LSTM layer
        # Processes the sequence of embeddings.
        # batch_first=True means input/output tensors have shape (batch, seq, feature).
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Layer 3: Linear layer (Output layer)
        # Maps LSTM hidden states to logits over the vocabulary.
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights explicitly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using standard methods."""
        logging.info("Initializing model weights...")
        # Initialize Embedding layer
        init.xavier_uniform_(self.embedding.weight)
        logging.info(f"  Embedding weights initialized (Xavier Uniform). Shape: {self.embedding.weight.shape}")
        logging.info(f"    Mean: {self.embedding.weight.mean():.4f}, Std: {self.embedding.weight.std():.4f}, Min: {self.embedding.weight.min():.4f}, Max: {self.embedding.weight.max():.4f}")

        # Initialize Linear layer
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
        logging.info(f"  Linear FC weights initialized (Xavier Uniform). Shape: {self.fc.weight.shape}")
        logging.info(f"    Mean: {self.fc.weight.mean():.4f}, Std: {self.fc.weight.std():.4f}, Min: {self.fc.weight.min():.4f}, Max: {self.fc.weight.max():.4f}")
        logging.info(f"  Linear FC bias initialized (Zeros). Shape: {self.fc.bias.shape}")

        # LSTM layers are often initialized reasonably by default, 
        # but we can initialize their linear components if needed.
        # Example for LSTM weights (more complex due to gates):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
                logging.info(f"  LSTM {name} initialized (Xavier Uniform). Shape: {param.shape}")
                logging.info(f"    Mean: {param.data.mean():.4f}, Std: {param.data.std():.4f}, Min: {param.data.min():.4f}, Max: {param.data.max():.4f}")
            elif 'weight_hh' in name:
                init.orthogonal_(param.data) # Orthogonal is common for recurrent weights
                logging.info(f"  LSTM {name} initialized (Orthogonal). Shape: {param.shape}")
                logging.info(f"    Mean: {param.data.mean():.4f}, Std: {param.data.std():.4f}, Min: {param.data.min():.4f}, Max: {param.data.max():.4f}")
            elif 'bias' in name:
                param.data.fill_(0)
                # Setting forget gate bias to 1 can sometimes help learning
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
                logging.info(f"  LSTM {name} initialized (Zeros, forget gate bias=1). Shape: {param.shape}")
        logging.info("Weight initialization complete.")

    # Add reinitialization method for weights
    def reinitialize_weights(self):
        """Reinitialize model weights to avoid pathological states."""
        self._initialize_weights()
        logging.info("Model weights reinitialized.")

    def forward(self, input_seq: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Defines the forward pass of the model.

        Args:
            input_seq (torch.Tensor): Tensor of input token IDs with shape (batch_size, seq_length).
            hidden_state (Tuple[torch.Tensor, torch.Tensor], optional): 
                Initial hidden state for the LSTM (h_0, c_0). 
                If None, it defaults to zeros. Shape for each tensor: (num_layers, batch_size, hidden_dim).

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                - output_logits (torch.Tensor): Logits over the vocabulary for each token in the sequence.
                  Shape: (batch_size, seq_length, vocab_size).
                - hidden_state (Tuple[torch.Tensor, torch.Tensor]): The final hidden state of the LSTM (h_n, c_n).
                  Shape for each tensor: (num_layers, batch_size, hidden_dim).
        """
        # 1. Get embeddings
        # input_seq shape: (batch_size, seq_length)
        # embedded shape: (batch_size, seq_length, embedding_dim)
        # Add positional encoding to embeddings (truncate if needed)\n        seq_len = embedded.size(1)\n        embedded = embedded + self.positional_encoding[:, :seq_len, :]\n
        embedded = self.embedding(input_seq)

        # 2. Pass embeddings through LSTM
        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        # hidden_state tuple shapes: (num_layers, batch_size, hidden_dim)
        lstm_out, hidden_state = self.lstm(embedded, hidden_state)
        # Optionally apply self-attention on top of LSTM outputs\n        if self.use_attention:\n            # MultiheadAttention expects (batch, seq, feature) with batch_first=True\n            attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)\n            lstm_out = attn_output\n

        # 3. Pass LSTM output through the final linear layer
        # output_logits shape: (batch_size, seq_length, vocab_size)
        output_logits = self.fc(lstm_out)

        return output_logits, hidden_state

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the hidden state (h_0, c_0) for the LSTM layer with zeros.

        Args:
            batch_size (int): The batch size for the input sequence.
            device (torch.device): The device (CPU or CUDA) to create the tensors on.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The initialized hidden state (h_0, c_0).
        """
        # The shape is (num_layers, batch_size, hidden_dim)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)

# Example usage (optional, for direct script execution testing)
if __name__ == '__main__':
    # Example parameters (replace with actual config values later)
    VOCAB_SIZE = 50 
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    BATCH_SIZE = 4
    SEQ_LENGTH = 10
    DEVICE = torch.device("cpu") # Or torch.device("cuda") if available

    # Instantiate model
    model = AltLAS_RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    print("Model instantiated:")
    print(model)
    
    # Create dummy input data
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH), device=DEVICE)
    print(f"\nDummy input shape: {dummy_input.shape}")

    # Initialize hidden state
    initial_hidden = model.init_hidden(BATCH_SIZE, DEVICE)
    print(f"Initial hidden state shape (h_0): {initial_hidden[0].shape}")
    print(f"Initial cell state shape (c_0): {initial_hidden[1].shape}")

    # Perform forward pass
    try:
        logits, final_hidden = model(dummy_input, initial_hidden)
        print(f"\nOutput logits shape: {logits.shape}") # Expected: (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
        print(f"Final hidden state shape (h_n): {final_hidden[0].shape}")
        print(f"Final cell state shape (c_n): {final_hidden[1].shape}")
        
        # Test forward pass without initial hidden state (should default to zeros)
        logits_no_init, _ = model(dummy_input)
        print(f"\nOutput logits shape (no initial hidden): {logits_no_init.shape}")

    except Exception as e:
        print(f"\nError during forward pass test: {e}")