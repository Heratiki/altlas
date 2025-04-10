import torch
import torch.nn as nn
import torch.nn.init as init  # Import init module
import logging  # Import logging
import math  # Needed for positional encoding
from typing import Tuple

# Configure logging - REMOVED basicConfig call
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AltLAS_RNN(nn.Module):
    """
    A simple Recurrent Neural Network (RNN) model using LSTM for AltLAS code generation.
    Takes token IDs as input and outputs logits over the vocabulary for the next token.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        use_attention: bool = False,
        num_attention_heads: int = 4,
        positional_encoding_type: str = "none",  # 'none', 'sinusoidal', 'learned'
        dropout: float = 0.0,
        use_layernorm: bool = False,
        attention_residual: bool = True,
        max_seq_len: int = 512,
    ):
        """
        Initializes the RNN model layers.

        Args:
            vocab_size (int): Total number of unique tokens.
            embedding_dim (int): Dimensionality of token embeddings.
            hidden_dim (int): Dimensionality of LSTM hidden state.
            num_layers (int): Number of stacked LSTM layers.
            use_attention (bool): Enable multi-head self-attention after LSTM.
            num_attention_heads (int): Number of attention heads if attention enabled.
            positional_encoding_type (str): 'none', 'sinusoidal', or 'learned'.
            dropout (float): Dropout probability (0.0 disables dropout).
            use_layernorm (bool): Enable LayerNorm after LSTM and attention.
            attention_residual (bool): Add residual connection over attention output.
            max_seq_len (int): Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_seq_len = max_seq_len

        # Initial user-specified flags (may be overridden)
        self.use_attention = use_attention
        self.use_layernorm = use_layernorm
        self.attention_residual = attention_residual
        self.positional_encoding_type = positional_encoding_type

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Auto-configure features based on heuristics
        self._auto_configure_features()

        # Validate attention head divisibility if attention enabled
        if self.use_attention:
            if self.hidden_dim % self.num_attention_heads != 0:
                raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_attention_heads ({self.num_attention_heads}) when using attention.")
            if self.embedding_dim % self.num_attention_heads != 0:
                raise ValueError(f"embedding_dim ({self.embedding_dim}) must be divisible by num_attention_heads ({self.num_attention_heads}) when using attention.")

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Positional encoding (only if enabled)
        if self.positional_encoding_type == "sinusoidal":
            pe = torch.zeros(max_seq_len, embedding_dim)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('positional_encoding', pe.unsqueeze(0))  # (1, max_seq_len, embedding_dim)
        elif self.positional_encoding_type == "learned":
            self.positional_encoding = nn.Embedding(max_seq_len, embedding_dim)
        else:
            self.positional_encoding = None

        # LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Optional LayerNorm after LSTM
        self.lstm_norm = nn.LayerNorm(hidden_dim) if self.use_layernorm else nn.Identity()

        # Attention
        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_attention_heads, batch_first=True)
            self.attention_norm = nn.LayerNorm(hidden_dim) if self.use_layernorm else nn.Identity()
        else:
            self.attention = None
            self.attention_norm = nn.Identity()

        # Final output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self._initialize_weights()

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

    def _auto_configure_features(self):
        """
        Automatically enable or disable features based on model size and heuristics.
        """
        # Enable attention if hidden_dim is large enough
        if self.hidden_dim >= 128:
            self.use_attention = True
        else:
            self.use_attention = False

        # Enable residuals and layernorm if attention is enabled
        if self.use_attention:
            self.attention_residual = True
            self.use_layernorm = True
        else:
            self.attention_residual = False
            self.use_layernorm = False

        # Enable positional encoding if max_seq_len is reasonably long
        if self.max_seq_len > 16:
            self.positional_encoding_type = "sinusoidal"
        else:
            self.positional_encoding_type = "none"

        # Log summary
        posenc_str = self.positional_encoding_type.capitalize() if self.positional_encoding_type != "none" else "None"
        logging.info(f"[MODEL CONFIG] Attention: {'✅' if self.use_attention else '❌'} | "
                     f"Residuals: {'✅' if self.attention_residual else '❌'} | "
                     f"PosEnc: {posenc_str} | "
                     f"LayerNorm: {'✅' if self.use_layernorm else '❌'}")
    def _initialize_weights(self):
        """Initialize model weights using standard methods."""
        logging.info("Initializing model weights...")
        # Initialize Embedding layer
        init.xavier_uniform_(self.embedding.weight)
        logging.info(f"  Embedding weights initialized (Xavier Uniform). Shape: {self.embedding.weight.shape}")
        logging.info(f"    Mean: {self.embedding.weight.mean():.4f}, Std: {self.embedding.weight.std():.4f}, Min: {self.embedding.weight.min():.4f}, Max: {self.embedding.weight.max():.4f}")

        # Initialize Linear layer
        init.xavier_uniform_(self.fc.weight)
        # Changed from zeros to small random values
        init.uniform_(self.fc.bias, -0.1, 0.1)  # Initialize with small random values
        logging.info(f"  Linear FC weights initialized (Xavier Uniform). Shape: {self.fc.weight.shape}")
        logging.info(f"    Mean: {self.fc.weight.mean():.4f}, Std: {self.fc.weight.std():.4f}, Min: {self.fc.weight.min():.4f}, Max: {self.fc.weight.max():.4f}")
        logging.info(f"  Linear FC bias initialized (Uniform [-0.1, 0.1]). Shape: {self.fc.bias.shape}")
        logging.info(f"    Mean: {self.fc.bias.mean():.4f}, Std: {self.fc.bias.std():.4f}, Min: {self.fc.bias.min():.4f}, Max: {self.fc.bias.max():.4f}")

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
        embedded = self.embedding(input_seq)
        # Add positional encoding if enabled
        if self.positional_encoding_type == "sinusoidal" and hasattr(self, 'positional_encoding'):
            seq_len = embedded.size(1)
            embedded = embedded + self.positional_encoding[:, :seq_len, :]
        elif self.positional_encoding_type == "learned" and hasattr(self, 'positional_encoding'):
            seq_len = embedded.size(1)
            positions = torch.arange(seq_len, device=embedded.device).unsqueeze(0).expand(embedded.size(0), seq_len)
            embedded = embedded + self.positional_encoding(positions)
        # else: no positional encoding

        # 2. Pass embeddings through LSTM
        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        # hidden_state tuple shapes: (num_layers, batch_size, hidden_dim)
        lstm_out, hidden_state = self.lstm(embedded, hidden_state)
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # Optionally apply self-attention on top of LSTM outputs
        if self.use_attention and self.attention is not None:
            attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
            attn_output = self.attention_norm(attn_output)
            attn_output = self.dropout(attn_output)
            if self.attention_residual:
                lstm_out = lstm_out + attn_output  # residual connection
            else:
                lstm_out = attn_output

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