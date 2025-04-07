import torch
import torch.nn as nn
from typing import Tuple

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

        # 2. Pass embeddings through LSTM
        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        # hidden_state tuple shapes: (num_layers, batch_size, hidden_dim)
        lstm_out, hidden_state = self.lstm(embedded, hidden_state)

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