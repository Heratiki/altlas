import json
import torch
from pathlib import Path
import logging
import re # Import regex

# Configure basic logging - REMOVED basicConfig call
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Tokenizer:
    """
    Handles encoding text strings into token ID tensors and decoding back to strings,
    based on a predefined vocabulary.
    """
    def __init__(self, vocab_path: str = "memory/vocab.json"):
        """
        Initializes the Tokenizer by loading the vocabulary.

        Args:
            vocab_path (str): Path to the vocabulary JSON file.
        """
        self.vocab_path = Path(__file__).parent / vocab_path
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        self._load_vocab()
        # Sort tokens by length descending to prioritize longer matches (e.g., <= before <)
        self._sorted_tokens = sorted(self.token_to_id.keys(), key=len, reverse=True)
        # Remove special tokens from sorted list used for matching code text
        self._sorted_tokens_no_special = [t for t in self._sorted_tokens if not (t.startswith('<') and t.endswith('>')) and t not in ["INDENT", "DEDENT"]]
        logging.info(f"Prepared {len(self._sorted_tokens_no_special)} non-special tokens for matching.")

    def _load_vocab(self):
        """Loads the vocabulary from the JSON file."""
        try:
            with open(self.vocab_path, 'r') as f:
                self.token_to_id = json.load(f)
            self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
            
            # Verify special tokens are present
            required_tokens = {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}
            if not required_tokens.issubset(self.token_to_id.keys()):
                 raise ValueError(f"Vocabulary missing one or more required special tokens: {required_tokens}")

            self.pad_token_id = self.token_to_id["<PAD>"]
            self.sos_token_id = self.token_to_id["<SOS>"]
            self.eos_token_id = self.token_to_id["<EOS>"]
            self.unk_token_id = self.token_to_id["<UNK>"]
            
            logging.info(f"Vocabulary loaded successfully from {self.vocab_path}. Size: {len(self.token_to_id)}")

        except FileNotFoundError:
            logging.error(f"Vocabulary file not found at {self.vocab_path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from vocabulary file: {self.vocab_path}")
            raise
        except ValueError as e:
             logging.error(f"Vocabulary validation error: {e}")
             raise
        except Exception as e:
            logging.error(f"An unexpected error occurred loading vocabulary: {e}")
            raise
            
    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.token_to_id)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes a string of text into a tensor of token IDs using greedy matching.
        Handles newlines and basic token structures.
        Adds SOS and EOS tokens. Uses UNK for unknown sequences.

        Args:
            text (str): The input string (e.g., a code snippet).

        Returns:
            torch.Tensor: A LongTensor containing the sequence of token IDs.
        """
        encoded_ids = [self.sos_token_id]
        remaining_text = text
        current_pos = 0

        while current_pos < len(text):
            # Handle whitespace - skip multiple spaces/tabs
            if text[current_pos].isspace():
                # If it's a newline, add the newline token
                if text[current_pos] == '\n':
                    encoded_ids.append(self.token_to_id.get("\\n", self.unk_token_id))
                current_pos += 1
                continue # Skip other whitespace for now

            # Greedy matching: Find the longest token in vocab that matches the start of remaining_text
            match_found = False
            for token in self._sorted_tokens_no_special: # Use list without special tokens
                # Need to handle potential regex special chars in tokens if using regex matching later
                # For now, simple startswith check
                if text.startswith(token, current_pos):
                    encoded_ids.append(self.token_to_id[token])
                    current_pos += len(token)
                    match_found = True
                    break # Found the longest match
            
            # If no token from vocab matches, treat as UNK (or handle differently, e.g., char-level)
            if not match_found:
                # Simple fallback: treat the single character as UNK
                # A better approach might involve character-level encoding or BPE later
                encoded_ids.append(self.unk_token_id)
                logging.warning(f"Unknown sequence starting at position {current_pos}: '{text[current_pos]}'. Treating as UNK.")
                current_pos += 1

        encoded_ids.append(self.eos_token_id)
        return torch.LongTensor(encoded_ids)

    def decode(self, tensor: torch.Tensor) -> str:
        """
        Decodes a tensor of token IDs back into a string.
        Filters out PAD, SOS, and EOS tokens. Handles newline tokens.

        Args:
            tensor (torch.Tensor): A tensor containing the sequence of token IDs.

        Returns:
            str: The decoded string.
        """
        if tensor.ndim > 1:
             # If batch dimension exists, decode the first sequence
             if tensor.shape[0] > 1:
                  logging.warning("Decoding a batch tensor, only decoding the first sequence.")
             tensor = tensor[0]
             
        decoded_string = ""
        for token_id in tensor.tolist():
            if token_id in [self.pad_token_id, self.sos_token_id, self.eos_token_id]:
                continue
                
            token = self.id_to_token.get(token_id, "<UNK>")
            
            # Handle special tokens for formatting
            if token == "\\n":
                decoded_string += "\n"
            # FUTURE: Handle INDENT/DEDENT here to add/remove leading spaces
            # elif token == "INDENT":
            #     current_indent += "    " # Or based on config
            #     decoded_string += "\n" + current_indent # Indent usually follows newline
            # elif token == "DEDENT":
            #     current_indent = current_indent[:-4] # Or based on config
            #     # Dedent might not always follow newline, handle context?
            else:
                # Basic joining - needs refinement for spacing around operators/punctuation
                # Add space unless previous char was newline or it's the start
                if decoded_string and not decoded_string.endswith('\n'):
                     decoded_string += " " 
                decoded_string += token

        return decoded_string

# Example usage (optional, for direct script execution testing)
if __name__ == '__main__':
    try:
        tokenizer = Tokenizer()
        print(f"Vocab size: {tokenizer.vocab_size}")
        
        test_code = "print \"hello\""
        encoded = tokenizer.encode(test_code)
        print(f"Encoded '{test_code}': {encoded}")
        
        decoded = tokenizer.decode(encoded)
        print(f"Decoded tensor: '{decoded}'")

        test_unknown = "unknown_token + 5"
        encoded_unk = tokenizer.encode(test_unknown)
        print(f"Encoded '{test_unknown}': {encoded_unk}")
        decoded_unk = tokenizer.decode(encoded_unk)
        print(f"Decoded tensor with UNK: '{decoded_unk}'")
        
        # Test decoding with special tokens mixed in
        test_tensor = torch.LongTensor([tokenizer.sos_token_id, 4, 12, 41, tokenizer.eos_token_id, tokenizer.pad_token_id])
        decoded_special = tokenizer.decode(test_tensor)
        print(f"Decoded tensor {test_tensor}: '{decoded_special}'")

    except Exception as e:
         print(f"Error during Tokenizer example usage: {e}")