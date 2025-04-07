# AltLAS PyTorch Transition Plan

This document outlines the steps to transition the AltLAS agent from its current simple weight-based learning mechanism to a more powerful core based on PyTorch and Reinforcement Learning (RL).

**Goal:** Replace the simple generator and learning logic with a randomly initialized ("stupid") PyTorch neural network trained via RL, enabling GPU acceleration and paving the way for learning more complex behaviors.

**Note on vLLM:** vLLM is for optimizing inference of large, pre-trained models. It is not suitable for the *training* phase of AltLAS, which starts from scratch. Standard PyTorch with CUDA support is the correct approach for training.

---

## Phase 1: Setup and Prerequisites

1.  **Install PyTorch with CUDA Support:**
    *   **Action:** Modify `.devcontainer/Dockerfile` to install `torch` with appropriate CUDA support (check official PyTorch website for current commands based on your environment/CUDA version).
    *   **Action:** Rebuild the dev container.

2.  **Device Handling:**
    *   **Action:** Add logic (e.g., in `runner.py` or `utils.py`) to detect GPU availability (`torch.cuda.is_available()`) and set the target PyTorch device (`cuda` or `cpu`).
    *   **Action:** Ensure this `device` variable is used consistently when creating tensors and moving the model.

## Phase 2: Data Representation

3.  **Vocabulary and Tokenization:**
    *   **Action:** Define a fixed vocabulary mapping all code elements (`all_elements`) plus special tokens (`<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`) to unique integer IDs. Store persistently (e.g., `memory/vocab.json`).
    *   **Action:** Create a `tokenizer.py` utility with `encode()` (string -> tensor of IDs) and `decode()` (tensor of IDs -> string) functions.
    *   **Impact:** Core components will primarily operate on sequences of token IDs.

## Phase 3: Model Definition

4.  **Define PyTorch Model (`nn.Module`):**
    *   **Action:** Create `agent_core/model.py`.
    *   **Action:** Define a class (e.g., `AltLAS_RNN`) inheriting from `torch.nn.Module`.
    *   **Initial Architecture:**
        *   `nn.Embedding(vocab_size, embedding_dim)`
        *   `nn.LSTM` or `nn.GRU(embedding_dim, hidden_dim, ...)`
        *   `nn.Linear(hidden_dim, vocab_size)`
    *   **Initialization:** Ensure random weight initialization (PyTorch default).
    *   **`forward` method:** Define the data flow through layers.

## Phase 4: Core Logic Replacement

5.  **Update Code Generation:**
    *   **Action:** Replace the generation logic in `CodeGenerator`.
    *   **New Process:**
        *   Initialize RNN hidden state.
        *   Start with `<SOS>` token ID.
        *   Loop: Feed current sequence -> Get model output logits -> Apply `softmax` -> Sample next token ID (`torch.multinomial`) -> Store token ID and its log-probability -> Check for `<EOS>` or max length.
        *   Decode final ID sequence back to code string.
        *   Return code string *and* the sequence of log-probabilities.

6.  **Implement RL Training Step:**
    *   **Action:** Replace `CodeGenerator._update_weights` with a new `learn(reward, log_probs)` method.
    *   **RL Algorithm (Start with REINFORCE):**
        *   Calculate return (e.g., the final `score`).
        *   Calculate policy loss (e.g., `- sum(log_prob * return)`).
        *   Instantiate `torch.optim.Adam` (or similar) in `CodeGenerator.__init__`.
        *   Perform update: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
    *   **Integration:** Call `generator.learn(...)` from `runner.py` after scoring.

## Phase 5: Integration and Persistence

7.  **Integrate Components:**
    *   **Action:** Modify `CodeGenerator.__init__` to load vocab, instantiate tokenizer, instantiate the PyTorch model (and move to `device`), instantiate the optimizer, and load saved states.
    *   **Action:** Ensure all tensors are consistently moved to the target `device`.

8.  **Update Persistence:**
    *   **Action:** Modify weight/state saving/loading functions.
    *   **Action:** Use `torch.save(model.state_dict(), ...)` and `model.load_state_dict(torch.load(...))` for the model.
    *   **Action:** Similarly save/load `optimizer.state_dict()`.

## Phase 6: Configuration and Refinement

9.  **Update `config.ini`:**
    *   **Action:** Add sections/keys for model hyperparameters (`embedding_dim`, `hidden_dim`), optimizer settings (`learning_rate`), and paths for vocab/model/optimizer states.

10. **Testing and Debugging:**
    *   **Action:** Test thoroughly, starting with simple tasks.
    *   **Action:** Verify device placement, loss trends, and overall loop functionality.

---

This plan provides a roadmap for evolving AltLAS towards a more capable, emergent learning system using standard deep learning tools.