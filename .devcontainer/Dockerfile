FROM python:3.11-slim

# System packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --upgrade pip && \
    pip install onnx onnxruntime numpy rich

# Create a working directory
WORKDIR /workspace

# Create the vscode user with sudo access
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user with explicit shell
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chown -R $USERNAME:$USERNAME /workspace

# Set the default user
USER $USERNAME

# Verify the user exists - helps with debugging
RUN id

CMD ["bash"]
