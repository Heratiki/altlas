FROM python:3.13-slim

# System packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# uv, copied from the official image
COPY --from=ghcr.io/astral-sh/uv:0.6.4 /uv /uvx /usr/local/bin/

# Create a working directory
WORKDIR /workspace

# Python dependencies from the lock (CPU torch from the PyTorch CPU index)
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock .python-version /workspace/
RUN uv sync --frozen && chmod -R a+rX /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

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
