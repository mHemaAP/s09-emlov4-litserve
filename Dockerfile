# FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace
COPY . .
RUN pip install uv
RUN uv pip install -r pyproject.toml --system

# CMD ["python", "src/train.py"]
CMD ["tail", "-f", "/dev/null"]

# uv sync --extra-index-url https://download.pytorch.org/whl/cpu

