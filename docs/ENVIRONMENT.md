# Reproducible environment notes

This public note intentionally omits host-specific filesystem paths, full GPU
inventory, driver metadata, and resource-capacity details. The benchmark
protocol and exact commands are documented in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Experimental target

- One NVIDIA RTX A5000 (24 GB) for the P1 experiments.
- Development model: `Qwen/Qwen2.5-3B-Instruct`.
- Main validation model: `Qwen/Qwen2.5-7B-Instruct`.
- CUDA device selection is explicit in each benchmark command.
- PEEK is pinned as the submodule revision recorded in `.gitmodules` and the
  parent tree.

## Local isolation

The original experiments used a project-local Python environment and local
model/tokenizer caches. These are intentionally ignored by Git and are not
part of this repository.

## Reproduction checklist

1. Verify Python, PyTorch, CUDA, and vLLM compatibility for the selected model.
2. Download model weights through the normal Hugging Face workflow.
3. Run the patch application script and CPU tests before GPU benchmarks.
4. Record the effective KV-cache capacity and device configuration with each
   result.
