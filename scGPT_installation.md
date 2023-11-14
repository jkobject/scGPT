to install, use a fresh config and please run:

```
pip install packaging torch torchvision torchaudio && pip install "flash-attn<1.0.5" --no-build-isolation && pip install ipykernel pandas scanpy scvi-tools numba --upgrade "numpy<1.24" torchtext scib datasets==2.14.5 transformers==4.33.2 wandb cell-gears torch_geometric && pip install --no-deps scgpt
```

add `--index-url https://download.pytorch.org/whl/cu118` to the first pip install command if running something like cuda/11.8
