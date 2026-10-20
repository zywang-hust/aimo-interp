# Codabench evaluation image

Checked 2026-10-18 from the registry, read-only: layers were listed and read with `tarfile`, never executed.

## Which image

`GET https://www.codabench.org/api/competitions/16180/` gives the `docker_image` for both ingestion and scoring
(the phases set none of their own):

```
ghcr.io/aimo-interp/aimo-codabench-runtime:torch2.12.1-cuda13.2-cudnn9-transformers5.13.0-sklearn1.8.0-kernels0.15.2
```

linux/amd64 manifest `sha256:a943de556a2288a41ae568a86c04438968f892216ca6202ab0dcb98614d3d8b3`, built 2026-08-31.
The public `ext/getting-started/Dockerfile.competition` is out of date. The image's last build step is:

```
RUN python -m pip install --no-cache-dir --break-system-packages "accelerate==1.13.0" "huggingface-hub==1.22.0"
    "joblib==1.5.3" "kernels==0.15.2" "pandas==3.0.3" "safetensors==0.8.0" "scikit-learn==1.8.0"
    "tokenizers==0.22.2" "transformers==5.13.0" && python -c "... assert is_kernels_available() ..."
ENV HF_HUB_OFFLINE=1 HF_HOME=/app/data/huggingface TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

Packages (`/usr/local/lib/python3.12/dist-packages`): torch 2.12.1+cu132, triton 3.7.1, numpy 2.4.6, transformers 5.13.0,
kernels 0.15.2 (+ kernels-data 0.16.1, tomlkit 0.15.1, PyYAML 6.0.3), huggingface_hub 1.22.0, accelerate 1.13.0,
safetensors 0.8.0, tokenizers 0.22.2, pandas 3.0.3, scikit-learn 1.8.0, scipy 1.18.1, joblib 1.5.3.
The image has no `/app` content and no HF cache; models and kernels come from the `/app/data/huggingface` mount.
Our replica (`engine/.venv-eval` + `engine/evalrt/kernels-overlay`) has the same versions.

## Native MXFP4 for gpt-oss-120b

transformers 5.13 (`quantizers/quantizer_mxfp4.py`, `integrations/hub_kernels.py`) loads the checkpoint natively only if
compute capability ≥ 7.5, triton ≥ 3.4 and `0.15.2 ≤ kernels < 0.16`; otherwise it warns and dequantizes to bf16
(about 230 GB, hopeless on a 96 GB card). It then calls `get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)`,
which has no fallback: offline, kernels 0.15.2 needs `refs/v1` and the full snapshot it points to in the HF hub cache.

Settled on Codabench: every submission since 1000361 loads gpt-oss-120b natively in MXFP4 on the eval GPU (60.8 GiB,
about 4.5 s) and samples on sm_120 (`../NOTES.md` §6). The GPU is shared and free VRAM may be below 96 GB; weights plus
activations peaked at 69.4 GiB in our 32×300 forward test.
