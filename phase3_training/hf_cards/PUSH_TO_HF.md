# Pushing Model Cards to HuggingFace Hub

The three `.md` files in this folder are ready-to-use model cards. Copy each one to the
corresponding HuggingFace repo as `README.md`.

## Method 1 — huggingface_hub CLI (recommended)

```bash
pip install huggingface_hub
huggingface-cli login   # enter your HF token (write access)

# pivotai-ft
huggingface-cli upload ishreyadev/pivotai-ft-gguf \
  phase3_training/hf_cards/pivotai_ft.md README.md

# pivotai-distill
huggingface-cli upload ishreyadev/pivotai-distill-gguf \
  phase3_training/hf_cards/pivotai_distill.md README.md

# pivotai-curriculum
huggingface-cli upload ishreyadev/pivotai-curriculum-gguf \
  phase3_training/hf_cards/pivotai_curriculum.md README.md
```

## Method 2 — HuggingFace web UI

1. Go to each model repo on huggingface.co
2. Click **Files** → **README.md** → **Edit**
3. Paste the contents of the corresponding `.md` file
4. Click **Commit changes**
