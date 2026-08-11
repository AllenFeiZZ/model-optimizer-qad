# Qwen2.5-Math-7B post-RL NVFP4 QAD

This directory contains the experiment-specific files used to run full-vocabulary
forward-KL quantization-aware distillation (QAD) after RL. It intentionally lives
with the ModelOpt source instead of the Slime launcher tree because QAD is executed
by ModelOpt's `examples/llm_qat/train.py`.

## Layout

```text
configs/dataset/       Dataset blend definitions
configs/train/         Smoke, gate, large, and medium training profiles
configs/quantization/  NVFP4 quantization recipe
tools/                 Sequence generation, merge validation, and KL comparison
run_qad.sh             Quantize-if-needed and launch QAD with FSDP2
resolve_env_yaml.py    Resolve dataset paths without hard-coding them in YAML
```

The AIME Mean@32 evaluation launcher remains in the Slime repository because it
uses the Slime/SGLang inference and evaluation stack; it is not a ModelOpt QAD
dependency.

## Source and tested environment

- ModelOpt source commit: `3d4d9249f4a3333f782e24fb9a830ca7a0dc5d5d`
- Python: `3.12.3`
- PyTorch: `2.11.0+cu130`
- Transformers: `5.12.1`
- Accelerate: `1.13.0`
- Transformer Engine: `2.17.0`
- Flash Attention: `2.7.3`
- Datasets: `5.0.0`

Use the full ModelOpt source checkout. `train.py` is not a standalone script: it
imports sibling helpers and the repository's `modelopt` package.

Install the example and experiment helper dependencies from the ModelOpt checkout:

```bash
pip install -r examples/llm_qat/experiments/qwen2_5_math_7b_nvfp4_qad/requirements.txt
```

## Run

The default profile is the 17,078-sequence, two-epoch run:

```bash
bash examples/llm_qat/experiments/qwen2_5_math_7b_nvfp4_qad/run_qad.sh
```

Available profiles:

```bash
QAD_PROFILE=smoke bash examples/llm_qat/experiments/qwen2_5_math_7b_nvfp4_qad/run_qad.sh
QAD_PROFILE=gate50_lr1e6 bash examples/llm_qat/experiments/qwen2_5_math_7b_nvfp4_qad/run_qad.sh
QAD_PROFILE=large4096_lr1e6 bash examples/llm_qat/experiments/qwen2_5_math_7b_nvfp4_qad/run_qad.sh
QAD_PROFILE=medium17078_2ep_lr5e6 bash examples/llm_qat/experiments/qwen2_5_math_7b_nvfp4_qad/run_qad.sh
```

Paths are configurable without editing tracked YAML files:

```bash
QAD_VENV=/path/to/venv \
QAD_TEACHER_MODEL=/path/to/bf16_teacher \
QAD_FAKEQUANT_MODEL=/path/to/fakequant_student \
QAD_DATA_FILE=/path/to/generated_sequences.jsonl \
QAD_OUTPUT_DIR=/path/to/output \
QAD_NUM_PROCESSES=8 \
bash examples/llm_qat/experiments/qwen2_5_math_7b_nvfp4_qad/run_qad.sh
```

Validate all required paths and resolved configuration without starting GPUs:

```bash
QAD_DRY_RUN=1 bash examples/llm_qat/experiments/qwen2_5_math_7b_nvfp4_qad/run_qad.sh
```

The launcher resolves `${QAD_...}` placeholders into temporary dataset and training
YAML files, then removes both files when the run exits. The generated JSONL, model
weights, checkpoints, dataset cache, TensorBoard logs, and evaluation outputs are
external artifacts and must not be committed to Git.

## Data preparation and checks

- `tools/qad_generate_sequences.py` calls an SGLang-compatible `/generate`
  endpoint to create BF16-teacher responses.
- `tools/qad_merge_validate.py` merges generation shards and rejects duplicate
  prompts or overlap with the fixed holdout.
- `tools/qad_compare_kl.py` compares pre/post-QAD students against the same BF16
  teacher on fixed sequences using full-vocabulary forward KL and top-1 agreement.

The training JSONL and fixed holdout must stay disjoint. Keep their shared-storage
locations in the experiment report or job configuration rather than embedding them
in this repository.
