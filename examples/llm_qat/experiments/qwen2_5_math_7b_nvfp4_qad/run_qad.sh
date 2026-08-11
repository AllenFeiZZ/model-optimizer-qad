#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
MODELOPT_ROOT=$(cd -- "${EXAMPLE_DIR}/../.." && pwd)

QAD_VENV=${QAD_VENV:-/codebase/qad-venv}
QAD_PROFILE=${QAD_PROFILE:-medium17078_2ep_lr5e6}
QAD_NUM_PROCESSES=${QAD_NUM_PROCESSES:-8}

QAD_TEACHER_MODEL=${QAD_TEACHER_MODEL:-/mnt/tidal-alsh01/dataset/pai/muming/nvfp4_rl/exports/Qwen2.5-Math-7B_nvfp4_cont200_step199_hf}
QAD_FAKEQUANT_MODEL=${QAD_FAKEQUANT_MODEL:-/codebase/bf16_slime/qad/qwen2.5_math_7b_nvfp4_step199_fakequant}

case "${QAD_PROFILE}" in
  smoke)
    DEFAULT_DATA_CONFIG=qad_dapo_gen256.yaml
    DEFAULT_DATA_FILE=/codebase/bf16_slime/qad/qwen2.5_math_7b_nvfp4_step199_bf16_gen256_seed42.jsonl
    DEFAULT_OUTPUT_DIR=/codebase/bf16_slime/qad/qwen2.5_math_7b_nvfp4_step199_qad_smoke10
    ;;
  gate50_lr1e6)
    DEFAULT_DATA_CONFIG=qad_dapo_gen256.yaml
    DEFAULT_DATA_FILE=/codebase/bf16_slime/qad/qwen2.5_math_7b_nvfp4_step199_bf16_gen256_seed42.jsonl
    DEFAULT_OUTPUT_DIR=/codebase/bf16_slime/qad/qwen2.5_math_7b_nvfp4_step199_qad_gate50_lr1e6
    ;;
  large4096_lr1e6)
    DEFAULT_DATA_CONFIG=qad_dapo_gen4096.yaml
    DEFAULT_DATA_FILE=/codebase/bf16_slime/qad/qwen2.5_math_7b_nvfp4_step199_bf16_gen4096_seed42.jsonl
    DEFAULT_OUTPUT_DIR=/codebase/bf16_slime/qad/qwen2.5_math_7b_nvfp4_step199_qad_large4096_lr1e6
    ;;
  medium17078_2ep_lr5e6)
    DEFAULT_DATA_CONFIG=qad_dapo_gen17078.yaml
    DEFAULT_DATA_FILE=/codebase/bf16_slime/qad/qwen2.5_math_7b_nvfp4_step199_bf16_gen17078_seed42.jsonl
    DEFAULT_OUTPUT_DIR=/codebase/bf16_slime/qad/qwen2.5_math_7b_nvfp4_step199_qad_medium17078_2ep_lr5e6
    ;;
  *)
    echo "Unknown QAD_PROFILE: ${QAD_PROFILE}" >&2
    echo "Expected one of: smoke, gate50_lr1e6, large4096_lr1e6, medium17078_2ep_lr5e6" >&2
    exit 2
    ;;
esac

QAD_TRAIN_CONFIG=${QAD_TRAIN_CONFIG:-${SCRIPT_DIR}/configs/train/qad_${QAD_PROFILE}.yaml}
QAD_DATASET_CONFIG=${QAD_DATASET_CONFIG:-${SCRIPT_DIR}/configs/dataset/${DEFAULT_DATA_CONFIG}}
QAD_QUANT_RECIPE=${QAD_QUANT_RECIPE:-${SCRIPT_DIR}/configs/quantization/qad_nvfp4_mlp0_23.yaml}
QAD_DATA_FILE=${QAD_DATA_FILE:-${DEFAULT_DATA_FILE}}
QAD_OUTPUT_DIR=${QAD_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}

QAD_PYTHON=${QAD_VENV}/bin/python
QAD_ACCELERATE=${QAD_VENV}/bin/accelerate
FSDP_CONFIG=${EXAMPLE_DIR}/configs/accelerate/fsdp2.yaml

for required_file in \
  "${QAD_PYTHON}" \
  "${QAD_ACCELERATE}" \
  "${QAD_TRAIN_CONFIG}" \
  "${QAD_DATASET_CONFIG}" \
  "${QAD_QUANT_RECIPE}" \
  "${QAD_DATA_FILE}" \
  "${FSDP_CONFIG}"; do
  if [[ ! -e "${required_file}" ]]; then
    echo "Required path does not exist: ${required_file}" >&2
    exit 1
  fi
done

if [[ ! -d "${QAD_TEACHER_MODEL}" ]]; then
  echo "Teacher model does not exist: ${QAD_TEACHER_MODEL}" >&2
  exit 1
fi

RESOLVED_DATASET_CONFIG=$(mktemp "${TMPDIR:-/tmp}/qad-dataset.XXXXXX.yaml")
RESOLVED_TRAIN_CONFIG=$(mktemp "${TMPDIR:-/tmp}/qad-train.XXXXXX.yaml")
trap 'rm -f -- "${RESOLVED_DATASET_CONFIG}" "${RESOLVED_TRAIN_CONFIG}"' EXIT

export QAD_DATA_FILE QAD_FAKEQUANT_MODEL QAD_OUTPUT_DIR QAD_TEACHER_MODEL
"${QAD_PYTHON}" "${SCRIPT_DIR}/resolve_env_yaml.py" \
  "${QAD_DATASET_CONFIG}" "${RESOLVED_DATASET_CONFIG}"
export QAD_RESOLVED_DATASET_CONFIG=${RESOLVED_DATASET_CONFIG}
"${QAD_PYTHON}" "${SCRIPT_DIR}/resolve_env_yaml.py" \
  "${QAD_TRAIN_CONFIG}" "${RESOLVED_TRAIN_CONFIG}"

export PYTHONPATH=${MODELOPT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false

if [[ "${QAD_DRY_RUN:-0}" == 1 ]]; then
  printf '%s\n' \
    "QAD profile: ${QAD_PROFILE}" \
    "Teacher: ${QAD_TEACHER_MODEL}" \
    "Fake-quant student: ${QAD_FAKEQUANT_MODEL}" \
    "Dataset: ${QAD_DATA_FILE}" \
    "Resolved dataset config: ${RESOLVED_DATASET_CONFIG}" \
    "Training config: ${QAD_TRAIN_CONFIG}" \
    "Resolved training config: ${RESOLVED_TRAIN_CONFIG}" \
    "Output: ${QAD_OUTPUT_DIR}" \
    "Processes: ${QAD_NUM_PROCESSES}"
  exit 0
fi

cd "${EXAMPLE_DIR}"

if [[ ! -f "${QAD_FAKEQUANT_MODEL}/modelopt_state.pth" ]]; then
  "${QAD_PYTHON}" quantize.py \
    --model_name_or_path "${QAD_TEACHER_MODEL}" \
    --dataset_config "${RESOLVED_DATASET_CONFIG}" \
    --recipe "${QAD_QUANT_RECIPE}" \
    --model_max_length 2048 \
    --calib_size 32 \
    --calib_batch_size 1 \
    --output_dir "${QAD_FAKEQUANT_MODEL}"
fi

"${QAD_ACCELERATE}" launch \
  --config-file "${FSDP_CONFIG}" \
  --num_processes "${QAD_NUM_PROCESSES}" \
  --fsdp_transformer_layer_cls_to_wrap Qwen2DecoderLayer \
  train.py \
  --config "${RESOLVED_TRAIN_CONFIG}"
