#!/usr/bin/env bash

default_config() {
    if [[ -n "$1" ]]; then
        printf '%s\n' "$1"
    else
        printf '%s\n' "bf16"
    fi
}

repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/../olmo2-training" && pwd -P
}

setup_env() {
    # module load cuda/13.0.0
    # source activate olmo-nvfp4

    export OLMO_SHARED_FS=1
    export NCCL_IB_DISABLE=0
    export NCCL_IB_TIMEOUT=50
    export NCCL_IB_RETRY_CNT=20
    export MASTER_PORT=23168

    # Keep W&B runs local by default; set WANDB_MODE=online here to sync.
    export WANDB_MODE=offline
}

train_args() {
    local model_size="$1"
    local config="$2"
    local save_folder="outputs/${model_size}/${config}"
    local config_path="configs/${model_size}/${config}.yaml"

    printf '%s\n' \
        "scripts/train.py" \
        "${config_path}" \
        "--run_name=${config}" \
        "--save_folder=${save_folder}" \
        "--save_overwrite=True" \
        "--try_load_latest_save=False"
}

run_single_node() {
    local model_size="$1"
    local config
    config="$(default_config "$2")"
    local load_path="$3"

    setup_env
    local root
    root="$(repo_root)"
    cd "${root}"

    local args=()
    mapfile -t args < <(train_args "${model_size}" "${config}")
    if [[ -n "${load_path}" ]]; then
        args+=("--load_path=${load_path}")
    fi

    torchrun --standalone --nproc_per_node=8 "${args[@]}"
}

run_multi_node() {
    local model_size="$1"
    local config
    config="$(default_config "$2")"
    local load_path="$3"

    setup_env
    local root
    root="$(repo_root)"
    cd "${root}"

    local nodes=()
    mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    local total_nodes="${#nodes[@]}"
    local master_addr="${nodes[0]}"

    local args=()
    mapfile -t args < <(train_args "${model_size}" "${config}")
    if [[ -n "${load_path}" ]]; then
        args+=("--load_path=${load_path}")
    fi

    for node_rank in $(seq 1 $((total_nodes - 1))); do
        srun --export=ALL --chdir="${root}" --ntasks=1 --nodes=1 --gres=gpu:8 --nodelist="${nodes[$node_rank]}" \
            torchrun --nnodes="${total_nodes}" --node_rank="${node_rank}" --nproc_per_node=8 \
                --master_addr="${master_addr}" --master_port="${MASTER_PORT}" \
                "${args[@]}" &
    done

    torchrun --nnodes="${total_nodes}" --node_rank=0 --nproc_per_node=8 \
        --master_addr="${master_addr}" --master_port="${MASTER_PORT}" \
        "${args[@]}" &

    wait
}
