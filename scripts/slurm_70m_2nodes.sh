#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -eo pipefail

source "${SLURM_SUBMIT_DIR}/common.sh"
run_multi_node 70m_50bt "$1" "$2"
