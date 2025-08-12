#!/bin/bash

#SBATCH -J clm_eval
#SBATCH -o ret-%j.out
#SBATCH -e ret-%j.err

#SBATCH -p nv-gpu
#SBATCH -t 0-18:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos=gpu-normal
#SBATCH --constraint="A100"

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
source /tools/module_env.sh
module load cluster-tools/v1.0
module load slurm-tools/v1.0
module load cmake/3.15.7
module load git/2.17.1
module load vim/8.1.2424
module load cuda-cudnn/11.8-8.8.1

echo $(module list)

echo $(which gcc)
echo $(which python)

echo "Use GPU ${CUDA_VISIBLE_DEVICES}"

source ~/.bashrc
conda activate olive801

echo "conda success!!!"

cd "$(dirname "$0")/.."

transformer_model=${1:-"/lustre/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-2025Q1"}
dataset_path=${2:-"/workspace/I/fangqiyan/Quantization_design/wikitext2_llama_block1024"}
q_mode=${3:-"ant-int-flint"}
q_bit=${4:-"4"}
batch_size=${5:-"8"}
port=${6:-46666}
desc=${7:-""}
n8=${8:-"0"}

srun -l --gres=gpu:1 -N1 -n1 bash scripts/clm_run.sh $transformer_model wikitext wikitext2_llama_block1024 $dataset_path $q_mode $q_bit $batch_size $port $desc $n8

wait
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
