#!/bin/bash

solver_model_path=$1
questioner_model_path=$2
save_path=$3
dataset_path=$4

if [ -z "$dataset_path" ]; then
    echo "Usage: $0 <solver_model_path> <questioner_model_path> <save_path> <dataset_path>"
    exit 1
fi

echo "save_path: $save_path"
RUN_ID=$(date +%s%N)
export RUN_ID

echo "RUN_ID=$RUN_ID"

bash vllm_service_init/start.sh $solver_model_path $RUN_ID
echo "vLLM services started"

# ============================================================================
# 自动读取目录中的所有 parquet 文件
# ============================================================================
if [ -d "$dataset_path" ]; then
    echo "Dataset path is a directory. Scanning for parquet files..."
    parquet_files=$(ls "$dataset_path"/*.parquet 2>/dev/null)

    if [ -z "$parquet_files" ]; then
        echo "ERROR: No parquet files found in directory: $dataset_path"
        exit 1
    fi

    # 用逗号拼接成 HF datasets 可识别的路径
    dataset_files=$(echo $parquet_files | tr ' ' ',')
else
    # 如果传入的是单个 parquet 文件
    dataset_files=$dataset_path
fi

echo "Detected parquet dataset files:"
echo "$dataset_files"

export QUESTIONER_DUMP_DIR=${STORAGE_PATH}/questioner_outputs/${save_path}

# ============================================================================
# 开始训练
# ============================================================================
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="$dataset_files" \
    data.val_files="$dataset_files" \
    data.prompt_key=text \
    data.context_key=text \
    data.answer_key=id \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    worker.reward.reward_function=./examples/reward_function/caller_penalty.py:compute_score \
    trainer.total_epochs=100 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=4 \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=16 \
    trainer.max_steps=20 \
    trainer.save_freq=5

sleep 5

# ============================================================================
# 合并模型
# ============================================================================
echo "Merging model..."
MERGE_DIR=$(ls -d ${STORAGE_PATH}/models/$save_path/global_step_*/actor 2>/dev/null | sort -V | tail -n 1)

if [ -z "$MERGE_DIR" ]; then
    echo "ERROR: No valid checkpoint directory found to merge."
else
    echo "Merging latest checkpoint: $MERGE_DIR"
    python scripts/model_merger.py --local_dir "$MERGE_DIR" #到actor为止
fi

sleep 10
pkill python

echo "questioner training finished"

 