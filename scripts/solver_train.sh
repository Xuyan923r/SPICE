solver_model_path=$1
questioner_model_path=$2
experiment_name=$3
dataset_path=${4:-${STORAGE_PATH}/datasets}

echo $STORAGE_PATH
DATASET_DIR=${STORAGE_PATH}/solver_data/${experiment_name}

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'

# 自动处理语料路径：如果给的是目录，则优先取其中的 parquet/jsonl/json 文件
resolved_dataset_path="$dataset_path"
if [ -d "$dataset_path" ]; then
    first_file=$(ls "$dataset_path"/*.parquet 2>/dev/null | head -n 1)
    if [ -z "$first_file" ]; then
        first_file=$(ls "$dataset_path"/*.jsonl 2>/dev/null | head -n 1)
    fi
    if [ -z "$first_file" ]; then
        first_file=$(ls "$dataset_path"/*.json 2>/dev/null | head -n 1)
    fi

    if [ -z "$first_file" ]; then
        echo "ERROR: No parquet/jsonl/json found under directory: $dataset_path"
        exit 1
    fi
    resolved_dataset_path="$first_file"
    echo "Detected dataset file: $resolved_dataset_path"
fi

bash question_generate/question_generate.bash $questioner_model_path 1000 $experiment_name $resolved_dataset_path
echo 'start evaluate generated question'
bash question_evaluate/evaluate.sh $solver_model_path $experiment_name
echo 'prepare local dataset'
python question_evaluate/upload.py --max_score 0.8 --min_score 0.3 --experiment_name ${experiment_name} --output_dir ${DATASET_DIR}
echo 'start train'

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=2048 \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    data.train_files=${DATASET_DIR}/train.parquet \
    data.val_files=${DATASET_DIR}/train.parquet \
    trainer.total_epochs=100 \
    trainer.max_steps=25 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    trainer.val_freq=4 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor

sleep 10

echo "solver training finished"

bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface
