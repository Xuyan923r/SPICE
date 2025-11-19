# load the model name from the command line
model_name=$1
num_samples=$2
save_name=$3
corpus_path=$4
context_column=${5:-text}
id_column=${6:-id}

if [ -z "$corpus_path" ]; then
    echo "Usage: $0 <model_name> <num_samples> <save_name> <corpus_path> [context_column] [id_column]"
    exit 1
fi
export VLLM_DISABLE_COMPILE_CACHE=1
COMMON_ARGS="--model $model_name --num_samples $num_samples --save_name $save_name --corpus_path $corpus_path --context_column $context_column --id_column $id_column"

CUDA_VISIBLE_DEVICES=0 python question_generate/question_generate.py $COMMON_ARGS --suffix 0 &
CUDA_VISIBLE_DEVICES=1 python question_generate/question_generate.py $COMMON_ARGS --suffix 1 &
CUDA_VISIBLE_DEVICES=2 python question_generate/question_generate.py $COMMON_ARGS --suffix 2 &
CUDA_VISIBLE_DEVICES=3 python question_generate/question_generate.py $COMMON_ARGS --suffix 3 &
CUDA_VISIBLE_DEVICES=4 python question_generate/question_generate.py $COMMON_ARGS --suffix 4 &
CUDA_VISIBLE_DEVICES=5 python question_generate/question_generate.py $COMMON_ARGS --suffix 5 &
CUDA_VISIBLE_DEVICES=6 python question_generate/question_generate.py $COMMON_ARGS --suffix 6 &
CUDA_VISIBLE_DEVICES=7 python question_generate/question_generate.py $COMMON_ARGS --suffix 7 &

wait
