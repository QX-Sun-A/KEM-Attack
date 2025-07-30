#!/bin/bash

# === Key Configuration Parameters ===
DATASET=${1:-"mr"}        # Dataset to process (default: mr)
GPU_ID=${2:-"0"}          # GPU ID (default: 0)
DATA_SIZE=1000            # Number of samples to process

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="KEM_results_${DATASET}_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR
cache_dir="$OUTPUT_DIR/use_cache"
mkdir -p $cache_dir

# Dataset configurations
declare -A DATASET_CONFIGS=(
    ["imdb"]="2 512 /imdb"
    ["yelp"]="2 256 /yelp"
    ["mr"]="2 256 /mr"
    ["ag"]="4 512 /ag"
)

# Get dataset configuration
config=(${DATASET_CONFIGS[$DATASET]})
nclasses=${config[0]}
max_seq_length=${config[1]}
model_path=${config[2]}

# Set attack parameters
sim_threshold=0.8
wolf_pop=60
max_iter=20
max_replacements=6

# Run the attack
python KEM-Attack.py \
    --dataset_path data/${DATASET} \
    --target_model bert \
    --target_model_path ${model_path} \
    --nclasses ${nclasses} \
    --relaxed_sememe_match \
    --sememe_similarity_threshold 0.3 \
    --wolf_population ${wolf_pop} \
    --max_iterations ${max_iter} \
    --output_dir $OUTPUT_DIR \
    --semantic_sim_model USE \
    --USE_cache_path "$cache_dir" \
    --preserved_stopwords "" \
    --synonym_num 50 \
    --max_replacements ${max_replacements} \
    --mutation_rate 0.1 \
    --sim_score_threshold ${sim_threshold} \
    --max_seq_length ${max_seq_length} \
    --data_size $DATA_SIZE \
    --data_offset 0 \
    --gpu_id $GPU_ID
