#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=pdtt_t5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

python dtt_t5.py\
    --model_input_length 1024\
    --model_output_length 512\
    --input_col input\
    --user_id user_id\
    --dtt_ref_col target\
    --pdtt_refs_col outputs\
    --parent_col parent\
    --data_config_path dtt_t5_config_example.json\
    --batch_size 8\
    --epochs 10\
    --train_size 0.8\
    --seed 42\
    --model_name t5-small\
    --authorship_model_path ../authorship/bert_models/bert_authorship_2.pt\
    --authorship_n_authors 2\
    --similarity_path ../style_paraphrase/evaluation/similarity/sim/
