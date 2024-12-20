# train.sh
#!/bin/bash

export PROJECT_ROOT="$(dirname "$(realpath "$0")")"

models=("bart" "legal_bert" "t5_summary")
data="lex_sum"

for model in "${models[@]}"
do
  python src/train.py model=$model data=$data name="${model}_output" task_name="${model}_summarization"
done

# For translation models
models=("mbart" "t5_translate")
data_sources=("ted_talks" "europarl")

for model in "${models[@]}"
do
  for data in "${data_sources[@]}"
  do
    python src/train.py model=$model data=$data name="${model}_${data}_output" task_name="${model}_${data}_translation"
  done
done
