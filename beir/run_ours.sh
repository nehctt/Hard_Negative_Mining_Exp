# usage: bash run.sh [dataset] [model] [epochs]
dataset=$1
model=$2
num_epochs=$3

#### run all negative selection methods
# random
cd ./hard_negative_sampler/random/
python random_sampler.py -trd ${dataset}
cd ../../
python main.py -trd ${dataset}_train -ted ${dataset}_test -hns random -m ${model} -l infoncedm -e ${num_epochs}
