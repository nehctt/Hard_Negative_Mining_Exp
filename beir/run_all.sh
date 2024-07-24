# usage: bash run.sh [dataset] [model] [epochs]
dataset=$1
model=$2
num_epochs=$3

#### run all negative selection methods
# random
cd ./hard_negative_sampler/random/
python random_sampler.py -trd ${dataset}
cd ../../
python main.py -trd ${dataset}_train -ted ${dataset}_test -hns random -m ${model} -l infonce -e ${num_epochs}
# bm25
cd ./hard_negative_sampler/bm25/
bash bm25.sh ${dataset}
cd ../../
python main.py -trd ${dataset}_train -ted ${dataset}_test -hns bm25 -m ${model} -l infonce -e ${num_epochs}
# ance
bash dynamic_sample.sh ${dataset} ance ${model} infonce ${num_epochs}
# adore
bash dynamic_sample.sh ${dataset} adore ${model} infonce ${num_epochs}
# ours - infoncedm
python main.py -trd ${dataset}_train -ted ${dataset}_test -hns random -m ${model} -l infoncedm -e ${num_epochs}

#### apply our method (indoncedm) on others
python main.py -trd ${dataset}_train -ted ${dataset}_test -hns bm25 -m ${model} -l infoncedm -e ${num_epochs}
bash dynamic_sample.sh ${dataset} ance ${model} infoncedm ${num_epochs}
bash dynamic_sample.sh ${dataset} adore ${model} infoncedm ${num_epochs}
