# usage: bash dynamic_sample.sh [dataset] [sampler] [model] [loss] [epochs]
dataset=$1
sampler=$2
model=$3
loss=$4
num_epochs=$5

# train first epoch
echo -e "\n\ntraining 1 epoch...\n"
cd hard_negative_sampler/dynamic/${sampler}/
python ${sampler}.py -trd ${dataset} -m ${model} -warmup True
cd ../../../
python main.py \
    -trd ${dataset}_train \
    -ted ${dataset}_test \
    -hns ${sampler} \
    -m ${model} \
    -e 1 \
    -l ${loss}

# train second~last epoch
for ((i=2; i<=${num_epochs}; i++))
do
    echo -e "\n\ntraining $i epoch...\n"
    cd hard_negative_sampler/dynamic/${sampler}/
    python ${sampler}.py -trd ${dataset} -m ../../../output/${dataset}_${sampler}5_${model//\//-}_${loss}_1epochs/
    cd ../../../
    python main.py \
        -trd ${dataset}_train \
        -ted ${dataset}_test \
        -hns ${sampler} \
        -m output/${dataset}_${sampler}5_${model//\//-}_${loss}_1epochs/ \
        -e 1 \
        -l ${loss}
done
