# usage: bash dynamic_sample.sh [dataset] [sampler] [model] [loss] [epochs]
dataset=$1
sampler=$2
model=$3
loss=$4
num_epochs=$5

# train first epoch
echo -e "\n\ntraining 1 epoch...\n"
cd hard_negative_sampler/${sampler}/
python ${sampler}.py -trd ${dataset} -m ${model}
cd ../../
python main.py -trd ${dataset}_train -ted ${dataset}_test -hns datasets/${dataset}-hns/${sampler}_5.jsonl -e 1 -m ${model} -l ${loss}

# train second~last epoch
for ((i=2; i<=${num_epochs}; i++))
do
    echo -e "\n\ntraining $i epoch...\n"
    cd hard_negative_sampler/${sampler}/
    python ${sampler}.py -trd ${dataset} -m ../../output/${model//\//-}-${dataset}_test-${loss}0-1epochs/
    cd ../../
    python main.py -trd ${dataset}_train -ted ${dataset}_test -hns datasets/${dataset}-hns/${sampler}_5.jsonl -e 1 -m output/${model//\//-}-${dataset}_test-${loss}0-1epochs/ -l ${loss}
done
