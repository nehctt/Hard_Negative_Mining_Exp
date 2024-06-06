# usage: bash bm25.sh [dataset]
dataset=$1 

# run elasticsearch
~/elasticsearch-7.9.2/bin/elasticsearch -d -p pid
sleep 15

# see if elastic run successfully
curl -sX GET "localhost:9200/"

# run bm25
python elastic_bm25.py -trd ${dataset}

# terminate elasticsearch
pkill -F ~/elasticsearch-7.9.2/pid
