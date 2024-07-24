# Evaluate IR Tasks through BEIR  

## Install (python=3.10)  
```
pip install beir sentence-transformers==2.2.2
```

## Evaluate pretrained LLMs  
```
python main.py -trd scifact_test -ted scifact_test -m intfloat/e5-small -e 0
```

## Test our method
```
bash run_ours.sh scifact intfloat/e5-small 10
```

## Run through all the hard negative selection methohds
```
bash run_all.sh scifact intfloat/e5-small 10
```
All methods:  
- random
- bm25
- ance(dynamic_sample)
- adore(dynamic_sample)
- ours
- ours+bm25
- ours+ance
- ours+adore
