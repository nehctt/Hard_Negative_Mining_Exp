# Evaluate IR Tasks through BEIR  

## Install  
```
pip install beir sentence-transformers==2.2.2
```

## Evaluate pretrained LLMs  
```
python main.py -trd scifact_test -ted scifact_test -m intfloat/e5-small -e 0
```

## Finetune and evaluate pretrained LLMs  
```
python main.py -trd scifact_train -ted scifact_test -hns random -m intfloat/e5-small -l infoncedm -e 10
```

## Run through all the hard negative selection methohds
```
bash run3.sh scifact intfloat/e5-small 10
```
