# Evaluate IR Tasks through BEIR  

## Install  
```
pip install beir sentence-transformers==2.2.2
```

<!-- ## Evaluate pretrained LLMs   -->
<!-- ``` -->
<!-- python main.py -trd nfcorpus_test -ted nfcorpus_test -m intfloat/e5-small -e 0 -->
<!-- ``` -->
<!--  -->
<!-- ## Finetune and evaluate pretrained LLMs   -->
<!-- ``` -->
<!-- python main.py -trd nfcorpus_train -ted nfcorpus_test -m intfloat/e5-small -e 5 -->
<!-- ``` -->
<!--  -->
<!-- ## Finetune on hard negative samples and evaluate pretrained LLMs   -->
<!-- ### BM25   -->
<!-- First, install elastic search   -->
<!-- Second, get training triplets (query_text, postive_text, negtive_text)   -->
<!-- ``` -->
<!-- cd hard_negative_sampler/bm25/ -->
<!-- ./run.sh -->
<!-- ``` -->
<!-- Third, finetune and evaluate   -->
<!-- ``` -->
<!-- python main.py -trd nfcorpus_test -ted nfcorpus_test -hns datasets/nfcorpus-hns/bm25.jsonl -m intfloat/e5-small -e 5 -->
<!-- ``` -->
