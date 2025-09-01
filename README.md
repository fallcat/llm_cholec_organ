# llm_cholec_organ
Using Large Vision-Language Models for Fine-Grained Organ Detection in Laparoscopic Cholecystectomy

## All experiments

1. Prepare few-shot examples

```
python notebooks_py/prepare_fewshot_examples.py
```

2. Evaluate pointing

```
EVAL_NUM_SAMPLES=20 python3 eval_pointing_original_size.py
```