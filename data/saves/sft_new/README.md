---
library_name: peft
license: other
base_model: /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct
tags:
- base_model:adapter:/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: sft_new
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_new

This model is a fine-tuned version of [/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct](https://huggingface.co//root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct) on the alpaca_gpt4_zh, the identity and the adgen_local datasets.
It achieves the following results on the evaluation set:
- Loss: 1.7883

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 20
- num_epochs: 5.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 2.1172        | 0.4246 | 50   | 1.9601          |
| 1.907         | 0.8493 | 100  | 1.8612          |
| 1.7488        | 1.2718 | 150  | 1.8222          |
| 1.7094        | 1.6964 | 200  | 1.8027          |
| 1.6739        | 2.1189 | 250  | 1.7965          |
| 1.5648        | 2.5435 | 300  | 1.7976          |
| 1.5735        | 2.9682 | 350  | 1.7883          |
| 1.4883        | 3.3907 | 400  | 1.8056          |
| 1.4377        | 3.8153 | 450  | 1.8064          |
| 1.4036        | 4.2378 | 500  | 1.8157          |
| 1.4182        | 4.6624 | 550  | 1.8191          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.8.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.1