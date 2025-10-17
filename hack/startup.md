# 微调示例
参考资料：https://zhuanlan.zhihu.com/p/695287607
### 环境

~~~text
OS Version      : Ubuntu 22.04.5 LTS
Kernel Version  : 5.15.0-126-generic
Hostname        : cci-8fa9873f-905c-48b1-840f-1cd6b97418f9-0
IP Address      : 172.16.246.28
CPU Model       : Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz
CPU Cores       : 10C
Memory Usage    : 124 MB / 81920 MB (0.15%)
GPU Information : NVIDIA L40S × 1
CUDA Version    : 12.8
~~~

### 下载 Meta-Llama-3-8B-Instruct 模型

~~~shell
pip install modelscope

modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct
~~~

### 开启 web 对话

~~~shell
export MODEL_PATH=/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct

CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
--model_name_or_path $MODEL_PATH \
--template llama3
~~~

### 微调

~~~shell
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
--stage sft \
--do_train \
--do_eval \
--model_name_or_path $MODEL_PATH \
--dataset alpaca_gpt4_zh,identity,adgen_local \
--dataset_dir ./data \
--template llama3 \
--finetuning_type lora \
--output_dir ./saves/LLaMA3-8B/lora/sft \
--overwrite_cache \
--overwrite_output_dir \
--cutoff_len 2048 \
--preprocessing_num_workers 16 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--logging_steps 50 \
--warmup_steps 20 \
--save_strategy steps \
--eval_strategy steps \
--save_steps 50 \
--eval_steps 50 \
--load_best_model_at_end \
--learning_rate 5e-5 \
--num_train_epochs 5.0 \
--max_samples 1000 \
--val_size 0.1 \
--plot_loss \
--fp16
~~~

#### 继续跑 --max_samples 10000 的微调

~~~shell
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
--stage sft \
--do_train \
--do_eval \
--model_name_or_path $MODEL_PATH \
--dataset alpaca_gpt4_zh,identity,adgen_local \
--dataset_dir ./data \
--template llama3 \
--finetuning_type lora \
--output_dir ./saves/LLaMA3-8B/lora/sft \
--overwrite_cache \
--resume_from_checkpoint ./saves/LLaMA3-8B/lora/sft/checkpoint-1500 \
--cutoff_len 1024 \
--preprocessing_num_workers 16 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--logging_steps 50 \
--warmup_steps 20 \
--save_strategy steps \
--eval_strategy steps \
--save_steps 50 \
--eval_steps 50 \
--load_best_model_at_end \
--learning_rate 5e-5 \
--num_train_epochs 5.0 \
--val_size 0.1 \
--plot_loss \
--fp16
~~~~


### 验证效果
~~~shell
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
--model_name_or_path $MODEL_PATH \
--adapter_name_or_path ./saves/LLaMA3-8B/lora/sft_new  \
--template llama3 \
--finetuning_type lora
~~~
