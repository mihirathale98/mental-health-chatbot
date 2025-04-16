import modal

# Modal setup for cloud infrastructure
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
image = (
    image
    .apt_install("git")
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "torch==2.5.1",
        "transformers==4.48.3",
        "unsloth",
    )
)
image = image.run_commands(
    "CXX=g++ pip install flash-attn --no-build-isolation"
)
app_image = image.run_commands(
    "pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton",
    "pip install --no-deps cut_cross_entropy unsloth_zoo",
    "pip install sentencepiece protobuf datasets huggingface_hub hf_transfer",
).pip_install(
    "psutil",
    "Pillow",
    'gguf',
    'protobuf',
    'tqdm',
    'pandas',
    'datasets',
).apt_install(
    'cmake',
    'libcurl4-openssl-dev',
)

app = modal.App("llama3-1-finetune", image=app_image)

with app_image.imports():
    import modal
    import torch
    import os
    from transformers import (
        TrainingArguments, 
    )
    from peft import get_peft_model, LoraConfig, TaskType
    import pandas as pd
    import random
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset


# Create secrets and volumes
huggingface_token = modal.Secret.from_name("huggingface-secret")
model_volume = modal.Volume.from_name("model_volume", create_if_missing=True)
data_volume = modal.Volume.from_name("data_volume", create_if_missing=True)

def format_dataset_for_llama3(example):
    """
    Format a single data point for Llama 3.1 finetuning
    Args:
        example: Dictionary containing 'instruction', 'input', and 'output'
    Returns:
        Formatted data string
    """
    # Handle empty input gracefully
    if not example["input"] or pd.isna(example["input"]):
        input_text = ""
    else:
        input_text = example["input"]
        
    if input_text:
        prompt = f"<|user|>\n{example['instruction']}\n\n{input_text}<|assistant|>\n{example['output']}<|end|>"
    else:
        prompt = f"<|user|>\n{example['instruction']}<|assistant|>\n{example['output']}<|end|>"
        
    return prompt

@app.function(
    cpu=(8, 8),
    memory=32768,
    gpu="A100-40GB:1",
    timeout=24 * 60 * 60,
    secrets=[huggingface_token],
    volumes={"/model": model_volume, "/data": data_volume},
)
def prepare_and_finetune(
    dataset_id: str, 
    model_name: str = "unsloth/Llama-3.1-8B-Instruct",
    max_length: int = 4096,
    seed: int = 42
):
    """
    Prepare dataset and finetune Llama 3 model
    
    Args:
        dataset_id: Hugging Face dataset ID
        model_name: Model to finetune
        val_split: Validation split ratio
        max_length: Maximum sequence length
        output_dir: Output directory for model
        seed: Random seed
    """
    # Set environment variables and random seeds
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/model/hf_cache"
    random.seed(seed)
    torch.manual_seed(seed)    

    max_seq_length = max_length # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    ## Load dataset

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    dataset = load_dataset(dataset_id, split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    ## split dataset
    dataset = dataset.train_test_split(test_size = 0.1, seed = 3407)
    dataset.save_to_disk(f"/data/{dataset_id}_custom")
    dataset = dataset["train"]

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 1,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps=2,
            warmup_steps = 5,
            num_train_epochs = 1,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = '/model/' + model_name.replace("/", "_").replace("-", "_") + "_finetuned",
            save_steps=300,
            save_total_limit=2
        ),
    )
    trainer_stats = trainer.train()

    return "Finetuning completed successfully!"


@app.local_entrypoint()
def main():
    dataset_id = "ShenLab/MentalChat16K"
    model_id = "unsloth/Llama-3.1-8B-Instruct"
    prepare_and_finetune.remote(dataset_id, model_id)
