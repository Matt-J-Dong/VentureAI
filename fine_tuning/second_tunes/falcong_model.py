import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
import argparse
import os
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from peft import LoraConfig, get_peft_model, TaskType
import logging

def setup_logging(log_file='cuda_memory.txt'):
    logger = logging.getLogger('CUDA_Memory_Logger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def log_cuda_memory(logger, message, enable_logging):
    """
    Logs CUDA memory usage if logging is enabled.
    """
    if enable_logging:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        logger.info(f"{message} | Allocated: {allocated / (1024**2):.2f} MB | Reserved: {reserved / (1024**2):.2f} MB | Max Allocated: {max_allocated / (1024**2):.2f} MB | Max Reserved: {max_reserved / (1024**2):.2f} MB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    args = parser.parse_args()
    local_rank = args.local_rank

    num_workers = 4  # Change this value as needed
    enable_logging = num_workers == 0  # Enable logging only if num_workers=0

    logger = setup_logging('cuda_memory.txt') if enable_logging else None
    if enable_logging:
        logger.info("Starting Training Script with logging enabled")

    try:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if enable_logging:
            logger.info(f"Process {local_rank}: Initialized on device {device}")

        model_name = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": device}
        )
        if enable_logging:
            log_cuda_memory(logger, "After model loading and moving to device", enable_logging)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query_key_value", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        if enable_logging:
            logger.info(f"Process {local_rank}: Wrapped model with LoRA")

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        if enable_logging:
            logger.info(f"Process {local_rank}: Wrapped model with DistributedDataParallel")

        data = pd.read_csv('../data_generation/combined_results.csv')
        data = data.head(int(len(data) * 0.1))
        if enable_logging:
            logger.info(f"Loaded {len(data)} samples for training")

        class PromptResponseDataset(Dataset):
            def __init__(self, dataframe, tokenizer, max_length=128):
                self.dataframe = dataframe.reset_index(drop=True)
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.dataframe)

            def __getitem__(self, idx):
                prompt = str(self.dataframe.iloc[idx]['prompt'])
                response = str(self.dataframe.iloc[idx]['response'])
                input_text = f"{prompt}{tokenizer.eos_token}{response}{tokenizer.eos_token}"
                encoding = tokenizer(
                    input_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].squeeze()
                attention_mask = encoding['attention_mask'].squeeze()
                labels = input_ids.clone()
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }

        dataset = PromptResponseDataset(data, tokenizer)
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, pin_memory=True, num_workers=num_workers)
        if enable_logging:
            logger.info(f"Process {local_rank}: Created DataLoader with batch size 1 and num_workers={num_workers}")

        optimizer = AdamW(model.parameters(), lr=5e-5)
        scaler = GradScaler()
        if enable_logging:
            logger.info("Initialized AdamW optimizer and GradScaler")

        losses = []
        gradient_accumulation_steps = 4
        accumulated_loss = 0.0

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            with_stack=True
        ) as prof:
            epochs = 1
            model.train()
            for epoch in range(epochs):
                sampler.set_epoch(epoch)
                if local_rank == 0 and enable_logging:
                    logger.info(f"Epoch {epoch+1}/{epochs} started")
                progress_bar = tqdm(dataloader, desc=f"Rank {local_rank} Training", leave=False) if local_rank == 0 else dataloader
                for batch_idx, batch in enumerate(progress_bar):
                    optimizer.zero_grad(set_to_none=True)
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    labels = batch['labels'].to(device, non_blocking=True)

                    log_cuda_memory(logger, f"Before forward pass - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)
                    with record_function("forward_pass"):
                        with autocast('cuda'):
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                            loss = outputs.loss / gradient_accumulation_steps
                    log_cuda_memory(logger, f"After forward pass - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)

                    with record_function("backward_pass"):
                        scaler.scale(loss).backward()
                        accumulated_loss += loss.item() * gradient_accumulation_steps
                    log_cuda_memory(logger, f"After backward pass - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)

                    if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                        with record_function("optimizer_step"):
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                        log_cuda_memory(logger, f"After optimizer step - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)

                        if local_rank == 0:
                            current_loss = accumulated_loss / gradient_accumulation_steps
                            losses.append(current_loss)
                            accumulated_loss = 0.0
                            progress_bar.set_postfix({'loss': current_loss})
                            if enable_logging:
                                logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)} - Loss: {current_loss:.4f}")
                    prof.step()

    except Exception as e:
        if enable_logging:
            logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
    finally:
        dist.destroy_process_group()
        if enable_logging:
            logger.info("Destroyed the distributed process group")

if __name__ == "__main__":
    main()
