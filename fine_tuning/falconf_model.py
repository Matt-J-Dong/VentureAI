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
    """
    Sets up logging to the specified log_file.
    """
    logger = logging.getLogger('CUDA_Memory_Logger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def log_cuda_memory(logger, message):
    """
    Logs the current CUDA memory usage with a custom message.
    """
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

    # Setup CUDA memory logging
    logger = setup_logging('cuda_memory.txt')
    logger.info("Starting Training Script")

    try:
        # Initialize the process group
        dist.init_process_group(backend='nccl')

        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        logger.info(f"Process {local_rank}: Initialized on device {device}")

        # Load the model and tokenizer
        model_name = "tiiuae/falcon-7b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

        logger.info(f"Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        logger.info(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

        # Configure bitsandbytes for 8-bit model loading
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

        # Load the model in 8-bit precision
        with record_function("model_loading"):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map={"": device}
            )
            log_cuda_memory(logger, "After model loading and moving to device")

        # Prepare LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query_key_value", "dense"],  # Adjust based on the model architecture
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Wrap the model with LoRA
        model = get_peft_model(model, lora_config)
        logger.info(f"Process {local_rank}: Wrapped model with LoRA")

        # Wrap the model with DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        logger.info(f"Process {local_rank}: Wrapped model with DistributedDataParallel")

        # Read the data from the CSV file
        data = pd.read_csv('../data_generation/combined_results.csv')

        # Load only the first 10% of the dataset for testing purposes
        data = data.head(int(len(data) * 0.1))
        logger.info(f"Loaded {len(data)} samples for training")

        # Define a custom Dataset class
        class PromptResponseDataset(Dataset):
            def __init__(self, dataframe, tokenizer, max_length=128):  # Reduced max_length
                self.dataframe = dataframe.reset_index(drop=True)
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.dataframe)

            def __getitem__(self, idx):
                prompt = str(self.dataframe.iloc[idx]['prompt'])
                response = str(self.dataframe.iloc[idx]['response'])
                # Combine prompt and response
                input_text = f"{prompt}{tokenizer.eos_token}{response}{tokenizer.eos_token}"
                # Tokenize the input text
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

        # Create the dataset and DistributedSampler
        dataset = PromptResponseDataset(data, tokenizer)
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, pin_memory=True, num_workers=0) #0 for now, but I want 4
        logger.info(f"Process {local_rank}: Created DataLoader with batch size 1")

        # Set up the optimizer to only optimize LoRA parameters
        optimizer = AdamW(model.parameters(), lr=5e-5)
        logger.info("Initialized AdamW optimizer for LoRA parameters")

        # Initialize the GradScaler for mixed precision
        scaler = GradScaler()
        logger.info("Initialized GradScaler for mixed precision")

        # List to store the loss values
        losses = []

        # Define gradient accumulation steps
        gradient_accumulation_steps = 4
        accumulated_loss = 0.0  # To track accumulated loss for logging

        # Memory profiling
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=lambda p: p.export_chrome_trace("trace_" + str(local_rank) + ".json"),
        ) as prof:

            # Training loop with mixed precision and gradient accumulation
            epochs = 1
            model.train()
            for epoch in range(epochs):
                sampler.set_epoch(epoch)  # Shuffle data differently at each epoch
                if local_rank == 0:
                    print(f"Epoch {epoch+1}/{epochs}")
                    logger.info(f"Epoch {epoch+1}/{epochs} started")
                progress_bar = tqdm(dataloader, desc=f"Rank {local_rank} Training", leave=False) if local_rank == 0 else dataloader
                for batch_idx, batch in enumerate(progress_bar):
                    optimizer.zero_grad(set_to_none=True)
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    labels = batch['labels'].to(device, non_blocking=True)

                    # Record memory usage before forward pass
                    log_cuda_memory(logger, f"Before forward pass - Epoch {epoch+1}, Batch {batch_idx+1}")

                    # Forward pass with autocast for mixed precision
                    with record_function("forward_pass"):
                        with autocast('cuda'):
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            loss = outputs.loss / gradient_accumulation_steps  # Scale loss for accumulation

                    # Record memory usage after forward pass
                    log_cuda_memory(logger, f"After forward pass - Epoch {epoch+1}, Batch {batch_idx+1}")

                    # Backward pass
                    with record_function("backward_pass"):
                        scaler.scale(loss).backward()
                        accumulated_loss += loss.item() * gradient_accumulation_steps

                    # Record memory usage after backward pass
                    log_cuda_memory(logger, f"After backward pass - Epoch {epoch+1}, Batch {batch_idx+1}")

                    # Update optimizer every gradient_accumulation_steps
                    if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                        with record_function("optimizer_step"):
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)

                        # Record memory usage after optimizer step
                        log_cuda_memory(logger, f"After optimizer step - Epoch {epoch+1}, Batch {batch_idx+1}")

                        # Log the loss
                        if local_rank == 0:
                            current_loss = accumulated_loss / gradient_accumulation_steps
                            losses.append(current_loss)
                            accumulated_loss = 0.0
                            progress_bar.set_postfix({'loss': current_loss})
                            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)} - Loss: {current_loss:.4f}")

                    # Record profiler events
                    prof.step()

                if local_rank == 0:
                    print(f"Epoch {epoch+1} completed. Loss: {loss.item():.4f}")
                    logger.info(f"Epoch {epoch+1} completed. Loss: {loss.item():.4f}")
                    output_dir = "./trained_model"
                    model.module.save_pretrained(output_dir)  # Use model.module when saving
                    tokenizer.save_pretrained(output_dir)
                    logger.info(f"Model saved to {output_dir}")

            if local_rank == 0:
                print("Training completed.")
                logger.info("Training completed.")
                # Save losses for visualization
                with open("losses.txt", "w") as f:
                    f.write("\n".join(map(str, losses)))
                logger.info("Saved training losses to losses.txt")

                # Plot the training loss curve
                plt.figure(figsize=(10, 6))
                plt.plot(losses, label="Training Loss")
                plt.xlabel("Batch Iterations")
                plt.ylabel("Loss")
                plt.title("Training Loss Curve")
                plt.legend()
                plt.grid(True)
                plt.savefig("loss_curve.png")
                plt.show()
                logger.info("Saved training loss curve to loss_curve.png")

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA Out of Memory Error: {e}")
        print(f"CUDA Out of Memory: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up the distributed environment
        dist.destroy_process_group()
        logger.info("Destroyed the distributed process group")

if __name__ == "__main__":
    main()
