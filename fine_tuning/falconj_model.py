import torch
from torch.cuda.amp import GradScaler, autocast
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
import warnings
import bitsandbytes as bnb

# [Other functions: setup_logging, log_cuda_memory, save_checkpoint, find_latest_checkpoint, load_checkpoint]

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
        # Initialize the process group
        dist.init_process_group(backend='nccl')
        if enable_logging:
            logger.info("Initialized NCCL process group")

        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if enable_logging:
            logger.info(f"Process {local_rank}: Initialized on device {device}")

        # Load the model and tokenizer
        model_name = "tiiuae/falcon-7b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

        if enable_logging:
            logger.info(f"Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
            logger.info(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

        # Configure bitsandbytes for 8-bit loading
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load the model with quantization
        with record_function("model_loading"):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map={"": device}  # Load the entire model on the specified device
            )
            log_cuda_memory(logger, "After model loading and moving to device", enable_logging)

        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query_key_value", "dense"],  # Adjust based on the model's architecture
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        if enable_logging:
            logger.info(f"Process {local_rank}: Wrapped model with LoRA")

        # Wrap the model with DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        if enable_logging:
            logger.info(f"Process {local_rank}: Wrapped model with DistributedDataParallel")

        # Read the data from the CSV file
        data_path = '../data_generation/combined_results.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        data = pd.read_csv(data_path)

        # Load only the first 1% of the dataset for testing purposes
        data = data.head(int(len(data) * 0.01))
        if enable_logging:
            logger.info(f"Loaded {len(data)} samples for training")
        print(f"Loaded {len(data)} samples for training")

        # Define a custom Dataset class
        class PromptResponseDataset(Dataset):
            def __init__(self, dataframe, tokenizer, max_length=512):  # Adjust max_length as needed
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

        # Create the dataset and DistributedSampler
        batch_size = 16
        dataset = PromptResponseDataset(data, tokenizer)
        sampler = DistributedSampler(dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=num_workers)
        if enable_logging:
            logger.info(f"Process {local_rank}: Created DataLoader with batch size {batch_size} and num_workers={num_workers}")
        print(f"Process {local_rank}: Created DataLoader with batch size {batch_size} and num_workers={num_workers}")

        # Find the latest checkpoint if available
        len_dataloader = len(dataloader)
        latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir='checkpoints', len_dataloader=len_dataloader)

        # Initialize optimizer and scaler before loading checkpoint
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5)  # 8-bit AdamW
        if enable_logging:
            logger.info("Initialized 8-bit AdamW optimizer")

        scaler = GradScaler()
        if enable_logging:
            logger.info("Initialized GradScaler for mixed precision")

        # Load the latest checkpoint if available
        loaded_epoch, loaded_batch_idx = 0, 0
        losses = []
        if latest_checkpoint_path:
            loaded_epoch, loaded_batch_idx = load_checkpoint(
                latest_checkpoint_path,
                model,
                optimizer,
                scaler,
                losses,
                device,
                local_rank,
                enable_logging,
                logger
            )
        else:
            if enable_logging:
                logger.info("No checkpoints found. Starting training from scratch.")
            print("No checkpoints found. Starting training from scratch.")

        # Broadcast the loaded_epoch and loaded_batch_idx to all ranks
        loaded_epoch_tensor = torch.tensor(loaded_epoch).to(device)
        loaded_batch_idx_tensor = torch.tensor(loaded_batch_idx).to(device)
        dist.broadcast(loaded_epoch_tensor, src=0)
        dist.broadcast(loaded_batch_idx_tensor, src=0)
        loaded_epoch = loaded_epoch_tensor.item()
        loaded_batch_idx = loaded_batch_idx_tensor.item()

        if enable_logging and loaded_epoch > 0:
            logger.info(f"Resuming from Epoch {loaded_epoch}, Batch {loaded_batch_idx}")
            print(f"Resuming from Epoch {loaded_epoch}, Batch {loaded_batch_idx}")

        # Initialize 'epoch' before the training loop
        epoch = loaded_epoch - 1  # Initialize to one less than loaded_epoch

        # Define gradient accumulation steps
        gradient_accumulation_steps = 4
        accumulated_loss = 0.0  # To track accumulated loss for logging

        # Create checkpoint directory
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Determine whether to enable profiling based on num_workers
        enable_profiling = num_workers == 0

        # Initialize profiler if profiling is enabled
        if enable_profiling:
            os.makedirs('profiler_logs', exist_ok=True)
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=lambda p: p.export_chrome_trace(os.path.join('profiler_logs', f'trace_{local_rank}.json')),
                with_stack=True
            )
            profiler.start()

        # Training loop with mixed precision and gradient accumulation
        total_epochs = 2
        model.train()
        for epoch in range(loaded_epoch, total_epochs):
            sampler.set_epoch(epoch)  # Shuffle data differently at each epoch
            if local_rank == 0 and enable_logging:
                logger.info(f"Epoch {epoch+1}/{total_epochs} started")
                print(f"Epoch {epoch+1}/{total_epochs} started")
            progress_bar = tqdm(dataloader, desc=f"Rank {local_rank} Training", leave=False) if local_rank == 0 else dataloader
            dataloader_iter = iter(dataloader)

            # If resuming in the middle of an epoch, skip the first 'loaded_batch_idx' batches
            if epoch == loaded_epoch and loaded_batch_idx > 0:
                if enable_logging:
                    logger.info(f"Skipping first {loaded_batch_idx} batches of Epoch {epoch+1}")
                print(f"Skipping first {loaded_batch_idx} batches of Epoch {epoch+1}")
                for _ in range(loaded_batch_idx):
                    try:
                        next(dataloader_iter)
                    except StopIteration:
                        break

            for batch_idx, batch in enumerate(dataloader_iter, start=loaded_batch_idx if epoch == loaded_epoch else 0):
                optimizer.zero_grad(set_to_none=True)
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                log_cuda_memory(logger, f"Before forward pass - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)
                with record_function("forward_pass"):
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss / gradient_accumulation_steps  # Scale loss for accumulation
                log_cuda_memory(logger, f"After forward pass - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)

                with record_function("backward_pass"):
                    scaler.scale(loss).backward()
                    accumulated_loss += loss.item() * gradient_accumulation_steps
                log_cuda_memory(logger, f"After backward pass - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)

                # Update optimizer every gradient_accumulation_steps
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
                        if enable_logging:
                            progress_bar.set_postfix({'loss': current_loss})
                            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)} - Loss: {current_loss:.4f}")

                    # Save checkpoint every 100 batches
                    if (batch_idx + 1) % 100 == 0 and (batch_idx + 1) <= len(dataloader):
                        save_checkpoint(
                            epoch=epoch + 1,
                            batch_idx=batch_idx + 1,
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            losses=losses,
                            checkpoint_dir=checkpoint_dir,
                            local_rank=local_rank,
                            enable_logging=enable_logging,
                            logger=logger
                        )

                # Step the profiler if profiling is enabled
                if enable_profiling:
                    profiler.step()

            # Reset loaded_batch_idx after the first epoch
            loaded_batch_idx = 0

        # Stop the profiler after training loop
        if enable_profiling:
            profiler.stop()

        if local_rank == 0:
            print("Training completed.")
            if enable_logging:
                logger.info("Training completed.")
            # Save final checkpoint at the end of the training
            save_checkpoint(
                epoch=epoch + 1,
                batch_idx=len(dataloader),
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                losses=losses,
                checkpoint_dir=checkpoint_dir,
                local_rank=local_rank,
                enable_logging=enable_logging,
                logger=logger
            )
            # Save the trained model and tokenizer
            output_dir = "./trained_model"
            model.module.save_pretrained(output_dir)  # Use model.module when saving
            tokenizer.save_pretrained(output_dir)
            if enable_logging:
                logger.info(f"Model saved to {output_dir}")

            # Save losses for visualization
            with open("losses.txt", "w") as f:
                f.write("\n".join(map(str, losses)))
            if enable_logging:
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
            if enable_logging:
                logger.info("Saved training loss curve to loss_curve.png")

    except torch.cuda.OutOfMemoryError as e:
        if enable_logging and logger is not None:
            logger.error(f"CUDA Out of Memory Error: {e}")
        print(f"CUDA Out of Memory: {e}")
    except Exception as e:
        if enable_logging and logger is not None:
            logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")
    finally:
        dist.destroy_process_group()
        if enable_logging and logger is not None:
            logger.info("Destroyed the distributed process group")

if __name__ == "__main__":
    main()
