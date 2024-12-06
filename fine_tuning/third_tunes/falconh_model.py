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
import datetime
import bitsandbytes as bnb
import warnings

def setup_logging(log_file='cuda_memory.txt'):
    """
    Sets up logging to the specified log_file.
    """
    logger = logging.getLogger('CUDA_Memory_Logger')
    logger.setLevel(logging.INFO)
    # Avoid adding multiple handlers if the logger already has handlers
    if not logger.handlers:
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

def save_checkpoint(epoch, batch_idx, model, optimizer, scaler, losses, checkpoint_dir='checkpoints', local_rank=0, enable_logging=False, logger=None):
    """
    Saves a training checkpoint.

    Args:
        epoch (int): Current epoch number.
        batch_idx (int): Current batch index.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler.
        losses (list): List of recorded losses.
        checkpoint_dir (str): Directory to save checkpoints.
        local_rank (int): Rank of the current process.
        enable_logging (bool): Whether logging is enabled.
        logger (logging.Logger): Logger instance.
    """
    if local_rank != 0:
        return  # Only rank 0 saves checkpoints

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pt')

    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'losses': losses
    }

    torch.save(checkpoint, checkpoint_path)
    if enable_logging and logger is not None:
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved checkpoint: {checkpoint_path}")

def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """
    Finds the latest checkpoint file based on epoch and batch index.

    Args:
        checkpoint_dir (str): Directory containing checkpoint files.

    Returns:
        str or None: Path to the latest checkpoint file, or None if none found.
    """
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not checkpoint_files:
        return None
    # Sort files by epoch and batch index
    def extract_epoch_batch(filename):
        # filename: 'checkpoint_epoch_{epoch}_batch_{batch_idx}.pt'
        parts = filename.rstrip('.pt').split('_')
        try:
            epoch = int(parts[2])
            batch = int(parts[4])
            return (epoch, batch)
        except (IndexError, ValueError):
            return (0,0)
    checkpoint_files.sort(key=lambda x: extract_epoch_batch(x), reverse=True)
    latest_checkpoint = checkpoint_files[0]
    return os.path.join(checkpoint_dir, latest_checkpoint)

def load_checkpoint(checkpoint_path, model, optimizer, scaler, losses, device, local_rank, enable_logging, logger):
    """
    Loads the checkpoint from the specified path into the model, optimizer, scaler, and losses.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model.
        optimizer (torch.optim.Optimizer): The optimizer.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler.
        losses (list): The list to append loaded losses.
        device (torch.device): The device to map the checkpoint.
        local_rank (int): Rank of the current process.
        enable_logging (bool): Whether logging is enabled.
        logger (logging.Logger): Logger instance.

    Returns:
        epoch (int): Loaded epoch.
        batch_idx (int): Loaded batch index.
    """
    if local_rank != 0:
        return 0, 0  # Other ranks start from epoch 0, batch 0

    if checkpoint_path is None:
        if enable_logging:
            logger.info("No checkpoint found. Starting training from scratch.")
        print("No checkpoint found. Starting training from scratch.")
        return 0, 0

    if enable_logging:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
    print(f"Loading checkpoint from {checkpoint_path}")

    # Suppress the FutureWarning for torch.load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    losses.extend(checkpoint['losses'])

    epoch = checkpoint['epoch']
    batch_idx = checkpoint['batch_idx']

    if enable_logging:
        logger.info(f"Loaded checkpoint: Epoch {epoch}, Batch {batch_idx}")

    return epoch, batch_idx

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

        # Initialize optimizer and scaler before loading checkpoint
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5)  # 8-bit AdamW
        if enable_logging:
            logger.info("Initialized 8-bit AdamW optimizer")

        scaler = GradScaler()
        if enable_logging:
            logger.info("Initialized GradScaler for mixed precision")

        # Load the latest checkpoint if available
        checkpoint_dir = 'checkpoints'
        latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir)
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
            def __init__(self, dataframe, tokenizer, max_length=128):  # Reduced max_length
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
        dataset = PromptResponseDataset(data, tokenizer)
        sampler = DistributedSampler(dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, pin_memory=True, num_workers=num_workers)
        if enable_logging:
            logger.info(f"Process {local_rank}: Created DataLoader with batch size 1 and num_workers={num_workers}")
        print(f"Process {local_rank}: Created DataLoader with batch size 1 and num_workers={num_workers}")

        # Initialize losses from checkpoint if any
        if latest_checkpoint_path:
            # The 'losses' list has already been loaded via 'load_checkpoint'
            pass
        else:
            losses = []

        # Define gradient accumulation steps
        gradient_accumulation_steps = 4
        accumulated_loss = 0.0  # To track accumulated loss for logging

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Memory profiling
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=lambda p: p.export_chrome_trace(os.path.join('profiler_logs', f'trace_{local_rank}.json')),
            with_stack=True
        ) as prof:

            # Training loop with mixed precision and gradient accumulation
            total_epochs = 2
            model.train()
            for epoch in range(loaded_epoch, total_epochs):
                sampler.set_epoch(epoch)  # Shuffle data differently at each epoch
                if local_rank == 0 and enable_logging:
                    logger.info(f"Epoch {epoch+1}/{total_epochs} started")
                    print(f"Epoch {epoch+1}/{total_epochs}")
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

                for batch_idx, batch in enumerate(progress_bar, start=loaded_batch_idx if epoch == loaded_epoch else 0):
                    optimizer.zero_grad(set_to_none=True)
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    labels = batch['labels'].to(device, non_blocking=True)

                    log_cuda_memory(logger, f"Before forward pass - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)
                    with record_function("forward_pass"):
                        with autocast('cuda', dtype=torch.float16):
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
                            progress_bar.set_postfix({'loss': current_loss})
                            if enable_logging:
                                logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)} - Loss: {current_loss:.4f}")

                        # Save checkpoint every 100 batches
                        if (batch_idx + 1) % 100 == 0:
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
                    prof.step()

                if local_rank == 0:
                    print(f"Epoch {epoch+1} completed. Loss: {loss.item():.4f}")
                    if enable_logging:
                        logger.info(f"Epoch {epoch+1} completed. Loss: {loss.item():.4f}")
                    # Save final checkpoint at the end of the epoch
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

            if local_rank == 0:
                print("Training completed.")
                if enable_logging:
                    logger.info("Training completed.")
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
