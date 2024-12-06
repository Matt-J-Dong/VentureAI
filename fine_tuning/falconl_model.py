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

def save_checkpoint(epoch, batch_idx, model, optimizer, scaler, losses, checkpoint_dir='checkpoints', local_rank=0, enable_logging=False, logger=None, max_checkpoints=5):
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
        max_checkpoints (int): Maximum number of checkpoints to retain.
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

    # Cleanup old checkpoints
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if len(checkpoint_files) > max_checkpoints:
        # Sort files by epoch and batch index in descending order
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
        # Remove older checkpoints beyond the max_checkpoints limit
        for old_ckpt in checkpoint_files[max_checkpoints:]:
            old_ckpt_path = os.path.join(checkpoint_dir, old_ckpt)
            os.remove(old_ckpt_path)
            if enable_logging and logger is not None:
                logger.info(f"Removed old checkpoint: {old_ckpt_path}")

def find_latest_checkpoint(checkpoint_dir='checkpoints', len_dataloader=None):
    """
    Finds the latest checkpoint file based on epoch and batch index,
    ensuring that the batch index does not exceed the total number of batches.

    Args:
        checkpoint_dir (str): Directory containing checkpoint files.
        len_dataloader (int): Total number of batches in the DataLoader.

    Returns:
        str or None: Path to the latest valid checkpoint file, or None if none found.
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
    # Filter out checkpoint files where batch_idx > len_dataloader
    if len_dataloader is not None:
        valid_checkpoint_files = [f for f in checkpoint_files if extract_epoch_batch(f)[1] <= len_dataloader]
    else:
        valid_checkpoint_files = checkpoint_files
    if not valid_checkpoint_files:
        return None
    # Sort files by epoch and batch index in descending order
    valid_checkpoint_files.sort(key=lambda x: extract_epoch_batch(x), reverse=True)
    latest_checkpoint = valid_checkpoint_files[0]
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
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    losses.extend(checkpoint['losses'])

    epoch = checkpoint['epoch']
    batch_idx = checkpoint['batch_idx']

    if enable_logging:
        logger.info(f"Loaded checkpoint: Epoch {epoch}, Batch {batch_idx}")

    return epoch, batch_idx

def count_trainable_parameters(model):
    """
    Counts the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def debug_dataset(dataloader, tokenizer, num_samples=3, rank=0):
    """
    Saves sample input_ids, attention_mask, and labels to verify dataset correctness.

    Args:
        dataloader (DataLoader): The DataLoader to debug.
        tokenizer (AutoTokenizer): The tokenizer used.
        num_samples (int): Number of samples to save.
        rank (int): Rank of the current process.
    """
    if rank != 0:
        return  # Only process 0 performs debugging

    with open("decoded_input.txt", "w") as f:
        f.write("\n--- Debugging Dataset Samples ---\n\n")
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            f.write(f"Sample {i+1}:\n")
            f.write(f"Input IDs: {input_ids}\n")
            f.write(f"Attention Mask: {attention_mask}\n")
            f.write(f"Labels: {labels}\n\n")
            f.write("Decoded Input:\n")
            f.write(tokenizer.decode(input_ids[i], skip_special_tokens=True) + "\n")
            f.write("Decoded Labels:\n")
            # Replace -100 with pad token for decoding
            decoded_labels = [token if token != -100 else tokenizer.pad_token_id for token in labels[i]]
            f.write(tokenizer.decode(decoded_labels, skip_special_tokens=True) + "\n")
            f.write("\n" + "-"*50 + "\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    args = parser.parse_args()
    local_rank = args.local_rank

    # Define standard boolean flags for logging and profiling
    enable_logging = False  # Set to True to enable logging
    enable_profiling = False  # Set to True to enable profiling

    logger = setup_logging('cuda_memory.txt') if enable_logging else None
    if enable_logging and local_rank == 0:
        logger.info("Starting Training Script with logging enabled")

    try:
        # Initialize the process group
        dist.init_process_group(backend='nccl')
        if enable_logging and local_rank == 0:
            logger.info("Initialized NCCL process group")

        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if enable_logging and local_rank == 0:
            logger.info(f"Process {local_rank}: Initialized on device {device}")

        # Enable cudnn benchmark for optimized performance
        torch.backends.cudnn.benchmark = True
        if enable_logging and local_rank == 0:
            logger.info("Enabled cuDNN benchmark")

        # Load the model and tokenizer
        model_name = "tiiuae/falcon-7b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

        if enable_logging and local_rank == 0:
            logger.info(f"Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
            logger.info(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

        # Configure bitsandbytes for 8-bit loading
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load the model with quantization and explicitly disable cache
        with record_function("model_loading"):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map={"": device},  # Load the entire model on the specified device
                use_cache=False  # **Explicitly disable cache to save memory**
            )
            if local_rank == 0:
                log_cuda_memory(logger, "After model loading and moving to device", enable_logging)

        # **Inspect the Model's Module Names (Only for process 0)**
        if local_rank == 0:
            print("\n--- Model Modules ---")
            for name, module in model.named_modules():
                print(name)
            print("----------------------\n")
            if enable_logging:
                logger.info("\n--- Model Modules ---")
                for name, module in model.named_modules():
                    logger.info(name)
                logger.info("----------------------\n")

        # Configure LoRA with updated target modules based on model's architecture
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query_key_value", "dense"],  # Update based on actual module names
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        if enable_logging and local_rank == 0:
            logger.info(f"Process {local_rank}: Wrapped model with LoRA")

        # Count and log trainable parameters
        if local_rank == 0:
            trainable_params = count_trainable_parameters(model)
            print(f"Number of trainable parameters: {trainable_params}")
            if enable_logging:
                logger.info(f"Number of trainable parameters: {trainable_params}")

        # Enable Gradient Checkpointing using PyTorch's built-in method
        model.gradient_checkpointing_enable()
        if enable_logging and local_rank == 0:
            logger.info("Enabled gradient checkpointing")
        if local_rank == 0:
            print("Enabled gradient checkpointing")

        # **Remove Activation Offloading with FairScale's checkpoint_wrapper**
        # This step was causing gradients to not be tracked properly.
        # If activation offloading is essential, consider alternative methods or ensure compatibility.
        # for name, module in model.named_modules():
        #     if isinstance(module, torch.nn.Linear):
        #         setattr(model, name, checkpoint_wrapper(module))
        # if enable_logging and local_rank == 0:
        #     logger.info("Wrapped Linear layers with FairScale checkpoint_wrapper for activation offloading")
        # if local_rank == 0:
        #     print("Wrapped Linear layers with FairScale checkpoint_wrapper for activation offloading")

        # Wrap the model with DistributedDataParallel with optimized settings
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=False  # **Set to False assuming all parameters are used**
        )
        if enable_logging and local_rank == 0:
            logger.info(f"Process {local_rank}: Wrapped model with DistributedDataParallel with find_unused_parameters=False")
        if local_rank == 0:
            print(f"Process {local_rank}: Wrapped model with DistributedDataParallel with find_unused_parameters=False")

        # Read the data from the CSV file
        data_path = '../data_generation/combined_results.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        data = pd.read_csv(data_path)

        # Load only the first 10% of the dataset for testing purposes
        data = data.head(int(len(data) * 0.1))
        if enable_logging and local_rank == 0:
            logger.info(f"Loaded {len(data)} samples for training")
        if local_rank == 0:
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
                # Concatenate prompt and response with eos_token as separator
                input_text = f"{prompt}{self.tokenizer.eos_token}{response}{self.tokenizer.eos_token}"
                encoding = self.tokenizer(
                    input_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].squeeze()
                attention_mask = encoding['attention_mask'].squeeze()
                
                # Create labels: mask the prompt tokens by setting them to -100
                # so that loss is not computed on them
                # Find the position where the response starts
                try:
                    # Locate the first eos_token_id which separates prompt and response
                    eos_indices = (input_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_indices) < 1:
                        # If no eos_token found, treat entire input as prompt
                        response_start = len(input_ids)
                    else:
                        # Response starts after the first eos_token
                        response_start = eos_indices[0].item()
                except IndexError:
                    # If eos_token not found, treat entire input as prompt
                    response_start = len(input_ids)
                
                labels = torch.full_like(input_ids, fill_value=-100)
                labels[response_start + 1:] = input_ids[response_start + 1:]
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }

        # Create the dataset and DistributedSampler
        batch_size = 4  # Adjust batch size as needed
        num_workers = 4 # Adjust num_workers as needed, 4 num_workers is about 500 MiB memory
        dataset = PromptResponseDataset(data, tokenizer)
        sampler = DistributedSampler(
            dataset, 
            shuffle=True, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank()
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=sampler, 
            pin_memory=True, 
            num_workers=num_workers,
            prefetch_factor=2,
            persistent_workers=True
        )
        if enable_logging and local_rank == 0:
            logger.info(f"Process {local_rank}: Created DataLoader with batch size {batch_size}, num_workers={num_workers}, prefetch_factor=2, persistent_workers=True")
        if local_rank == 0:
            print(f"Process {local_rank}: Created DataLoader with batch size {batch_size}, num_workers={num_workers}, prefetch_factor=2, persistent_workers=True")

        # **Debugging the Dataset** (Only for process 0)
        if local_rank == 0:
            debug_dataset(dataloader, tokenizer, num_samples=3, rank=local_rank)

        # Find the latest checkpoint if available
        len_dataloader = len(dataloader)
        latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir='checkpoints', len_dataloader=len_dataloader)

        # Initialize optimizer and scaler before loading checkpoint
        optimizer = bnb.optim.AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)  # 8-bit AdamW with only trainable params
        if enable_logging and local_rank == 0:
            logger.info("Initialized 8-bit AdamW optimizer")

        scaler = GradScaler()
        if enable_logging and local_rank == 0:
            logger.info("Initialized GradScaler for mixed precision")

        # Load the latest checkpoint if available
        loaded_epoch, loaded_batch_idx = 0, 0
        losses = []
        if latest_checkpoint_path and local_rank == 0:
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
            if enable_logging and local_rank == 0:
                logger.info("No checkpoints found. Starting training from scratch.")
            if local_rank == 0:
                print("No checkpoints found. Starting training from scratch.")

        # Broadcast the loaded_epoch and loaded_batch_idx to all ranks
        loaded_epoch_tensor = torch.tensor(loaded_epoch).to(device)
        loaded_batch_idx_tensor = torch.tensor(loaded_batch_idx).to(device)
        dist.broadcast(loaded_epoch_tensor, src=0)
        dist.broadcast(loaded_batch_idx_tensor, src=0)
        loaded_epoch = loaded_epoch_tensor.item()
        loaded_batch_idx = loaded_batch_idx_tensor.item()

        if enable_logging and local_rank == 0 and loaded_epoch > 0:
            logger.info(f"Resuming from Epoch {loaded_epoch}, Batch {loaded_batch_idx}")
            print(f"Resuming from Epoch {loaded_epoch}, Batch {loaded_batch_idx}")

        # Define gradient accumulation steps
        gradient_accumulation_steps = 4
        accumulated_loss = 0.0  # To track accumulated loss for logging

        # Create checkpoint directory
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize profiler if profiling is enabled
        if enable_profiling and local_rank == 0:
            os.makedirs('profiler_logs', exist_ok=True)
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=lambda p: p.export_chrome_trace(os.path.join('profiler_logs', f'trace_{local_rank}.json')),
                with_stack=True
            )
            profiler.start()
            if enable_logging:
                logger.info("Profiling started")
            print("Profiling started")

        # **Verify That All Model Parameters Require Gradients**
        # This step ensures that gradients are being tracked correctly.
        all_require_grad = True
        for name, param in model.module.named_parameters():
            if not param.requires_grad:
                if local_rank == 0:
                    print(f"WARNING: Parameter '{name}' does not require grad.")
                all_require_grad = False
        if not all_require_grad and enable_logging and local_rank == 0:
            logger.warning("Some parameters do not require gradients.")

        # Initialize 'epoch' before the loop
        epoch = loaded_epoch
        # Training loop with mixed precision and gradient accumulation
        total_epochs = 2  # Adjust total epochs as needed
        model.train()
        for epoch in range(loaded_epoch, total_epochs):
            sampler.set_epoch(epoch)  # Shuffle data differently at each epoch
            if local_rank == 0 and enable_logging:
                logger.info(f"Epoch {epoch+1}/{total_epochs} started")
                print(f"Epoch {epoch+1}/{total_epochs} started")

            # Initialize progress bar for all processes
            progress_bar = tqdm(dataloader, desc=f"Process {local_rank} - Epoch {epoch+1}", leave=False)

            # If resuming in the middle of an epoch, skip the first 'loaded_batch_idx' batches
            if epoch == loaded_epoch and loaded_batch_idx > 0:
                if enable_logging and local_rank == 0:
                    logger.info(f"Skipping first {loaded_batch_idx} batches of Epoch {epoch+1}")
                if local_rank == 0:
                    print(f"Skipping first {loaded_batch_idx} batches of Epoch {epoch+1}")
                dataloader_iter = iter(dataloader)
                for _ in range(loaded_batch_idx):
                    try:
                        next(dataloader_iter)
                    except StopIteration:
                        break
            else:
                dataloader_iter = iter(dataloader)

            # Iterate over batches
            for batch_idx, batch in enumerate(dataloader_iter, start=loaded_batch_idx if epoch == loaded_epoch else 0):
                optimizer.zero_grad(set_to_none=True)
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                if local_rank == 0:
                    log_cuda_memory(logger, f"Before forward pass - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)
                with record_function("forward_pass"):
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss / gradient_accumulation_steps  # Scale loss for accumulation
                if local_rank == 0:
                    log_cuda_memory(logger, f"After forward pass - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)

                with record_function("backward_pass"):
                    scaler.scale(loss).backward()
                    accumulated_loss += loss.item() * gradient_accumulation_steps
                if local_rank == 0:
                    log_cuda_memory(logger, f"After backward pass - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)

                # Update optimizer every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    with record_function("optimizer_step"):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    if local_rank == 0:
                        log_cuda_memory(logger, f"After optimizer step - Epoch {epoch+1}, Batch {batch_idx+1}", enable_logging)

                        current_loss = accumulated_loss / gradient_accumulation_steps
                        losses.append(current_loss)
                        accumulated_loss = 0.0
                        progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})
                        if enable_logging:
                            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)} - Loss: {current_loss:.4f}")

                    # Save checkpoint every 100 batches
                    if (batch_idx + 1) % 100 == 0 and (batch_idx + 1) <= len(dataloader):
                        if local_rank == 0:
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

                # **Free Up Memory by Deleting Unused Variables and Clearing Cache**
                del loss, outputs
                torch.cuda.empty_cache()

                # Step the profiler if profiling is enabled
                if enable_profiling and local_rank == 0:
                    profiler.step()

            # Reset loaded_batch_idx after the first epoch
            loaded_batch_idx = 0

        # Stop the profiler after training loop
        if enable_profiling and local_rank == 0:
            profiler.stop()
            if enable_logging:
                logger.info("Profiling stopped")
            print("Profiling stopped")

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
        if enable_logging and logger is not None and local_rank == 0:
            logger.error(f"CUDA Out of Memory Error: {e}")
        if local_rank == 0:
            print(f"CUDA Out of Memory: {e}")
    except Exception as e:
        if enable_logging and logger is not None and local_rank == 0:
            logger.error(f"An unexpected error occurred: {e}")
        if local_rank == 0:
            print(f"An unexpected error occurred: {e}")
    finally:
        dist.destroy_process_group()
        if enable_logging and logger is not None and local_rank == 0:
            logger.info("Destroyed the distributed process group")

if __name__ == "__main__":
    main()
