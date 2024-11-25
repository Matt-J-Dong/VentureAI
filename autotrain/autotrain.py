from autotrain import AutoTrain
from datasets import load_dataset

# Load your dataset
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'validation.csv'})

# Initialize AutoTrain
autotrain = AutoTrain(
    project_name="falcon-40b-finetune",
    task="text-generation",
    model="tiiuae/falcon-40b",
    train_data=dataset['train'],
    eval_data=dataset['validation'],
    output_dir="./finetuned-falcon-40b",
    max_length=1024,
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Adjusted per GPU batch size
    gradient_accumulation_steps=8,
    fp16=True,
    use_peft=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    distributed=True,  # Enable distributed training
)

# Start training
autotrain.train()

