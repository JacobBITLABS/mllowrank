import os
import argparse
import torch
import wandb
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    get_linear_schedule_with_warmup
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from mix_dataset import OnlineMixedDataset
from engine import train_one_epoch, validate
import json

def setup_model_and_tokenizer(model_name, lora_config, accelerator):
    """Setup Gemma model with PEFT configuration"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_8bit=True,  # Use 8-bit quantization
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters only on main process
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    
    return model, tokenizer

def setup_optimizer_and_scheduler(model, lr, num_training_steps, warmup_steps):
    """Setup optimizer and learning rate scheduler"""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler

def main(args):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # mixed_precision='fp16',  # Use mixed precision for better performance
        log_with="wandb" if accelerator.is_main_process else None,
        project_dir=args.output_dir
    )
    
    # Initialize wandb only on main process
    if accelerator.is_main_process:
        wandb.init(
            project="gemma-continual-training",
            name=f"gemma-{args.experiment_name}",
            config=vars(args)
        )
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # train 
    train_commonpile = load_dataset("EleutherAI/pile", split="train", streaming=True)
    train_domain = load_dataset("your_dataset", split="train")
    # valid
    valid_commonpile = load_dataset("EleutherAI/pile", split="valid", streaming=True)
    valid_domain = load_dataset("your_dataset", split="valid")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, lora_config, accelerator)
    
    train_dataset = OnlineMixedDataset(
        general_iter=train_commonpile,
        domain_data=train_domain,
        tokenizer=tokenizer,
        general_ratio=0.3,
        stream_domain=False  # or True if streaming domain too
    )

    valid_dataset = OnlineMixedDataset(
        general_iter=valid_commonpile,
        domain_data=valid_domain,
        tokenizer=tokenizer,
        general_ratio=0.3,
        stream_domain=False  # or True if streaming domain too
    )

    # # Setup streaming datasets
    # train_dataset = ContinualStreamingDataset(
    #     new_data_path=args.new_data_path,
    #     pile_dataset_name=args.pile_dataset_name,
    #     tokenizer=tokenizer,
    #     max_length=args.max_length,
    #     mixing_ratio=args.mixing_ratio,
    #     split="train",
    #     max_samples=args.max_train_samples,
    #     seed=args.seed
    # )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    num_training_steps = (len(train_dataset) // (args.batch_size * accelerator.num_processes)) * args.num_epochs
    warmup_steps = int(0.1 * num_training_steps)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
            print("-" * 50)
        
        train_loss, train_perplexity = train_one_epoch(
            model, train_loader, optimizer, scheduler, accelerator
        )
        val_loss, val_perplexity = validate(model, val_loader, accelerator)
        
        # Log metrics (only on main process)
        if accelerator.is_main_process:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_perplexity": train_perplexity,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_perplexity:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_perplexity:.4f}")
        
        # Save best model (only on main process)
        if accelerator.is_main_process and val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Unwrap model for saving
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(f"{args.output_dir}/best_model")
            tokenizer.save_pretrained(f"{args.output_dir}/best_model")
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Save checkpoint every few epochs (only on main process)
        if accelerator.is_main_process and (epoch + 1) % args.save_every == 0:
            checkpoint_dir = f"{args.output_dir}/checkpoint-epoch-{epoch+1}"
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        
        # Wait for all processes to finish epoch
        accelerator.wait_for_everyone()
    
    # Save final model (only on main process)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(f"{args.output_dir}/final_model")
        tokenizer.save_pretrained(f"{args.output_dir}/final_model")
        
        wandb.finish()
        print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Training for Gemma")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="google/gemma-2b", help="Pretrained model name")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Data arguments
    parser.add_argument("--new_data_path", type=str, required=True, help="Path to new specialized dataset")
    parser.add_argument("--pile_dataset_name", type=str, default="common-pile/common_pile_2024-05", help="Common Pile dataset name from HuggingFace")
    parser.add_argument("--mixing_ratio", type=float, default=0.3, help="Ratio of pile data to mix (0.3 = 30% pile, 70% new data)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_train_samples", type=int, default=100000, help="Maximum number of training samples to use")
    parser.add_argument("--max_val_samples", type=int, default=5000, help="Maximum number of validation samples to use")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for models")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--experiment_name", type=str, default="continual_training", help="Experiment name for wandb")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)