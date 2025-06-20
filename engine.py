import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

def train_one_epoch(model, dataloader, optimizer, scheduler, accelerator):
    """
    Train the model for one epoch using Accelerate
    
    Args:
        model: The PEFT model to train
        dataloader: Training data loader (already prepared by accelerator)
        optimizer: Optimizer (already prepared by accelerator)
        scheduler: Learning rate scheduler (already prepared by accelerator)
        accelerator: Accelerate accelerator object
    
    Returns:
        avg_loss: Average training loss for the epoch
        avg_perplexity: Average perplexity for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Only show progress bar on main process
    if accelerator.is_main_process:
        progress_bar = tqdm(dataloader, desc="Training")
    else:
        progress_bar = dataloader
    
    for step, batch in enumerate(progress_bar):
        # Forward pass - no need to move to device, accelerator handles it
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Backward pass with accelerator
        accelerator.backward(loss)
        
        # Update weights - accelerator handles gradient accumulation
        if accelerator.sync_gradients:
            # Clip gradients to prevent exploding gradients
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Gather loss from all processes
        loss_gathered = accelerator.gather(loss.repeat(batch['input_ids'].shape[0]))
        total_loss += loss_gathered.mean().item()
        num_batches += 1
        
        # Update progress bar (only on main process)
        if accelerator.is_main_process:
            current_loss = total_loss / num_batches
            current_perplexity = torch.exp(torch.tensor(current_loss)).item()
            
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'ppl': f'{current_perplexity:.2f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log step-level metrics to wandb
            if step % 100 == 0:
                wandb.log({
                    "step_loss": loss.item(),
                    "step_perplexity": torch.exp(loss).item(),
                    "step": step + num_batches * accelerator.process_index,
                })
    
    avg_loss = total_loss / num_batches
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, avg_perplexity

def validate(model, dataloader, accelerator):
    """
    Validate the model using Accelerate
    
    Args:
        model: The PEFT model to validate
        dataloader: Validation data loader (already prepared by accelerator)
        accelerator: Accelerate accelerator object
    
    Returns:
        avg_loss: Average validation loss
        avg_perplexity: Average perplexity
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Only show progress bar on main process
    if accelerator.is_main_process:
        progress_bar = tqdm(dataloader, desc="Validation")
    else:
        progress_bar = dataloader
    
    with torch.no_grad():
        for batch in progress_bar:
            # Forward pass - no need to move to device, accelerator handles it
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            
            # Gather loss from all processes
            loss_gathered = accelerator.gather(loss.repeat(batch['input_ids'].shape[0]))
            total_loss += loss_gathered.mean().item()
            num_batches += 1
            
            # Update progress bar (only on main process)
            if accelerator.is_main_process:
                current_loss = total_loss / num_batches
                current_perplexity = torch.exp(torch.tensor(current_loss)).item()
                
                progress_bar.set_postfix({
                    'val_loss': f'{current_loss:.4f}',
                    'val_ppl': f'{current_perplexity:.2f}'
                })
    
    avg_loss = total_loss / num_batches
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, avg_perplexity

def evaluate_on_tasks(model, tokenizer, evaluation_tasks, device, max_length=512):
    """
    Evaluate model on specific downstream tasks
    
    Args:
        model: The PEFT model to evaluate
        tokenizer: Tokenizer
        evaluation_tasks: List of evaluation task datasets
        device: Device to run evaluation on
        max_length: Maximum sequence length
    
    Returns:
        results: Dictionary with evaluation results
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        for task_name, task_data in evaluation_tasks.items():
            task_losses = []
            
            for sample in tqdm(task_data, desc=f"Evaluating {task_name}"):
                # Tokenize input
                inputs = tokenizer(
                    sample['text'],
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs['labels'] = inputs['input_ids'].clone()
                
                # Forward pass
                outputs = model(**inputs)
                task_losses.append(outputs.loss.item())
            
            avg_task_loss = sum(task_losses) / len(task_losses)
            avg_task_ppl = torch.exp(torch.tensor(avg_task_loss)).item()
            
            results[task_name] = {
                'loss': avg_task_loss,
                'perplexity': avg_task_ppl
            }
            
            # Log to wandb
            wandb.log({
                f"{task_name}_loss": avg_task_loss,
                f"{task_name}_perplexity": avg_task_ppl
            })
    
    return results

def generate_samples(model, tokenizer, prompts, device, max_new_tokens=100, temperature=0.7):
    """
    Generate text samples for qualitative evaluation
    
    Args:
        model: The PEFT model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        device: Device to run generation on
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
    
    Returns:
        generated_texts: List of generated text samples
    """
    model.eval()
    generated_texts = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Tokenize prompt
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode generated text
            generated_text = tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            generated_texts.append({
                'prompt': prompt,
                'generated': generated_text[len(prompt):].strip()
            })
    
    return generated_texts