import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import os
from typing import Dict, List, Optional, Union
from datasets import load_dataset, concatenate_datasets
import numpy as np

class ContinualDataset(Dataset):
    """
    Dataset for continual learning that mixes new specialized data with 
    The Pile data to prevent catastrophic forgetting
    """
    
    def __init__(
        self,
        new_data_path: str,
        pile_data_path: str,
        tokenizer,
        max_length: int = 512,
        mixing_ratio: float = 0.3,  # Ratio of pile data to include
        split: str = "train",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the continual learning dataset
        
        Args:
            new_data_path: Path to new specialized dataset
            pile_data_path: Path to The Pile dataset
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            mixing_ratio: Ratio of pile data to mix (0.3 = 30% pile, 70% new data)
            split: Dataset split ('train' or 'validation')
            cache_dir: Cache directory for datasets
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mixing_ratio = mixing_ratio
        self.split = split
        
        # Load new specialized dataset
        self.new_data = self._load_new_dataset(new_data_path, split)
        
        # Load pile data for anti-forgetting
        self.pile_data = self._load_pile_dataset(pile_data_path, split, cache_dir)
        
        # Create mixed dataset
        self.mixed_data = self._create_mixed_dataset()
        
        print(f"Created {split} dataset with {len(self.mixed_data)} samples")
        print(f"New data: {len(self.new_data)} samples")
        print(f"Pile data: {len(self.pile_data)} samples")
        print(f"Mixing ratio: {self.mixing_ratio:.2f}")
    
    def _load_new_dataset(self, data_path: str, split: str):
        """Load the new specialized dataset"""
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            return self._load_json_dataset(data_path, split)
        else:
            # Assume it's a Hugging Face dataset
            try:
                dataset = load_dataset(data_path, split=split)
                return dataset
            except Exception as e:
                print(f"Error loading dataset from {data_path}: {e}")
                return []
    
    def _load_json_dataset(self, file_path: str, split: str):
        """Load dataset from JSON/JSONL file"""
        data = []
        
        try:
            if file_path.endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Simple train/validation split if not already split
            if isinstance(data, list):
                split_idx = int(0.9 * len(data))
                if split == "train":
                    return data[:split_idx]
                else:
                    return data[split_idx:]
            
            return data
            
        except Exception as e:
            print(f"Error loading JSON dataset: {e}")
            return []
    
    def _load_pile_dataset(self, pile_path: str, split: str, cache_dir: Optional[str] = None):
        """Load The Pile dataset for anti-forgetting"""
        try:
            # Try to load from local path first
            if os.path.exists(pile_path):
                return self._load_json_dataset(pile_path, split)
            
            # Otherwise, load from Hugging Face Hub
            # Note: The Pile is a large dataset, so we'll sample a subset
            dataset = load_dataset(
                "monology/pile-uncopyrighted", 
                split=split,
                cache_dir=cache_dir,
                streaming=True  # Use streaming for large datasets
            )
            
            # Convert streaming dataset to list (sample subset)
            pile_samples = []
            max_pile_samples = 10000  # Limit pile samples for memory
            
            for i, sample in enumerate(dataset):
                if i >= max_pile_samples:
                    break
                pile_samples.append(sample)
            
            return pile_samples
            
        except Exception as e:
            print(f"Warning: Could not load pile dataset: {e}")
            print("Continuing without pile data (may lead to catastrophic forgetting)")
            return []
    
    def _create_mixed_dataset(self):
        """Create mixed dataset with specified ratio"""
        if not self.pile_data:
            print("No pile data available, using only new data")
            return list(self.new_data)
        
        # Calculate number of samples from each dataset
        total_new = len(self.new_data)
        num_pile_samples = int((self.mixing_ratio / (1 - self.mixing_ratio)) * total_new)
        num_pile_samples = min(num_pile_samples, len(self.pile_data))
        
        # Sample from pile data
        pile_samples = random.sample(self.pile_data, num_pile_samples) if num_pile_samples > 0 else []
        
        # Combine datasets
        mixed_data = list(self.new_data) + pile_samples
        
        # Shuffle the combined dataset
        random.shuffle(mixed_data)
        
        return mixed_data
    
    def _extract_text(self, sample):
        """Extract text from a sample, handling different data formats"""
        if isinstance(sample, dict):
            # Try common text fields
            for field in ['text', 'content', 'body', 'input', 'prompt']:
                if field in sample:
                    return sample[field]
            
            # If it's a conversation format
            if 'messages' in sample:
                return self._format_conversation(sample['messages'])
            
            # If it's instruction format
            if 'instruction' in sample and 'response' in sample:
                instruction = sample.get('instruction', '')
                response = sample.get('response', '')
                return f"Instruction: {instruction}\nResponse: {response}"
            
            # Fallback: concatenate all string values
            text_parts = []
            for key, value in sample.items():
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
            return "\n".join(text_parts)
        
        elif isinstance(sample, str):
            return sample
        
        else:
            return str(sample)
    
    def _format_conversation(self, messages):
        """Format conversation messages into a single text"""
        formatted_text = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted_text += f"{role.capitalize()}: {content}\n"
        return formatted_text.strip()
    
    def __len__(self):
        return len(self.mixed_data)
    
    def __getitem__(self, idx):
        """Get a tokenized sample"""
        sample = self.mixed_data[idx]
        text = self._extract_text(sample)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Prepare labels (for causal language modeling, labels = input_ids)
        labels = encoding['input_ids'].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

class InstructionDataset(Dataset):
    """
    Dataset specifically for instruction-following tasks
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_instruction_data(data_path, split)
    
    def _load_instruction_data(self, data_path: str, split: str):
        """Load instruction dataset"""
        try:
            if data_path.endswith(('.json', '.jsonl')):
                data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    if data_path.endswith('.jsonl'):
                        for line in f:
                            data.append(json.loads(line.strip()))
                    else:
                        data = json.load(f)
                
                # Simple split
                split_idx = int(0.9 * len(data))
                return data[:split_idx] if split == "train" else data[split_idx:]
            else:
                # Load from Hugging Face
                dataset = load_dataset(data_path, split=split)
                return list(dataset)
        except Exception as e:
            print(f"Error loading instruction dataset: {e}")
            return []
    
    def _format_instruction_sample(self, sample):
        """Format instruction sample for training"""
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', sample.get('response', ''))
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        full_text = prompt + output
        
        return prompt, full_text
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt, full_text = self._format_instruction_sample(sample)
        
        # Tokenize full text
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize prompt to find where to start loss calculation
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels - only compute loss on the response part
        labels = full_encoding['input_ids'].clone()
        prompt_length = prompt_encoding['input_ids'].shape[1]
        labels[:, :prompt_length] = -100  # Don't compute loss on prompt
        labels[labels == self.tokenizer.pad_token_id] = -100  # Don't compute loss on padding
        
        return {
            'input_ids': full_encoding['input_ids'].squeeze(),
            'attention_mask': full_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def create_dataloaders(
    new_data_path: str,
    pile_data_path: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    mixing_ratio: float = 0.3,
    num_workers: int = 4
):
    """
    Create train and validation dataloaders
    """
    train_dataset = ContinualDataset(
        new_data_path=new_data_path,
        pile_data_path=pile_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mixing_ratio=mixing_ratio,
        split="train"
    )
    
    val_dataset = ContinualDataset(
        new_data_path=new_data_path,
        pile_data_path=pile_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mixing_ratio=mixing_ratio,
        split="validation"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader