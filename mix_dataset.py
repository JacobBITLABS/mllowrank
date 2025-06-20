
import random
import itertools
from torch.utils.data import IterableDataset

class OnlineMixedDataset(IterableDataset):
    def __init__(self, general_iter, domain_data, tokenizer, general_ratio=0.3, max_length=1024):
        self.general_iter = general_iter
        self.domain_data = domain_data
        self.tokenizer = tokenizer
        self.general_ratio = general_ratio
        self.max_length = max_length
        
    def __iter__(self):
        # Create fresh iterators for each epoch
        general_iter = iter(self.general_iter)
        domain_iter = itertools.cycle(self.domain_data)
        
        # Track consecutive failures to avoid infinite loops
        consecutive_failures = 0
        max_failures = 100
        
        while consecutive_failures < max_failures:
            try:
                if random.random() < self.general_ratio:
                    # Sample from general data
                    sample = next(general_iter)
                    source = 'general'
                else:
                    # Sample from domain data
                    sample = next(domain_iter)
                    source = 'domain'
                
                # Tokenize the sample
                tokenized = self.tokenizer(
                    sample['text'], 
                    truncation=True, 
                    padding='max_length', 
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Add source information
                tokenized['source'] = source
                consecutive_failures = 0  # Reset failure counter
                
                yield tokenized
                
            except StopIteration:
                # General iterator exhausted, now only sample from domain
                try:
                    sample = next(domain_iter)
                    tokenized = self.tokenizer(
                        sample['text'], 
                        truncation=True, 
                        padding='max_length', 
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    tokenized['source'] = 'domain'
                    yield tokenized
                except StopIteration:
                    # Both iterators exhausted
                    break
            except Exception as e:
                consecutive_failures += 1
                print(f"Warning: Error processing sample: {e}")
                continue
