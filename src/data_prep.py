from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import os

class TransliterationDataset(Dataset):
    def __init__(self, lines, src_vocab, tgt_vocab, max_src_len=None, max_tgt_len=None, greedy=False):
        """
        Initialize dataset for Hindi-to-Latin transliteration.
        
        Args:
            lines: List of strings in format 'hindi_text\tlatin_text\tcount'
            src_vocab: Source vocabulary (Latin)
            tgt_vocab: Target vocabulary (Hindi)
            max_src_len: Maximum source sequence length (will pad/truncate to this)
            max_tgt_len: Maximum target sequence length (will pad/truncate to this)
        """
        import torch
        
        self.samples = []
        self.weights = []
        temp_dict = dict()
        
        # Process the lines and extract pairs with weights
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
                
            tgt_text, src_text, count = parts
            # Add start and end tokens to target
            tgt_text_with_tokens = '\t' + tgt_text + '\n'
            
            self.samples.append((src_text, tgt_text_with_tokens))
            self.weights.append(int(count))
            
            if greedy:
                if (src_text) not in temp_dict or temp_dict[(src_text)][1] < count:
                    temp_dict[(src_text)] = [tgt_text_with_tokens, count]
            
        # Determine max lengths if not specified
        if max_src_len is None:
            self.max_src_len = max(len(src) for src, _ in self.samples)
        else:
            self.max_src_len = max_src_len
            
        if max_tgt_len is None:
            self.max_tgt_len = max(len(tgt) for _, tgt in self.samples)
        else:
            self.max_tgt_len = max_tgt_len
            
        if greedy:
            self.samples = [(k, v[0]) for k, v in temp_dict.items()]
        
            
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Create sampling probabilities based on attestation weights
        total_weight = sum(self.weights)
        self.sample_probs = [w / total_weight for w in self.weights]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        import torch
        src_text, tgt_text = self.samples[idx]
        
        # Convert source text to tensor
        src_indices = [self.src_vocab.get(char, self.src_vocab['<unk>']) for char in src_text]
        src_indices = src_indices[:self.max_src_len]  # Truncate if needed
        src_len = len(src_indices)
        # Pad if needed
        if len(src_indices) < self.max_src_len:
            src_indices += [self.src_vocab['<pad>']] * (self.max_src_len - len(src_indices))
        
        # Convert target text to tensor
        tgt_indices = [self.tgt_vocab.get(char, self.tgt_vocab['<unk>']) for char in tgt_text]
        tgt_indices = tgt_indices[:self.max_tgt_len]  # Truncate if needed
        tgt_len = len(tgt_indices)
        # Pad if needed
        if len(tgt_indices) < self.max_tgt_len:
            tgt_indices += [self.tgt_vocab['<pad>']] * (self.max_tgt_len - len(tgt_indices))
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
            'src_len': src_len,
            'tgt_len': tgt_len,
            'weight': self.weights[idx]
        }
    
    def weighted_sampler(self, batch_size=32):
        """
        Returns a weighted sampler that can be used with DataLoader to 
        sample according to attestation counts.
        """
        from torch.utils.data import WeightedRandomSampler
        
        # Create sampler for weighted sampling based on attestations
        sampler = WeightedRandomSampler(
            weights=self.sample_probs,
            num_samples=len(self.samples),
            replacement=True
        )
        
        return sampler


def build_vocab(lines, min_freq=1, special_tokens=None):
    """
    Build vocabulary from the data.
    
    Args:
        lines: List of strings in format 'hindi_text\tlatin_text\tcount'
        min_freq: Minimum frequency for a character to be included
        special_tokens: Dictionary of special tokens to add (e.g., {'<pad>': 0, '<unk>': 1})
    
    Returns:
        src_vocab: Source vocabulary (Latin)
        tgt_vocab: Target vocabulary (Hindi)
    """
    if special_tokens is None:
        special_tokens = {'<pad>': 0, '<unk>': 1}
    
    src_counter = Counter()
    tgt_counter = Counter()
    
    # Count character frequencies
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
            
        tgt_text, src_text, count = parts
        count = int(count)
        
        # Add tab and newline to target vocab (for start/end tokens)
        tgt_text_with_tokens = '\t' + tgt_text + '\n'
        
        for char in src_text:
            src_counter[char] += count
        
        for char in tgt_text_with_tokens:
            tgt_counter[char] += count
    
    # Build source vocabulary
    src_vocab = {token: idx for idx, token in enumerate(special_tokens.keys())}
    idx = len(src_vocab)
    for char, freq in src_counter.items():
        if freq >= min_freq:
            src_vocab[char] = idx
            idx += 1
    
    # Build target vocabulary
    tgt_vocab = {token: idx for idx, token in enumerate(special_tokens.keys())}
    idx = len(tgt_vocab)
    for char, freq in tgt_counter.items():
        if freq >= min_freq:
            tgt_vocab[char] = idx
            idx += 1
    
    return src_vocab, tgt_vocab


def create_data_loaders(train_lines, val_lines, test_lines, batch_size=32, min_freq=1,
                        max_src_len=None, max_tgt_len=None, use_weighted_sampling=True):
    """
    Create data loaders for training, validation, and test sets.
    
    Args:
        lines: List of strings in format 'hindi_text\tlatin_text\tcount'
        batch_size: Batch size for data loaders
        min_freq: Minimum frequency for a character to be included in vocabulary
        val_split: Fraction of data to use for validation (if val_lines not provided)
        test_split: Fraction of data to use for testing (if test_lines not provided)
        max_src_len: Maximum source sequence length
        max_tgt_len: Maximum target sequence length
        use_weighted_sampling: Whether to use weighted sampling based on attestation counts
        val_lines: Optional validation data lines (if None, will use val_split of lines)
        test_lines: Optional test data lines (if None, will use test_split of lines)
    
    Returns:
        train_loader, val_loader, test_loader: Data loaders
        src_vocab, tgt_vocab: Source and target vocabularies
    """
    # Build vocabularies from all available data
    all_lines = train_lines.copy()
    all_lines.extend(val_lines)
    all_lines.extend(test_lines)
        
    special_tokens = {'<pad>': 0, '<unk>': 1}
    src_vocab, tgt_vocab = build_vocab(all_lines, min_freq, special_tokens)
    
    # Create datasets
    train_dataset = TransliterationDataset(train_lines, src_vocab, tgt_vocab, max_src_len, max_tgt_len)
    val_dataset = TransliterationDataset(val_lines, src_vocab, tgt_vocab, max_src_len, max_tgt_len, greedy=True)
    test_dataset = TransliterationDataset(test_lines, src_vocab, tgt_vocab, max_src_len, max_tgt_len, greedy=True)
    
    # Create data loaders
    train_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': not use_weighted_sampling,  # Don't shuffle if using weighted sampler
        'num_workers': 7,  # Set to 0 to avoid multiprocessing issues
        'pin_memory': True,
    }
    
    if use_weighted_sampling:
        train_loader_kwargs['sampler'] = train_dataset.weighted_sampler()
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab

def load_dakshina_data(base_path_data = f"./dataset/dakshina_dataset_v1.0", verbose=False):
    """
    Load data from the Dakshina dataset for the specified language.
    Default is Hindi (hi).
    
    Format expected: native_text\tlatin_text\tcount
    """
    
    if verbose:
        print(os.listdir(base_path_data))
    
    # Check if the paths exist
    if not os.path.exists(base_path_data):
        raise FileNotFoundError(f"Dakshina dataset not found at {base_path_data}")
    
    # Load train, dev, test files
    train_path = os.path.join(base_path_data, f"hi/lexicons/hi.translit.sampled.train.tsv")
    dev_path = os.path.join(base_path_data, f"hi/lexicons/hi.translit.sampled.dev.tsv")
    test_path = os.path.join(base_path_data, f"hi/lexicons/hi.translit.sampled.test.tsv")
    
    # Convert to our expected format
    def process_file(filepath):
        lines = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # The Dakshina dataset has format: native\tlatin
                # We'll add a count of 1 to match our expected format
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    native, latin, count = parts
                    lines.append(f"{latin}\t{native}\t{count}")  # Note the reversal: source is Latin, target is native
        return lines
    
    train_lines = process_file(train_path)
    dev_lines = process_file(dev_path)
    test_lines = process_file(test_path)
    
    print(f"Loaded {len(train_lines)} training examples")
    print(f"Loaded {len(dev_lines)} validation examples")
    print(f"Loaded {len(test_lines)} test examples")
    
    return train_lines, dev_lines, test_lines

# Example usage
if __name__ == "__main__":
    # Example data
    train_lines, val_lines, test_lines = load_dakshina_data(base_path_data=f"../dataset/dakshina_dataset_v1.0")
    
    # Create data loaders
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_data_loaders(
        train_lines, 
        val_lines, 
        test_lines, 
        batch_size=2,
        min_freq=1,
        val_split=0.2,
        test_split=0.2
    )
    
    # Check vocabulary
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Check a batch
    for batch in train_loader:
        print("Batch:")
        print(f"Source shape: {batch['src'].shape}")
        print(f"Target shape: {batch['tgt'].shape}")
        print(f"Source lengths: {batch['src_len']}")
        print(f"Target lengths: {batch['tgt_len']}")
        print(f"Weights: {batch['weight']}")
        break