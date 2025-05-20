import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import os
import numpy as np
from pathlib import Path
import time
from .data_prep import create_data_loaders
from .seq2seq_model import Seq2SeqTransliteration

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Helper function to save model outputs
def save_predictions(model, test_loader, src_vocab, tgt_vocab, output_path="predictions.tsv"):
    """
    Generate predictions using the model and save them to a file.
    """
    # Create a reverse mapping of indices to characters
    idx_to_char = {idx: char for char, idx in tgt_vocab.items()}
    idx_to_char_src = {idx: char for char, idx in src_vocab.items()}
    
    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = []
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for batch in test_loader:
            src = batch['src'].to(model.device)
            tgt = batch['tgt'].to(model.device)
            src_lengths = batch['src_len']
            
            # Use beam search if configured
            if model.beam_size > 1:
                outputs = model.beam_search_decode(src, src_lengths)
            else:
                outputs = model(src, src_lengths)
                outputs = outputs.argmax(-1)
            
            # Convert indices to characters
            for i in range(outputs.shape[0]):
                pred_seq = outputs[i].cpu().tolist()
                target_seq = tgt[i].cpu().tolist()
                src_seq = src[i].cpu().tolist()
                
                # Convert to characters and stop at end token
                pred_chars = []
                for idx in pred_seq:
                    char = idx_to_char.get(idx, '<unk>')
                    if char == '\n':  # End token
                        break
                    if char != '<pad>' and char != '<unk>' and char != '\t':  # Skip special tokens
                        pred_chars.append(char)
                
                # Do the same for ground truth
                target_chars = []
                for idx in target_seq:
                    char = idx_to_char.get(idx, '<unk>')
                    if char == '\n':  # End token
                        break
                    if char != '<pad>' and char != '<unk>' and char != '\t':  # Skip special tokens
                        target_chars.append(char)
                
                src_chars = []
                for idx in src_seq:
                    char = idx_to_char_src.get(idx, '<unk>')
                    if char == '\n':  # End token
                        break
                    if char != '<pad>' and char != '<unk>' and char != '\t':  # Skip special tokens
                        src_chars.append(char)
                
                predictions.append(''.join(pred_chars))
                ground_truth.append(''.join(target_chars))
                inputs.append(''.join(src_chars))
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("prediction\tground_truth\n")
        for inp, pred, gt in zip(inputs, predictions, ground_truth):
            f.write(f"{inp}\t{pred}\t{gt}\n")
    
    print(f"Saved predictions to {output_path}")
    
    return [[inp, pred, gt] for (inp, pred, gt) in zip(inputs, predictions, ground_truth)]

# Main sweep function
def run_wandb_sweep(train_lines, val_lines, test_lines, num_runs=1, cont_id=None):
    # Initialize wandb
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {'name': 'val_char_acc', 'goal': 'maximize'},
        'parameters': {
            'embedding_dim': {'values': [16, 32, 64, 128]},
            'hidden_size': {'values': [32, 64, 128, 256]},
            'encoder_layers': {'values': [1, 2, 3]},
            'decoder_layers': {'values': [1, 2, 3]},
            'cell_type': {'values': ['gru', 'lstm']},  # Skip simple RNN as GRU/LSTM usually perform better
            'dropout': {'values': [0.1, 0.2, 0.3]},
            'learning_rate': {'values': [0.001, 0.0005, 0.0001]}
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            's': 2,
        }
    }
    
    # Create data loaders once to avoid recreating them for each run
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_data_loaders(
        train_lines,
        batch_size=32,
        min_freq=1,
        val_lines=val_lines,
        test_lines=test_lines
    )

    if cont_id is not None:
        sweep_id = cont_id
    else:
        sweep_id = wandb.sweep(sweep_config, project="hindi-transliteration")
    
    # Define training function for each sweep run
    def train():
        # Initialize wandb run
        run = wandb.init()
        
        # Get hyperparameters for this run
        config = wandb.config
        
        run.name = f'{time.strftime("%Y-%m-%d %H:%M:%S")}'
        
        num_epochs = 15
        
        # Initialize model with sweep config
        model = Seq2SeqTransliteration(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.encoder_layers,
            dropout=config.dropout,
            cell_type=config.cell_type,
            learning_rate=config.learning_rate
        )
        
        # Log model complexity metrics
        param_count = model.compute_parameters()
        # Log example computation count with standard sequence length T=10
        comp_count = model.compute_computations(T=10, m=config.embedding_dim, 
                                              k=config.hidden_size, V=len(src_vocab))
        
        wandb.log({
            "parameter_count": param_count,
            "computation_count": comp_count,
            "Epochs": list(range(num_epochs))
        })
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            devices=1 if torch.cuda.is_available() else None,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=WandbLogger(log_model=True),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, mode='min'),
                ModelCheckpoint(monitor='val_char_acc', mode='max', save_top_k=1),
                LearningRateMonitor(logging_interval='epoch')
            ],
            gradient_clip_val=1.0
        )
        
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        # Test model and log results
        results = trainer.test(model, test_loader)[0]
        for key, value in results.items():
            wandb.log({f"test_{key}": value})
        
        # Save the best model's predictions
        if trainer.checkpoint_callback.best_model_path:
            # Load the best checkpoint
            best_model = Seq2SeqTransliteration.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )
            # Save predictions
            save_predictions(best_model, test_loader, src_vocab, tgt_vocab, 
                            f"./data/predictions_{run.id}.tsv")
    
    # Run the sweep
    wandb.agent(sweep_id, train, count=num_runs)  # Run up to 15 experiments
