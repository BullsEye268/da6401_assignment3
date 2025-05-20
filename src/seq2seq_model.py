import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Union


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        cell_type: str = "gru",
    ):
        """
        Encoder for sequence-to-sequence model.
        
        Args:
            input_size: Size of the vocabulary
            embedding_dim: Size of the character embeddings
            hidden_size: Size of the hidden state
            num_layers: Number of RNN layers
            dropout: Dropout probability
            cell_type: Type of RNN cell (rnn, lstm, gru)
        """
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        # Select RNN type
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                embedding_dim, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # Default to simple RNN
            self.rnn = nn.RNN(
                embedding_dim, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple:
        """
        Forward pass through the encoder.
        
        Args:
            src: Source sequence tensor [batch_size, seq_len]
            src_lengths: Length of each sequence in batch
            
        Returns:
            outputs: RNN outputs for all timesteps
            hidden: Final hidden state (and cell state for LSTM)
        """
        # Apply embedding and dropout
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, embedding_dim]
        
        # Pack padded sequence for more efficient computation
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            src_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Pass through RNN
        outputs, hidden = self.rnn(packed)
        
        # Unpack the sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(
        self,
        output_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        cell_type: str = "gru",
    ):
        """
        Decoder for sequence-to-sequence model.
        
        Args:
            output_size: Size of the target vocabulary
            embedding_dim: Size of the character embeddings
            hidden_size: Size of the hidden state
            num_layers: Number of RNN layers
            dropout: Dropout probability
            cell_type: Type of RNN cell (rnn, lstm, gru)
        """
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type.lower()
        
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # Select RNN type
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                embedding_dim, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # Default to simple RNN
            self.rnn = nn.RNN(
                embedding_dim, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input: torch.Tensor, hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple:
        """
        Forward pass for a single decoder step.
        
        Args:
            input: Input tensor [batch_size, 1]
            hidden: Hidden state from encoder or previous decoder step
            
        Returns:
            output: Prediction for next character in sequence
            hidden: Updated hidden state
        """
        # Apply embedding and dropout [batch_size, 1, embedding_dim]
        embedded = self.dropout(self.embedding(input))
        
        # Pass through RNN
        output, hidden = self.rnn(embedded, hidden)
        
        # Project to vocabulary size
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_size]
        
        return prediction, hidden


class Seq2SeqTransliteration(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        dropout: float = 0.1,
        cell_type: str = "gru",
        learning_rate: float = 0.001,
        pad_idx: int = 0,
        teacher_forcing_ratio: float = 0.5,
        beam_size: int = 1,  # 1 means greedy decoding
    ):
        """
        Sequence-to-sequence model for transliteration using PyTorch Lightning.
        
        Args:
            src_vocab_size: Size of the source vocabulary
            tgt_vocab_size: Size of the target vocabulary
            embedding_dim: Size of the character embeddings
            hidden_size: Size of RNN hidden states
            encoder_layers: Number of encoder RNN layers
            decoder_layers: Number of decoder RNN layers
            dropout: Dropout probability
            cell_type: Type of RNN cell (rnn, lstm, gru)
            learning_rate: Initial learning rate
            pad_idx: Index of padding token
            teacher_forcing_ratio: Probability of using teacher forcing during training
            beam_size: Beam size for beam search decoding (1 for greedy)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Model components
        self.encoder = Encoder(
            input_size=src_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            dropout=dropout,
            cell_type=cell_type,
        )
        
        self.decoder = Decoder(
            output_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            dropout=dropout,
            cell_type=cell_type,
        )
        
        # Training parameters
        self.learning_rate = learning_rate
        self.pad_idx = pad_idx
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam_size = beam_size
        
        # Metrics
        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()
        # self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.hparams.tgt_vocab_size,
            ignore_index=self.hparams.pad_idx,
            # top_k=1 (default, usually what you want for simple accuracy)
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.hparams.tgt_vocab_size,
            ignore_index=self.hparams.pad_idx,
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.hparams.tgt_vocab_size,
            ignore_index=self.hparams.pad_idx,
        )
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_lengths: torch.Tensor, 
        tgt: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass through the seq2seq model.
        
        Args:
            src: Source sequence tensor [batch_size, src_len]
            src_lengths: Length of each source sequence
            tgt: Target sequence tensor (for teacher forcing) [batch_size, tgt_len]
            max_len: Maximum decoding length (if tgt is None)
            
        Returns:
            outputs: Model predictions [batch_size, tgt_len, tgt_vocab_size]
        """
        batch_size = src.shape[0]
        
        # Encode the source sequence
        _, hidden = self.encoder(src, src_lengths)
        
        # Determine target sequence length for decoding
        if tgt is not None:
            max_len = tgt.shape[1]
        else:
            if max_len is None:
                # Default to twice the maximum source length if not specified
                max_len = src.shape[1] * 2
        
        # Initialize decoder input with start token (tab character)
        # Assuming <tab> is at index 2 in target vocabulary
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=self.device) * 2
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, max_len, self.decoder.output_size, device=self.device)
        
        # Decide decoding strategy
        if tgt is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
            # Teacher forcing: use ground truth as input to decoder at each step
            for t in range(max_len):
                # Pass through decoder for current timestep
                output, hidden = self.decoder(decoder_input, hidden)
                outputs[:, t] = output

                # Next input is current target token
                decoder_input = tgt[:, t].unsqueeze(1)
        else:
            # No teacher forcing: use own predictions as the next input
            for t in range(max_len):
                # Pass through decoder for current timestep
                output, hidden = self.decoder(decoder_input, hidden)
                outputs[:, t] = output
                
                # Get the most likely next token
                top1 = output.argmax(1).unsqueeze(1)
                decoder_input = top1
                
                # Stop if all sequences have predicted the end token
                # Assuming <newline> is the end token
                if t > 0 and (top1 == 3).all():  # 3 is the index for <newline>
                    break
        
        return outputs
    
    def beam_search_decode(self, src: torch.Tensor, src_lengths: torch.Tensor, max_len: int = 100) -> List[List[int]]:
        """
        Perform beam search decoding.
        
        Args:
            src: Source sequence tensor [batch_size, src_len]
            src_lengths: Length of each source sequence
            max_len: Maximum decoding length
            
        Returns:
            best_sequences: List of best decoded sequences for each example in batch
        """
        if self.beam_size <= 1:
            # Fall back to greedy decoding
            outputs = self.forward(src, src_lengths, max_len=max_len)
            preds = outputs.argmax(-1)
            return preds
        
        batch_size = src.shape[0]
        
        # Encode the source sequence
        _, encoder_hidden = self.encoder(src, src_lengths)
        
        # Store the final results
        best_sequences = []
        
        # Process each item in batch separately
        for b in range(batch_size):
            # Get encoder hidden state for this example
            if self.encoder.cell_type == "lstm":
                # For LSTM, hidden state is a tuple (hidden, cell)
                decoder_hidden = (
                    encoder_hidden[0][:, b:b+1].contiguous(),
                    encoder_hidden[1][:, b:b+1].contiguous()
                )
            else:
                # For GRU and RNN, hidden state is just hidden
                decoder_hidden = encoder_hidden[:, b:b+1].contiguous()
            
            # Start with a beam containing just the start token
            # Beam format: (sequence, score, hidden_state)
            beam = [(
                [2],  # Start with tab token
                0,    # Initial log probability is 0
                decoder_hidden
            )]
            
            # Beam search loop
            for _ in range(max_len):
                candidates = []
                
                # Process all sequences in current beam
                for seq, score, hidden in beam:
                    # Check if the sequence has ended
                    if seq[-1] == 3:  # End token (newline)
                        candidates.append((seq, score, hidden))
                        continue
                    
                    # Prepare decoder input (last token)
                    decoder_input = torch.tensor(
                        [[seq[-1]]], 
                        dtype=torch.long, 
                        device=self.device
                    )
                    
                    # Get next token predictions
                    output, new_hidden = self.decoder(decoder_input, hidden)
                    
                    # Get top k tokens
                    log_probs = F.log_softmax(output, dim=-1)
                    topk_probs, topk_idx = log_probs.topk(self.beam_size)
                    
                    # Create new candidates
                    for i in range(self.beam_size):
                        new_seq = seq + [topk_idx[0, i].item()]
                        new_score = score + topk_probs[0, i].item()
                        candidates.append((new_seq, new_score, new_hidden))
                
                # Select top-beam_size candidates
                beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_size]
                
                # Check if all beams ended with the end token
                if all(seq[-1] == 3 for seq, _, _ in beam):
                    break
            
            # Add the best sequence to the result
            best_sequences.append(beam[0][0])
        
        # Convert to tensor with padding
        max_seq_len = max(len(seq) for seq in best_sequences)
        result = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)
        for i, seq in enumerate(best_sequences):
            result[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
        
        return result
    
    def training_step(self, batch, batch_idx):
        """Training step for Lightning"""
        src, tgt = batch['src'], batch['tgt']
        src_lengths, tgt_lengths = batch['src_len'], batch['tgt_len']
        
        # Forward pass
        outputs = self(src, src_lengths, tgt)
        
        # Calculate loss (ignoring padding)
        tgt_out = tgt[:, 1:]  # Shift right to get targets (skip start token)
        outputs = outputs[:, :-1]  # Remove last output (we don't need to predict after the end)
        
        # Flatten for CrossEntropyLoss
        outputs = outputs.reshape(-1, outputs.shape[-1])
        tgt_out = tgt_out.reshape(-1)
        
        # Calculate loss (ignoring padding)
        loss = F.cross_entropy(outputs, tgt_out, ignore_index=self.pad_idx)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy
        preds = outputs.argmax(-1)
        mask = tgt_out != self.pad_idx
        self.train_acc(preds[mask], tgt_out[mask])
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning"""
        src, tgt = batch['src'], batch['tgt']
        src_lengths, tgt_lengths = batch['src_len'], batch['tgt_len']
        
        # Forward pass (no teacher forcing)
        outputs = self(src, src_lengths, tgt)
        
        # Calculate loss
        tgt_out = tgt[:, 1:]  # Shift right
        outputs = outputs[:, :-1]  # Remove last output
        
        # Flatten
        outputs_flat = outputs.reshape(-1, outputs.shape[-1])
        tgt_out_flat = tgt_out.reshape(-1)
        
        # Calculate loss (ignoring padding)
        loss = F.cross_entropy(outputs_flat, tgt_out_flat, ignore_index=self.pad_idx)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        
        # Calculate accuracy
        preds_flat = outputs_flat.argmax(-1)
        mask = tgt_out_flat != self.pad_idx
        self.val_acc(preds_flat[mask], tgt_out_flat[mask])
        self.log('val_acc', self.val_acc, prog_bar=True)
        
        # Get character-level accuracy
        preds = outputs.argmax(-1)
        total_correct = 0
        total_chars = 0
        
        # Count correct characters per sequence (ignoring padding)
        for i in range(len(src)):
            seq_len = tgt_lengths[i] - 1  # Subtract 1 to exclude the start token
            pred_seq = preds[i, :seq_len]
            target_seq = tgt_out[i, :seq_len]
            
            # Count correct predictions
            correct = (pred_seq == target_seq).sum().item()
            total_correct += correct
            total_chars += seq_len
        
        char_acc = total_correct / total_chars if total_chars > 0 else 0
        self.log('val_char_acc', char_acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        src, tgt = batch['src'], batch['tgt']
        src_lengths, tgt_lengths = batch['src_len'], batch['tgt_len']
        
        if self.beam_size > 1:
            preds = self.beam_search_decode(src, src_lengths)
            # Add evaluation logic here
        else:
            # Regular greedy decoding with target length
            outputs = self(src, src_lengths, max_len=tgt.shape[1])
            
            # Calculate loss
            tgt_out = tgt[:, 1:]  # Shape: [batch_size, tgt_max_len - 1]
            outputs = outputs[:, :-1]  # Shape: [batch_size, tgt_max_len - 1, tgt_vocab_size]
            
            # Flatten
            outputs_flat = outputs.reshape(-1, outputs.shape[-1])  # [batch_size * (tgt_max_len - 1), tgt_vocab_size]
            tgt_out_flat = tgt_out.reshape(-1)  # [batch_size * (tgt_max_len - 1)]
            
            # Calculate loss (ignoring padding)
            loss = F.cross_entropy(outputs_flat, tgt_out_flat, ignore_index=self.pad_idx)
            
            # Log metrics
            self.log('test_loss', loss)
            
            # Calculate accuracy
            preds_flat = outputs_flat.argmax(-1)
            mask = tgt_out_flat != self.pad_idx
            self.test_acc(preds_flat[mask], tgt_out_flat[mask])
            self.log('test_acc', self.test_acc)
            
            # Character-level accuracy
            preds = outputs.argmax(-1)
            total_correct = 0
            total_chars = 0
            
            for i in range(len(src)):
                seq_len = tgt_lengths[i] - 1
                pred_seq = preds[i, :seq_len]
                target_seq = tgt_out[i, :seq_len]
                correct = (pred_seq == target_seq).sum().item()
                total_correct += correct
                total_chars += seq_len
            
            char_acc = total_correct / total_chars if total_chars > 0 else 0
            self.log('test_char_acc', char_acc)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers for Lightning"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def compute_computations(self, T, m, k, V):
        """
        Compute the total number of computations for the model.
        
        Args:
            T: Length of input/output sequence
            m: Embedding dimension
            k: Hidden state dimension
            V: Vocabulary size (assumed same for source and target)
            
        Returns:
            total_ops: Total number of computations
        """
        # Encoder computations
        # Embedding layer: T * m multiplications
        encoder_embed_ops = T * m
        
        # RNN operations for each timestep
        if self.encoder.cell_type == "lstm":
            # LSTM has 4 gates, each with m inputs and k outputs
            # Each gate has m*k multiplications + k additions
            encoder_gate_ops = 4 * (m * k + k)
            # Cell and hidden state updates: ~3k operations each
            encoder_lstm_ops = T * (encoder_gate_ops + 6 * k)
        elif self.encoder.cell_type == "gru":
            # GRU has 3 gates with similar operations
            encoder_gate_ops = 3 * (m * k + k)
            # Hidden state updates: ~4k operations
            encoder_gru_ops = T * (encoder_gate_ops + 4 * k)
        else:  # Simple RNN
            # One transformation: m*k multiplications + k additions
            encoder_rnn_ops = T * (m * k + k)
        
        # Use the appropriate cell type computations
        if self.encoder.cell_type == "lstm":
            encoder_ops = encoder_embed_ops + encoder_lstm_ops
        elif self.encoder.cell_type == "gru":
            encoder_ops = encoder_embed_ops + encoder_gru_ops
        else:
            encoder_ops = encoder_embed_ops + encoder_rnn_ops
        
        # Decoder computations
        # Embedding layer: T * m multiplications
        decoder_embed_ops = T * m
        
        # RNN operations (similar to encoder)
        if self.decoder.cell_type == "lstm":
            decoder_gate_ops = 4 * (m * k + k)
            decoder_lstm_ops = T * (decoder_gate_ops + 6 * k)
        elif self.decoder.cell_type == "gru":
            decoder_gate_ops = 3 * (m * k + k)
            decoder_gru_ops = T * (decoder_gate_ops + 4 * k)
        else:  # Simple RNN
            decoder_rnn_ops = T * (m * k + k)
        
        # Output projection: k * V multiplications + V additions for each timestep
        output_ops = T * (k * V + V)
        
        # Use the appropriate cell type computations
        if self.decoder.cell_type == "lstm":
            decoder_ops = decoder_embed_ops + decoder_lstm_ops + output_ops
        elif self.decoder.cell_type == "gru":
            decoder_ops = decoder_embed_ops + decoder_gru_ops + output_ops
        else:
            decoder_ops = decoder_embed_ops + decoder_rnn_ops + output_ops
        
        # Total computations
        total_ops = encoder_ops + decoder_ops
        
        return total_ops
    
    def compute_parameters(self, m=None, k=None, V=None):
        """
        Compute the total number of parameters in the model.
        
        Args:
            m: Embedding dimension (if None, use the model's value)
            k: Hidden state dimension (if None, use the model's value)
            V: Vocabulary size (if None, use the model's values)
            
        Returns:
            total_params: Total number of parameters
        """
        # Use provided values or get from model
        if m is None:
            m = self.hparams.embedding_dim
        if k is None:
            k = self.hparams.hidden_size
        if V is None:
            src_V = self.hparams.src_vocab_size
            tgt_V = self.hparams.tgt_vocab_size
        else:
            src_V = tgt_V = V
        
        # Encoder parameters
        # Embedding layer: V * m
        encoder_embed_params = src_V * m
        
        # RNN parameters
        if self.encoder.cell_type == "lstm":
            # 4 gates (input, forget, cell, output), each with:
            # - m inputs to k outputs: m*k weights
            # - k bias terms
            # - k hidden to k hidden: k*k weights
            encoder_rnn_params = 4 * (m * k + k + k * k)
        elif self.encoder.cell_type == "gru":
            # 3 gates (reset, update, new), each with similar parameters
            encoder_rnn_params = 3 * (m * k + k + k * k)
        else:  # Simple RNN
            # m inputs to k outputs + k bias terms + k hidden to k hidden
            encoder_rnn_params = m * k + k + k * k
        
        # Decoder parameters
        # Embedding layer: V * m
        decoder_embed_params = tgt_V * m
        
        # RNN parameters (similar to encoder)
        if self.decoder.cell_type == "lstm":
            decoder_rnn_params = 4 * (m * k + k + k * k)
        elif self.decoder.cell_type == "gru":
            decoder_rnn_params = 3 * (m * k + k + k * k)
        else:  # Simple RNN
            decoder_rnn_params = m * k + k + k * k
        
        # Output projection: k inputs to V outputs + V bias terms
        output_params = k * tgt_V + tgt_V
        
        # Total parameters
        total_params = encoder_embed_params + encoder_rnn_params + decoder_embed_params + decoder_rnn_params + output_params
        
        return total_params


if __name__=='__main__':

    from data_prep import create_data_loaders, load_dakshina_data
    
    train_lines, val_lines, test_lines = load_dakshina_data(base_path_data='../dataset/dakshina_dataset_v1.0/', verbose=True)

    # Create data loaders
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_data_loaders(
        train_lines,
        batch_size=32,
        min_freq=1,
        val_lines=val_lines,
        test_lines=test_lines
    )

    # Initialize model
    model = Seq2SeqTransliteration(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embedding_dim=64,
        hidden_size=128,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.1,
        cell_type="gru"
    )

    print(f'Using device: {"gpu" if torch.cuda.is_available() else "cpu"}')
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=3),
            pl.callbacks.ModelCheckpoint(monitor='val_char_acc', mode='max')
        ]
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Test model
    trainer.test(model, test_loader)

    # Total number of parameters
    total_params = model.compute_parameters()
    print(f"Total number of parameters: {total_params}")