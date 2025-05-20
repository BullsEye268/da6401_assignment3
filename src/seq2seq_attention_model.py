
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
                batch_first=True,
                bidirectional=True  # Use bidirectional for better encoding
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                embedding_dim, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True  # Use bidirectional for better encoding
            )
        else:  # Default to simple RNN
            self.rnn = nn.RNN(
                embedding_dim, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True  # Use bidirectional for better encoding
            )
        
        # Projection layer to reduce bidirectional hidden states to expected decoder hidden size
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout_layer = nn.Dropout(dropout) # Renamed from self.dropout to avoid conflict with arg
    
    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple:
        """
        Forward pass through the encoder.
        
        Args:
            src: Source sequence tensor [batch_size, seq_len]
            src_lengths: Length of each sequence in batch
            
        Returns:
            outputs: RNN outputs for all timesteps [batch_size, seq_len, hidden_size * 2]
            hidden: Final hidden state (and cell state for LSTM) [num_layers, batch_size, hidden_size]
        """
        # Apply embedding and dropout
        embedded = self.dropout_layer(self.embedding(src))  # [batch_size, seq_len, embedding_dim]
        
        # Pack padded sequence for more efficient computation
        # src_lengths might need to be on CPU
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
        
        # Process the final hidden state
        if self.cell_type == "lstm":
            # For LSTM, hidden contains (hidden_state, cell_state)
            hidden_state, cell_state = hidden
            
            # Combine bidirectional hidden states
            # shape: [num_layers * 2, batch_size, hidden_size] -> [num_layers, batch_size, hidden_size * 2]
            hidden_state = hidden_state.view(self.num_layers, 2, -1, self.hidden_size)
            hidden_state = torch.cat([hidden_state[:, 0], hidden_state[:, 1]], dim=2)
            
            # Project to the expected hidden size
            hidden_state = torch.tanh(self.fc(hidden_state))
            
            # Do the same for cell state
            cell_state = cell_state.view(self.num_layers, 2, -1, self.hidden_size)
            cell_state = torch.cat([cell_state[:, 0], cell_state[:, 1]], dim=2)
            cell_state = torch.tanh(self.fc(cell_state))
            
            hidden = (hidden_state, cell_state)
        else:
            # For GRU/RNN, hidden is just the hidden state
            # shape: [num_layers * 2, batch_size, hidden_size] -> [num_layers, batch_size, hidden_size * 2]
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
            
            # Project to the expected hidden size
            hidden = torch.tanh(self.fc(hidden))
        
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int, method: str = "general"):
        """
        Attention mechanism for sequence-to-sequence model.
        
        Args:
            enc_hidden_size: Size of encoder hidden states
            dec_hidden_size: Size of decoder hidden states
            method: Attention method ('dot', 'general', or 'concat')
        """
        super().__init__()
        self.method = method.lower()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(enc_hidden_size, dec_hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size)
            self.v = nn.Linear(dec_hidden_size, 1, bias=False)
    
    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate attention scores.
        
        Args:
            decoder_hidden: Current decoder hidden state [batch_size, dec_hidden_size]
            encoder_outputs: All encoder outputs [batch_size, src_len, enc_hidden_size]
            mask: Mask for padded elements [batch_size, src_len]
            
        Returns:
            attention_weights: Attention weights [batch_size, src_len]
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # Create a score for each encoder output
        if self.method == 'dot':
            # Dot product attention
            # decoder_hidden: [batch_size, dec_hidden_size]
            # encoder_outputs: [batch_size, src_len, enc_hidden_size]
            # Assumes dec_hidden_size == enc_hidden_size
            decoder_hidden_unsqueezed = decoder_hidden.unsqueeze(2)  # [batch_size, dec_hidden_size, 1]
            scores = torch.bmm(encoder_outputs, decoder_hidden_unsqueezed).squeeze(2)  # [batch_size, src_len]
            
        elif self.method == 'general':
            # General attention
            # decoder_hidden: [batch_size, dec_hidden_size]
            # encoder_outputs: [batch_size, src_len, enc_hidden_size]
            decoder_hidden_unsqueezed = decoder_hidden.unsqueeze(2)  # [batch_size, dec_hidden_size, 1]
            energy = self.attn(encoder_outputs)  # [batch_size, src_len, dec_hidden_size]
            scores = torch.bmm(energy, decoder_hidden_unsqueezed).squeeze(2)  # [batch_size, src_len]
            
        elif self.method == 'concat':
            # Concatenation attention
            # decoder_hidden: [batch_size, dec_hidden_size]
            # encoder_outputs: [batch_size, src_len, enc_hidden_size]
            decoder_hidden_repeated = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, dec_hidden_size]
            energy = torch.tanh(self.attn(torch.cat((decoder_hidden_repeated, encoder_outputs), dim=2)))  # [batch_size, src_len, dec_hidden_size]
            scores = self.v(energy).squeeze(2)  # [batch_size, src_len]
        
        # Apply mask if provided (to avoid attending to padding)
        if mask is not None:
            # Ensure mask and scores have compatible shapes for masked_fill
            # scores: [batch_size, src_len]
            # mask: [batch_size, src_len]
            if scores.shape[1] != mask.shape[1]:
                # This case should ideally be handled by ensuring data consistency upstream
                # or by adjusting mask creation to match encoder_outputs.shape[1]
                # For now, this is where the error occurs if shapes mismatch.
                pass # Error will occur here if src_len from encoder_outputs != mask's src_len
            scores = scores.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, src_len]
        
        return attention_weights


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        output_size: int,
        embedding_dim: int,
        hidden_size: int,
        encoder_hidden_size: int,  # For attention
        num_layers: int = 1,
        dropout: float = 0.0,
        cell_type: str = "gru",
        attention_method: str = "general",
    ):
        """
        Decoder with attention for sequence-to-sequence model.
        
        Args:
            output_size: Size of the target vocabulary
            embedding_dim: Size of the character embeddings
            hidden_size: Size of the hidden state
            encoder_hidden_size: Size of encoder hidden states for attention
            num_layers: Number of RNN layers
            dropout: Dropout probability
            cell_type: Type of RNN cell (rnn, lstm, gru)
            attention_method: Attention method ('dot', 'general', or 'concat')
        """
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.cell_type = cell_type.lower()
        
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # Attention mechanism
        self.attention = Attention(encoder_hidden_size, hidden_size, method=attention_method)
        
        # Context vector and embedding will be combined as input to the RNN
        self.input_size = embedding_dim + encoder_hidden_size
        
        # Select RNN type
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                self.input_size, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                self.input_size, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # Default to simple RNN
            self.rnn = nn.RNN(
                self.input_size, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_size + encoder_hidden_size + embedding_dim, output_size)
        self.dropout_layer = nn.Dropout(dropout) # Renamed from self.dropout
    
    def forward(
        self, 
        input: torch.Tensor, 
        hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple:
        """
        Forward pass for a single decoder step with attention.
        
        Args:
            input: Input tensor [batch_size, 1]
            hidden: Hidden state from encoder or previous decoder step
            encoder_outputs: Outputs from the encoder [batch_size, src_len, encoder_hidden_size]
            src_mask: Mask for source padding [batch_size, src_len]
            
        Returns:
            output: Prediction for next character in sequence
            hidden: Updated hidden state
            attention_weights: Attention weights for visualization
        """
        # Apply embedding and dropout [batch_size, 1, embedding_dim]
        embedded = self.dropout_layer(self.embedding(input))
        
        # Get the current hidden state to use for attention
        if self.cell_type == "lstm":
            # For LSTM, hidden is a tuple (hidden_state, cell_state)
            # Use only the hidden state for attention
            attn_hidden = hidden[0][-1]  # Get last layer's hidden state
        else:
            # For GRU/RNN, hidden is just the hidden state
            attn_hidden = hidden[-1]  # Get last layer's hidden state
        
        # Calculate attention weights
        attention_weights = self.attention(attn_hidden, encoder_outputs, src_mask)  # [batch_size, src_len]
        
        # Calculate context vector using attention weights
        attention_weights_unsqueezed = attention_weights.unsqueeze(1)  # [batch_size, 1, src_len]
        context = torch.bmm(attention_weights_unsqueezed, encoder_outputs)  # [batch_size, 1, encoder_hidden_size]
        
        # Combine embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, embedding_dim + encoder_hidden_size]
        
        # Pass through RNN
        output_rnn, hidden = self.rnn(rnn_input, hidden) # output_rnn shape: [batch_size, 1, hidden_size]
        
        # Combine RNN output, context and embedding for prediction
        # Squeeze dimensions of size 1 before cat
        output_combined = torch.cat((output_rnn.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        prediction = self.fc_out(output_combined)  # [batch_size, output_size]
        
        return prediction, hidden, attention_weights # attention_weights is already [batch_size, src_len]


class Seq2SeqAttentionTransliteration(pl.LightningModule):
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
        attention_method: str = "general",
        learning_rate: float = 0.001,
        pad_idx: int = 0,
        teacher_forcing_ratio: float = 0.5,
        beam_size: int = 1,  # 1 means greedy decoding
    ):
        """
        Sequence-to-sequence model with attention for transliteration using PyTorch Lightning.
        
        Args:
            src_vocab_size: Size of the source vocabulary
            tgt_vocab_size: Size of the target vocabulary
            embedding_dim: Size of the character embeddings
            hidden_size: Size of RNN hidden states
            encoder_layers: Number of encoder RNN layers
            decoder_layers: Number of decoder RNN layers
            dropout: Dropout probability
            cell_type: Type of RNN cell (rnn, lstm, gru)
            attention_method: Method for attention ('dot', 'general', or 'concat')
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
        
        # The encoder is bidirectional, so its outputs will have double the hidden size
        encoder_output_hidden_size = hidden_size * 2 # This is the feature size of encoder_outputs
        
        self.decoder = AttentionDecoder(
            output_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size, # Decoder's own hidden state size
            encoder_hidden_size=encoder_output_hidden_size, # Feature size of encoder outputs for attention
            num_layers=decoder_layers,
            dropout=dropout,
            cell_type=cell_type,
            attention_method=attention_method,
        )
        
        # Training parameters
        self.learning_rate = learning_rate
        self.pad_idx = pad_idx
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam_size = beam_size
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.hparams.tgt_vocab_size,
            ignore_index=self.hparams.pad_idx,
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
        Forward pass through the seq2seq model with attention.
        
        Args:
            src: Source sequence tensor [batch_size, src_len]
            src_lengths: Length of each source sequence
            tgt: Target sequence tensor (for teacher forcing) [batch_size, tgt_len]
            max_len: Maximum decoding length (if tgt is None)
            
        Returns:
            outputs: Model predictions [batch_size, tgt_len, tgt_vocab_size]
        """
        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Create source padding mask for attention.
        # The mask's sequence length must match encoder_outputs.shape[1].
        batch_size = encoder_outputs.shape[0]
        effective_src_len = encoder_outputs.shape[1] 
        src_mask = torch.zeros(batch_size, effective_src_len, device=self.device, dtype=torch.bool) # Use bool for mask
        for i, length_val in enumerate(src_lengths):
            # Ensure length_val is an int for slicing, and cap at effective_src_len
            # fill_len = min(length_val.item() if isinstance(length_val, torch.Tensor) else length_val, effective_src_len)
            # Simpler: src_lengths contains true lengths. These lengths define effective_src_len via pad_packed_sequence.
            # So length_val should be <= effective_src_len.
            # The mask should be 1 for actual tokens, 0 for padding.
            # masked_fill fills where mask is True (if mask is bool). If mask is float/int, fills where mask == 0.
            # Let's stick to 0 for padding, 1 for actual tokens, and convert to bool in Attention.forward.
            # Or, make mask True for padding elements.
            # Current Attention.forward: scores.masked_fill(mask == 0, -1e10)
            # So mask should be 0 for padding, 1 for content.
            current_len = length_val.item() if isinstance(length_val, torch.Tensor) else length_val
            src_mask[i, :current_len] = 1
        
        # Determine target sequence length for decoding
        if tgt is not None:
            # Use target length if provided (e.g., during training)
            # max_len is inclusive of start token, exclusive of end token for inputs,
            # or inclusive of end token for outputs.
            # Typically, tgt has shape [batch_size, tgt_seq_len_with_start_end]
            # We generate tgt_seq_len_with_start_end - 1 tokens
            # (from first prediction up to potential end token)
            max_decode_steps = tgt.shape[1] -1 
        else:
            if max_len is None:
                # Default to twice the effective source length if not specified
                max_decode_steps = effective_src_len * 2
            else:
                max_decode_steps = max_len
        
        # Initialize decoder input with start token (assumed to be index 2)
        decoder_input = torch.full((batch_size, 1), 2, dtype=torch.long, device=self.device)
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, max_decode_steps, self.decoder.output_size, device=self.device)
        
        # Store attention weights for visualization (optional)
        # attention_weights_history = torch.zeros(batch_size, max_decode_steps, effective_src_len, device=self.device)
        
        # Decide decoding strategy
        use_teacher_forcing = self.training and tgt is not None and torch.rand(1).item() < self.teacher_forcing_ratio
        
        # Decode step by step
        for t in range(max_decode_steps):
            output, hidden, attn_weights = self.decoder(
                decoder_input, 
                hidden, 
                encoder_outputs,
                src_mask # Pass the correctly shaped mask
            )
            
            outputs[:, t] = output
            # attention_weights_history[:, t] = attn_weights
            
            if use_teacher_forcing:
                # Teacher forcing: use ground truth as next input
                # tgt[:, t+1] because tgt includes <sos> at t=0. So tgt[:,1] is first actual target char.
                decoder_input = tgt[:, t+1].unsqueeze(1)
            else:
                # No teacher forcing: use own predictions as the next input
                top1 = output.argmax(1).unsqueeze(1)
                decoder_input = top1
                
                # Optional: Stop if all sequences in batch have predicted the end token (e.g., index 3)
                # This early stopping is more relevant for inference/validation if not using fixed max_len
                if not self.training and tgt is None and t > 0 and (top1 == 3).all():
                    # Trim outputs to actual length if all ended early
                    outputs = outputs[:, :t+1]
                    break
        
        return outputs
    
    def beam_search_decode(self, src: torch.Tensor, src_lengths: torch.Tensor, max_len: int = 100) -> List[List[int]]:
        """
        Perform beam search decoding with attention.
        
        Args:
            src: Source sequence tensor [batch_size, src_len]
            src_lengths: Length of each source sequence
            max_len: Maximum decoding length
            
        Returns:
            best_sequences: List of best decoded sequences for each example in batch
        """
        if self.beam_size <= 1:
            # Fall back to greedy decoding
            # For greedy, self.forward is used, but it needs careful handling of max_len
            # This part might need to be a separate greedy_decode method or ensure self.forward behaves correctly for inference
            outputs_greedy = self.forward(src, src_lengths, tgt=None, max_len=max_len)
            preds = outputs_greedy.argmax(-1)
            return preds # Return tensor for consistency with test_step
        
        # Encode the source sequence
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)

        batch_size = encoder_outputs.shape[0]
        effective_src_len = encoder_outputs.shape[1]
        
        # Create source padding mask for attention
        src_mask = torch.zeros(batch_size, effective_src_len, device=self.device, dtype=torch.bool)
        for i, length_val in enumerate(src_lengths):
            current_len = length_val.item() if isinstance(length_val, torch.Tensor) else length_val
            src_mask[i, :current_len] = 1
        
        # Store the final results
        final_sequences = []
        
        # Process each item in batch separately for beam search
        for b_idx in range(batch_size):
            # Expand encoder_outputs and hidden state for this item for beam
            # This step is complex if you want to do batch beam search.
            # The provided code does beam search one by one.
            
            # Get encoder hidden state and outputs for this example
            if self.encoder.cell_type == "lstm":
                decoder_hidden_b = (
                    encoder_hidden[0][:, b_idx:b_idx+1].contiguous(),
                    encoder_hidden[1][:, b_idx:b_idx+1].contiguous()
                )
            else:
                decoder_hidden_b = encoder_hidden[:, b_idx:b_idx+1].contiguous()
            
            encoder_outputs_b = encoder_outputs[b_idx:b_idx+1] # [1, src_len, enc_hidden_size]
            src_mask_b = src_mask[b_idx:b_idx+1]             # [1, src_len]
            
            # Start with a beam containing just the start token (index 2)
            # Beam: (sequence_tensor, log_probability, decoder_hidden_state)
            beam = [(torch.tensor([2], device=self.device), 0.0, decoder_hidden_b)]
            
            completed_hypotheses = []

            for _ in range(max_len):
                new_beam = []
                for seq_prefix, score, current_hidden in beam:
                    if seq_prefix[-1].item() == 3: # End token (newline, index 3)
                        completed_hypotheses.append((seq_prefix, score))
                        continue

                    decoder_input = seq_prefix[-1].view(1,1) # Last token of prefix

                    output, new_hidden, _ = self.decoder( # attn_weights not used here
                        decoder_input,
                        current_hidden,
                        encoder_outputs_b,
                        src_mask_b
                    )
                    
                    log_probs = F.log_softmax(output, dim=-1).squeeze(0) # [vocab_size]
                    topk_log_probs, topk_indices = log_probs.topk(self.beam_size) # [beam_size]

                    for k in range(self.beam_size):
                        token_idx = topk_indices[k].unsqueeze(0) # [1]
                        token_log_prob = topk_log_probs[k].item()
                        
                        new_seq = torch.cat((seq_prefix, token_idx))
                        new_score = score + token_log_prob
                        new_beam.append((new_seq, new_score, new_hidden))

                # Sort all candidates by score and select top beam_size
                # Add completed hypotheses to candidates for ranking (length normalization might be needed for fair comparison)
                # For simplicity, we only extend active hypotheses here.
                # A more robust beam search would handle completed ones better.
                
                # Add completed ones to new_beam to keep them if they are still in top-K
                # Or, only add to new_beam if not completed.
                # Let's keep it simple: if something completed, it's out of the active beam.
                # If all active beams are shorter than a completed one, it might be suboptimal.
                
                if not new_beam: # All hypotheses might have completed
                    break

                new_beam.sort(key=lambda x: x[1], reverse=True)
                beam = new_beam[:self.beam_size]

                # If all remaining hypotheses in beam ended
                if all(h[0][-1].item() == 3 for h in beam):
                    completed_hypotheses.extend(beam) # Add them all
                    break
            
            # Add remaining beam hypotheses to completed ones
            completed_hypotheses.extend(beam)
            
            if not completed_hypotheses: # If no hypothesis completed (e.g. max_len too short)
                # Fallback: take the best from the current beam if any, or an empty/default sequence
                if beam:
                    best_hyp = beam[0]
                else: # Should not happen if max_len > 0 and beam_size > 0
                    best_hyp = (torch.tensor([2,3], device=self.device), -float('inf'), None) # SOS, EOS
            else:
                # Sort all completed hypotheses by score (length normalization could be applied here)
                completed_hypotheses.sort(key=lambda x: x[1] / len(x[0]), reverse=True) # Example: length normalization
                best_hyp = completed_hypotheses[0]
            
            final_sequences.append(best_hyp[0].tolist()) # .tolist() to convert tensor to list of ints
        
        # Pad the list of sequences to form a tensor
        max_result_len = 0
        if final_sequences: # Ensure final_sequences is not empty
             max_result_len = max(len(s) for s in final_sequences) if final_sequences else 0
        
        # If final_sequences is empty or max_result_len is 0, handle appropriately
        if max_result_len == 0:
            # Return a tensor with a default shape, e.g., batch_size x 1 with EOS token, or handle error
            # For now, let's assume it produces at least one token. If not, this padding will fail.
            # Let's ensure a minimum length of 1 for padding if all sequences are empty (shouldn't happen with SOS)
            max_result_len = 1 


        padded_sequences = torch.full((batch_size, max_result_len), self.pad_idx, dtype=torch.long, device=self.device)
        for i, seq in enumerate(final_sequences):
            if seq: # Ensure sequence is not empty before trying to tensor it
                padded_sequences[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
            elif padded_sequences.shape[1] > 0: # If seq is empty but we have columns, fill with pad
                 padded_sequences[i,0] = 3 # At least EOS token if empty or only SOS
        
        return padded_sequences

    def training_step(self, batch, batch_idx):
        """Training step for Lightning"""
        src, tgt = batch['src'], batch['tgt']
        src_lengths, tgt_lengths = batch['src_len'], batch['tgt_len']
        
        # Forward pass using teacher forcing according to self.teacher_forcing_ratio
        # self.forward handles this internally if self.training is True
        outputs = self(src, src_lengths, tgt) # tgt is passed for teacher forcing and determining max_len
        
        # Calculate loss (ignoring padding)
        # Targets for loss are tgt excluding <sos> token. outputs are predictions for these.
        # tgt shape: [batch_size, tgt_len_full], outputs shape: [batch_size, tgt_len_full-1, vocab_size]
        tgt_for_loss = tgt[:, 1:]  # Shift right to get targets (skip start token) [B, T_out]
        # outputs already has shape [B, T_out, V] due to max_decode_steps = tgt.shape[1]-1
        
        # Flatten for CrossEntropyLoss
        outputs_flat = outputs.reshape(-1, outputs.shape[-1]) # [B*T_out, V]
        tgt_for_loss_flat = tgt_for_loss.reshape(-1) # [B*T_out]
        
        loss = F.cross_entropy(outputs_flat, tgt_for_loss_flat, ignore_index=self.pad_idx)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy (on non-padded tokens)
        preds_flat = outputs_flat.argmax(-1)
        mask = (tgt_for_loss_flat != self.pad_idx)
        self.train_acc(preds_flat[mask], tgt_for_loss_flat[mask])
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning"""
        src, tgt = batch['src'], batch['tgt']
        src_lengths, tgt_lengths = batch['src_len'], batch['tgt_len']
        
        # Forward pass (no teacher forcing in validation by default, self.forward handles this if self.training is False)
        outputs = self(src, src_lengths, tgt=None, max_len=tgt.shape[1]-1) # Use tgt length for fair comparison
        
        tgt_for_loss = tgt[:, 1:]
        outputs_for_loss = outputs # outputs are already [B, tgt_len-1, V]

        # Safety check for sequence length dimension if outputs were trimmed by early EOS
        # This should align if max_len was tgt.shape[1]-1
        min_len = min(outputs_for_loss.shape[1], tgt_for_loss.shape[1])
        outputs_for_loss = outputs_for_loss[:, :min_len, :]
        tgt_for_loss = tgt_for_loss[:, :min_len]

        outputs_flat = outputs_for_loss.reshape(-1, outputs_for_loss.shape[-1])
        tgt_for_loss_flat = tgt_for_loss.reshape(-1)
        
        loss = F.cross_entropy(outputs_flat, tgt_for_loss_flat, ignore_index=self.pad_idx)
        self.log('val_loss', loss, prog_bar=True)
        
        preds_flat = outputs_flat.argmax(-1)
        mask = (tgt_for_loss_flat != self.pad_idx)
        self.val_acc(preds_flat[mask], tgt_for_loss_flat[mask])
        self.log('val_acc', self.val_acc, prog_bar=True)
        
        # Character-level accuracy (more intuitive for seq2seq)
        preds = outputs_for_loss.argmax(-1) # [B, T_out]
        total_correct_chars = 0
        total_chars_to_predict = 0
        
        for i in range(batch['src'].shape[0]): # Iterate over batch
            # tgt_lengths includes <sos> and <eos>. Prediction is for tokens after <sos> up to <eos>.
            # So, number of characters to predict for sequence i is tgt_lengths[i] - 1.
            # (e.g., if tgt = <sos> a b <eos>, len=4, predict 'a', 'b', <eos> -> 3 tokens)
            # tgt_lengths are original lengths of target sequences including SOS and EOS
            # We predict tgt_lengths[i]-1 tokens.
            
            # Length of actual target characters to compare (excluding SOS, including EOS if predicted)
            # Tgt tokens for comparison are tgt[i, 1:tgt_lengths[i]]
            # Pred tokens are preds[i, :tgt_lengths[i]-1]
            
            len_to_compare = tgt_lengths[i].item() - 1 # Number of tokens to predict for this sequence
            if len_to_compare <= 0: continue

            pred_seq = preds[i, :len_to_compare]
            target_seq = tgt_for_loss[i, :len_to_compare] # tgt_for_loss is already shifted tgt[:,1:]
            
            # Mask out padding within this comparison length (if any, shouldn't be if len_to_compare is correct)
            non_pad_mask_seq = (target_seq != self.pad_idx)
            
            total_correct_chars += (pred_seq[non_pad_mask_seq] == target_seq[non_pad_mask_seq]).sum().item()
            total_chars_to_predict += non_pad_mask_seq.sum().item()

        char_acc = total_correct_chars / total_chars_to_predict if total_chars_to_predict > 0 else 0
        self.log('val_char_acc', char_acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step for Lightning"""
        src, tgt = batch['src'], batch['tgt']
        src_lengths, tgt_lengths = batch['src_len'], batch['tgt_len']
        
        max_decode_len = tgt.shape[1] -1 # Max number of tokens to generate in test for fair comparison
        # Or use a fixed large number like 100 if not comparing to tgt for generation length
        # max_decode_len = int(src_lengths.max().item() * 1.5) + 10 # Example heuristic

        if self.beam_size > 1:
            preds = self.beam_search_decode(src, src_lengths, max_len=max_decode_len) # preds: [B, GenMaxLen]
            # No loss calculation for beam search typically, focus on metrics like BLEU, CharAcc
            loss = None 
        else: # Greedy decoding
            outputs = self(src, src_lengths, tgt=None, max_len=max_decode_len) # outputs: [B, GenMaxLen, V]
            preds = outputs.argmax(-1) # preds: [B, GenMaxLen]

            # Optional: Calculate loss if needed for greedy test
            tgt_for_loss = tgt[:, 1:]
            min_len = min(outputs.shape[1], tgt_for_loss.shape[1])
            outputs_for_loss = outputs[:, :min_len, :]
            tgt_for_loss = tgt_for_loss[:, :min_len]
            outputs_flat = outputs_for_loss.reshape(-1, outputs_for_loss.shape[-1])
            tgt_for_loss_flat = tgt_for_loss.reshape(-1)
            loss = F.cross_entropy(outputs_flat, tgt_for_loss_flat, ignore_index=self.pad_idx)
            self.log('test_loss', loss)

            # Token-level accuracy from torchmetrics (if loss is calculated)
            mask = (tgt_for_loss_flat != self.pad_idx)
            self.test_acc(outputs_flat.argmax(-1)[mask], tgt_for_loss_flat[mask])
            self.log('test_acc', self.test_acc)

        # Calculate character-level accuracy for both beam search and greedy
        # preds are [B, GenMaxLen], tgt is [B, TgtMaxLenWithSosEos]
        tgt_eval = tgt[:, 1:] # [B, TgtMaxLenWithSosEos-1] , target characters without SOS

        total_correct_chars = 0
        total_chars_to_evaluate = 0 # Sum of non-padded characters in target sequences

        for i in range(batch['src'].shape[0]):
            # True length of target sequence (number of characters to predict, e.g. for "a b <eos>" -> 3)
            true_tgt_len = tgt_lengths[i].item() - 1
            if true_tgt_len <= 0: continue

            # Limit comparison to the shorter of generated length and true target length
            len_to_compare = min(preds.shape[1], true_tgt_len)

            pred_seq_comp = preds[i, :len_to_compare]
            target_seq_comp = tgt_eval[i, :len_to_compare]

            # Mask out padding in the target part used for comparison
            non_pad_mask_comp = (target_seq_comp != self.pad_idx)

            total_correct_chars += (pred_seq_comp[non_pad_mask_comp] == target_seq_comp[non_pad_mask_comp]).sum().item()
            
            # Denominator for char_acc should be total true characters in target (non-padded)
            # Mask for true target tokens up to true_tgt_len
            true_target_tokens = tgt_eval[i, :true_tgt_len]
            total_chars_to_evaluate += (true_target_tokens != self.pad_idx).sum().item()


        char_acc = total_correct_chars / total_chars_to_evaluate if total_chars_to_evaluate > 0 else 0
        self.log('test_char_acc', char_acc, prog_bar=True)
        
        return loss # or char_acc or a dict of metrics

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
                'monitor': 'val_loss', # Default monitor for ReduceLROnPlateau
                'interval': 'epoch',
                'frequency': 1
            }
        }