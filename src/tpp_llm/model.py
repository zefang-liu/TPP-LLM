"""
TPP-LLM Model
"""
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
from peft import get_peft_model, PeftConfig
from torch import Tensor
from torch.nn import functional
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

from tpp_llm.layers import TimePositionalEncoding, TimeShiftedPositionalEncoding


class TPPLLMModel(nn.Module):
    """
    TPP-LLM Model
    """

    def __init__(
        self, model_name: str, num_event_types: int, num_integral_samples: int, temporal_emb_type: str,
        temporal_emb_first: bool = False, prompt: str = '', bnb_config: BitsAndBytesConfig = None,
        peft_config: PeftConfig = None, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the TPP-LLM model

        :param model_name: LLM name
        :param num_event_types: number of event types
        :param num_integral_samples: number of samples in the intensity integral
        :param temporal_emb_type: temporal embedding type
        :param temporal_emb_first: temporal embedding first (before the text embedding) for each event
        :param prompt: prompt before the event sequences
        :param bnb_config: bits and bytes configuration
        :param peft_config: PEFT configuration
        :param device: device
        """
        super().__init__()

        # Load the LLM
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float32,
            device_map=self.device,
        )

        # Apply the PEFT config
        if peft_config is not None:
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.train()
            self.llm.print_trainable_parameters()
        else:
            self.llm.eval()
            for param in self.llm.parameters():
                param.requires_grad = False

        # Set the model parameters
        self.hidden_size = self.llm.config.hidden_size
        self.llm_embedder = self.llm.get_input_embeddings()
        self.embedding_dim = self.llm_embedder.embedding_dim
        self.num_integral_samples = num_integral_samples
        self.num_event_types = num_event_types
        self.temporal_emb_type = temporal_emb_type
        self.temporal_emb_first = temporal_emb_first
        self.dtype = self.llm.dtype
        self.prompt = prompt

        # Creat layers for computing intensities
        self.intensity_current = nn.Linear(
            1, self.num_event_types, bias=True, dtype=self.dtype, device=self.device)
        self.intensity_history = nn.Linear(
            self.hidden_size, self.num_event_types, bias=False, dtype=self.dtype, device=self.device)
        self.softplus = nn.Softplus()

        # Creat layers for predicting next events
        self.head_type = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_event_types, bias=True, dtype=self.dtype, device=self.device),
            nn.Softmax(dim=-1))
        self.head_time = nn.Linear(self.hidden_size, 1, bias=True, dtype=self.dtype, device=self.device)

        # Load the temporal embedding
        if self.temporal_emb_type == 'linear':
            self.temporal_embedder = nn.Linear(1, self.embedding_dim, dtype=self.dtype, device=self.device)
        elif self.temporal_emb_type == 'positional':
            self.temporal_embedder = TimePositionalEncoding(
                embedding_dim=self.embedding_dim, dtype=self.dtype, device=self.device)
        elif self.temporal_emb_type == 'shifted':
            self.temporal_embedder = TimeShiftedPositionalEncoding(
                embedding_dim=self.embedding_dim, dtype=self.dtype, device=self.device)
        else:
            raise KeyError(f'Temporal embedding type {self.temporal_emb_type} not implemented.')

        # Generate the prompt embedding
        self.prompt_embeddings = self.embed_event_text(event_texts=[self.prompt], add_special_tokens=True)[0]

    @torch.no_grad()
    def embed_event_text(self, event_texts: List[str], add_special_tokens: bool = False) -> List[Tensor]:
        """
        Embed the texts of events

        :param event_texts: event texts
        :param add_special_tokens: add special tokens or not
        :return: token embeddings of event texts, [(text_token_len, embedding_dim), ...]
        """
        event_tokens = self.tokenizer(
            event_texts, return_tensors='pt', add_special_tokens=add_special_tokens,
            padding=True, truncation=False)
        nums_tokens = event_tokens['attention_mask'].to(self.device).sum(dim=-1)
        event_embeddings_padded = self.llm_embedder(event_tokens['input_ids'].to(self.device))
        event_embeddings = [
            event_embedding_padded[:num_tokens]
            for num_tokens, event_embedding_padded in zip(nums_tokens, event_embeddings_padded)]
        return event_embeddings

    def forward(
        self, batch_event_times: List[Tensor], batch_event_time_deltas: List[Tensor],
        batch_event_texts: List[List[str]]) -> List[Tensor]:
        """
        Forward function to get hidden states of event sequences in a batch

        :param batch_event_times: batch of event times in sequences, [(seq_len,), ...]
        :param batch_event_time_deltas: batch of event time deltas in sequences, [(seq_len,), ...]
        :param batch_event_texts: batch of event texts in sequences, [(seq_len,), ...]
        :return: hidden states of events, [(seq_len, hidden_size), ...]
        """
        batch_sequence_embeddings = []
        batch_attention_masks = []
        batch_event_emb_indices = []

        # Process each event sequence in the batch
        for event_times, event_time_deltas, event_texts in zip(batch_event_times, batch_event_time_deltas,
                                                               batch_event_texts):
            sequence_embeddings = []
            sequence_attention_mask = []
            event_emb_indices = []

            # Add the prompt embeddings
            for prompt_token_embedding in self.prompt_embeddings:
                sequence_embeddings.append(prompt_token_embedding)
                sequence_attention_mask.append(1)

            # Get the temporal embeddings for event times
            if self.temporal_emb_type == 'shifted':
                temporal_embeddings = self.temporal_embedder(
                    event_times.unsqueeze(-1), event_time_deltas.unsqueeze(-1))  # (seq_len, embedding_dim)
            else:
                temporal_embeddings = self.temporal_embedder(event_times.unsqueeze(-1))  # (seq_len, embedding_dim)

            # Get token embeddings for the event texts
            event_text_embeddings = self.embed_event_text(event_texts)  # [(text_token_len, embedding_dim), ...]

            # Process each event with event times and texts
            for temporal_embedding, event_token_embeddings in zip(temporal_embeddings, event_text_embeddings):
                if self.temporal_emb_first:
                    # Add the event time embedding, temporal_embedding: (embedding_dim,)
                    sequence_embeddings.append(temporal_embedding)
                    sequence_attention_mask.append(1)

                    # Add event text token embeddings, event_token_embedding: (embedding_dim,)
                    for event_token_embedding in event_token_embeddings:
                        sequence_embeddings.append(event_token_embedding)
                        sequence_attention_mask.append(1)

                else:
                    # Add event text token embeddings, event_token_embedding: (embedding_dim,)
                    for event_token_embedding in event_token_embeddings:
                        sequence_embeddings.append(event_token_embedding)
                        sequence_attention_mask.append(1)

                    # Add the event time embedding, temporal_embedding: (embedding_dim,)
                    sequence_embeddings.append(temporal_embedding)
                    sequence_attention_mask.append(1)

                # Record the index of the last embedding of this event
                event_emb_indices.append(len(sequence_embeddings) - 1)

            # Convert sequence embeddings to tensors
            batch_sequence_embeddings.append(torch.stack(sequence_embeddings))  # [(seq_emb_len, embedding_dim), ...]
            batch_attention_masks.append(torch.tensor(sequence_attention_mask).to(self.device))  # [(seq_emb_len,), ...]
            batch_event_emb_indices.append(torch.tensor(event_emb_indices).to(self.device))  # [(seq_len,), ...]

        # Pad the sequence embeddings in the batch
        padded_embeddings = torch.nn.utils.rnn.pad_sequence(
            batch_sequence_embeddings, batch_first=True)  # (num_seqs, max_seq_emb_len, embedding_dim)
        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
            batch_attention_masks, batch_first=True)  # (num_seqs, max_seq_emb_len)

        # Pass the padded embeddings through the LLM
        llm_output = self.llm(
            inputs_embeds=padded_embeddings, attention_mask=padded_attention_masks,
        ).last_hidden_state  # (batch_size, max_seq_emb_len, hidden_size)

        # Collect the hidden states at the last event embedding positions
        batch_hidden_states = [
            llm_output[i, batch_event_emb_indices[i], :]
            for i in range(len(batch_event_emb_indices))
        ]  # [(seq_len, hidden_size), ...]

        return batch_hidden_states

    def compute_intensities(
        self, batch_event_time_deltas: List[Tensor], batch_hidden_states: List[Tensor],
    ) -> List[Tensor]:
        """
        Compute intensities from event times and hidden states

        :param batch_event_time_deltas: event time deltas in a batch of event sequences, [(seq_len, n_samples), ...]
        :param batch_hidden_states: hidden states for a batch of event sequences, [(seq_len, hidden_size), ...]
        :return: a batch of event intensities, [(seq_len - 1, num_types), ...]
        """
        batch_intensities = []

        for event_time_deltas, hidden_states in zip(batch_event_time_deltas, batch_hidden_states):
            event_time_deltas_tensor = event_time_deltas.unsqueeze(1)[1:]  # (seq_len - 1, 1)
            intensities_current = self.intensity_current(event_time_deltas_tensor)  # (seq_len - 1, num_types)
            intensities_history = self.intensity_history(hidden_states[:-1])  # (seq_len - 1, num_types)
            intensities = self.softplus(intensities_current + intensities_history)  # (seq_len - 1, num_types)
            batch_intensities.append(intensities)  # [(seq_len - 1, num_types), ...]

        return batch_intensities

    def generate_time_deltas(self, batch_event_time_deltas: List[Tensor]) -> List[Tensor]:
        """
        Generate the time delta samples for every interval without padding.

        :param batch_event_time_deltas: list of event times since the last event, [(seq_len,), ...]
        :return: list of time samples for every interval, [(seq_len - 1, n_samples), ...]
        """
        batch_sampled_time_deltas = []

        # Process each sequence in the batch
        for event_time_deltas in batch_event_time_deltas:
            # Convert the tensor of time deltas
            event_time_deltas_tensor = event_time_deltas.unsqueeze(1)[1:]  # (seq_len - 1, 1)

            # Generate the sampling ratios
            time_delta_ratios = torch.linspace(
                start=0.0, end=1.0, steps=self.num_integral_samples, device=self.device)  # (1, n_samples)

            # Sample the time deltas across the intervals
            sampled_time_deltas = event_time_deltas_tensor * time_delta_ratios.unsqueeze(0)  # (seq_len - 1, n_samples)
            batch_sampled_time_deltas.append(sampled_time_deltas)  # [(seq_len - 1, n_samples), ...]

        return batch_sampled_time_deltas

    def compute_sampled_intensities(
        self, batch_sampled_time_deltas: List[Tensor], batch_hidden_states: List[Tensor],
    ) -> List[Tensor]:
        """
        Compute intensities at sampled time deltas in a batch

        :param batch_sampled_time_deltas: a batch of sampled time delta sequence, [(seq_len - 1, n_samples), ...]
        :param batch_hidden_states: a batch of hidden state sequences, [(seq_len, hidden_size), ...]
        :return: a batch of intensities at sampled times, [(seq_len - 1, n_samples, num_types), ...]
        """
        batch_sampled_intensities = []

        for sampled_time_deltas, hidden_states in zip(batch_sampled_time_deltas, batch_hidden_states):
            sampled_time_deltas_tensor = sampled_time_deltas.unsqueeze(-1)  # (seq_len - 1, n_samples, 1)
            sampled_intensities_current = self.intensity_current(
                sampled_time_deltas_tensor)  # (seq_len - 1, n_samples, num_types)
            sampled_intensities_history = self.intensity_history(
                hidden_states[:-1]).unsqueeze(1)  # (seq_len - 1, 1, num_types)
            sampled_intensities = self.softplus(
                sampled_intensities_current + sampled_intensities_history)  # (seq_len - 1, n_samples, num_types)
            batch_sampled_intensities.append(sampled_intensities)  # [(seq_len - 1, n_samples, num_types), ...]

        return batch_sampled_intensities

    def compute_log_likelihood(
        self, batch_event_time_deltas: List[Tensor], batch_event_types: List[Tensor],
        batch_hidden_states: List[Tensor]) -> List[Tensor]:
        """
        Compute log likelihoods of event sequences in a batch

        :param batch_event_time_deltas: a batch of event time deltas, [(seq_len,), ...]
        :param batch_event_types: a batch of event types, [(seq_len,), ...]
        :param batch_hidden_states: a batch of hidden states, [(seq_len, hidden_size), ...]
        :return: a batch of log likelihoods, [(seq_len - 1,), ...]
        """
        # Compute intensities of events
        batch_event_intensities = self.compute_intensities(
            batch_event_time_deltas=batch_event_time_deltas,
            batch_hidden_states=batch_hidden_states)  # [(seq_len - 1, num_types), ...]

        # Sample time deltas and compute their intensities
        batch_sampled_time_deltas = self.generate_time_deltas(
            batch_event_time_deltas=batch_event_time_deltas)  # [(seq_len - 1, n_samples), ...]
        batch_sampled_intensities = self.compute_sampled_intensities(
            batch_sampled_time_deltas=batch_sampled_time_deltas,
            batch_hidden_states=batch_hidden_states)  # [(seq_len - 1, n_samples, num_types), ...]

        # Compute log likelihoods for the event part
        batch_log_likelihoods = []

        # event_types: (seq_len,), event_time_deltas: (seq_len,)
        # event_intensities: (seq_len - 1, num_types), sampled_intensities: (seq_len - 1, n_samples, num_types)
        for event_types, event_time_deltas, event_intensities, sampled_intensities in zip(
            batch_event_types, batch_event_time_deltas, batch_event_intensities, batch_sampled_intensities):
            # Compute the log likelihood part of events
            event_type_masks = functional.one_hot(
                event_types[1:], num_classes=self.num_event_types).to(self.device)  # (seq_len - 1, num_types)
            event_likelihoods = torch.sum(event_intensities * event_type_masks, dim=-1)  # (seq_len - 1,)
            event_log_likelihoods = torch.log(event_likelihoods)  # (seq_len - 1,)

            # Compute the log likelihood part of non-events
            sampled_total_intensities = torch.sum(sampled_intensities, dim=-1)  # (seq_len - 1, n_samples)
            non_event_log_likelihoods = \
                sampled_total_intensities.mean(dim=-1) * event_time_deltas[1:]  # (seq_len - 1,)

            log_likelihoods = event_log_likelihoods - non_event_log_likelihoods  # (seq_len - 1,)
            batch_log_likelihoods.append(log_likelihoods)

        return batch_log_likelihoods

    def compute_loss(self, batch: Dict[str, list]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute the loss terms

        :param batch: a batch of event sequences
        :return: numbers of events, negative log likelihood (NLL) losses, event type prediction losses,
            event time prediction losses
        """
        batch_event_times = batch['time_since_start']
        batch_event_time_deltas = batch['time_since_last_event']
        batch_event_types = batch['type_event']
        batch_event_texts = batch['type_text']

        # Compute the hidden states
        batch_hidden_states = self.forward(
            batch_event_times=batch_event_times,
            batch_event_time_deltas=batch_event_time_deltas,
            batch_event_texts=batch_event_texts)  # [(seq_len, hidden_size), ...]

        # Compute the log likelihoods
        batch_log_likelihoods = self.compute_log_likelihood(
            batch_event_time_deltas=batch_event_time_deltas,
            batch_event_types=batch_event_types,
            batch_hidden_states=batch_hidden_states)  # [(seq_len - 1,), ...]

        # Predict the next events
        batch_next_event_type_probs, batch_next_event_times = self.predict_next_event_probs(
            batch_hidden_states=batch_hidden_states)

        batch_nll_losses = []
        batch_type_losses = []
        batch_time_losses = []

        for log_likelihoods, next_event_type_probs, event_types, next_event_times, event_times in zip(
            batch_log_likelihoods, batch_next_event_type_probs, batch_event_types, batch_next_event_times,
            batch_event_times):
            nll_loss = - torch.sum(log_likelihoods, dim=0)
            event_type_masks = functional.one_hot(
                event_types[1:], num_classes=self.num_event_types).to(self.device)  # (seq_len - 1, num_types)
            type_loss = torch.sum(- event_type_masks * torch.log(next_event_type_probs[:-1]))
            time_loss = torch.sum((next_event_times[:-1] - event_times[1:]) ** 2, dim=0)

            batch_nll_losses.append(nll_loss)
            batch_type_losses.append(type_loss)
            batch_time_losses.append(time_loss)

        batch_event_nums = torch.LongTensor([len(event_times) - 1 for event_times in batch_event_times])
        batch_nll_losses = torch.stack(batch_nll_losses)
        batch_type_losses = torch.stack(batch_type_losses)
        batch_time_losses = torch.stack(batch_time_losses)

        return batch_event_nums, batch_nll_losses, batch_type_losses, batch_time_losses

    def predict_next_event_probs(self, batch_hidden_states: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Predict next events with probabilities (for training)

        :param batch_hidden_states: hidden states for a batch of event sequences, [(seq_len, hidden_size), ...]
        :return: a batch of next event type probabilities (in [(seq_len, num_types), ...])
            and next event times (in [(seq_len,), ...])
        """
        batch_next_event_type_probs = []
        batch_next_event_times = []

        for hidden_states in batch_hidden_states:
            next_event_type_probs = self.head_type(hidden_states)  # (seq_len, num_types)
            next_event_times = self.head_time(hidden_states).squeeze()  # (seq_len,)
            batch_next_event_type_probs.append(next_event_type_probs)  # [(seq_len, num_types), ...]
            batch_next_event_times.append(next_event_times)  # [(seq_len,), ...]

        return batch_next_event_type_probs, batch_next_event_times

    @torch.no_grad()
    def predict_next_events(self, batch: Dict[str, list]) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        """
        Predict next events (for evaluating)

        :param batch: a batch of event sequences
        :return: numbers of events, sequence log likelihoods, next event types, next event times
        """
        batch_event_times = batch['time_since_start']
        batch_event_time_deltas = batch['time_since_last_event']
        batch_event_types = batch['type_event']
        batch_event_texts = batch['type_text']

        # Compute the hidden states
        batch_hidden_states = self.forward(
            batch_event_times=batch_event_times,
            batch_event_time_deltas=batch_event_time_deltas,
            batch_event_texts=batch_event_texts)  # [(seq_len, hidden_size), ...]

        # Compute the log likelihoods
        batch_log_likelihoods = self.compute_log_likelihood(
            batch_event_time_deltas=batch_event_time_deltas,
            batch_event_types=batch_event_types,
            batch_hidden_states=batch_hidden_states)  # [(seq_len - 1,), ...]
        batch_log_likelihood_sums = torch.stack(
            [log_likelihoods.sum(dim=-1) for log_likelihoods in batch_log_likelihoods])
        batch_event_nums = torch.LongTensor([len(event_times) - 1 for event_times in batch_event_times])

        # Predict the next events
        batch_next_event_type_probs, batch_next_event_times = self.predict_next_event_probs(
            batch_hidden_states=batch_hidden_states)
        batch_next_event_types = [
            next_event_type_probs.argmax(dim=-1) for next_event_type_probs in batch_next_event_type_probs]

        return batch_event_nums, batch_log_likelihood_sums, batch_next_event_types, batch_next_event_times
