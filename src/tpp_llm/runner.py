"""
TPP-LLM Runner
"""
from typing import Dict, Tuple, List, Union

import numpy as np
import torch
import transformers
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tpp_llm.data import TPPLLMDataset, collate_fn
from tpp_llm.model import TPPLLMModel
from tpp_llm.utils import get_prompt


class TPPLLMRunner(object):
    """
    TPP-LLM Runner
    """

    def __init__(
        self, model: TPPLLMModel, learning_rate: float, beta_type: float = 1, beta_time: float = 1,
        device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the TPP-LLM runner

        :param model: TPP-LLM model
        :param learning_rate: learning rate
        :param beta_type: coefficient for the event type prediction loss
        :param beta_time: coefficient for the event time prediction loss
        :param device: device
        """
        self.model = model
        self.beta_type = beta_type
        self.beta_time = beta_time
        self.learning_rate = learning_rate
        self.optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device(device)

        self.move_to_numpy_prev = lambda list_tensors: [tensor[1:].numpy(force=True) for tensor in list_tensors]
        self.move_to_numpy_next = lambda list_tensors: [tensor[:-1].numpy(force=True) for tensor in list_tensors]

    def run_batch(self, batch: Dict[str, list], phase: str) \
        -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Run a batch of event sequences

        :param batch: a batch of event sequences
        :param phase: running phase (train or eval)
        :return: numbers of events, sequence log likelihoods, event types, predicted event types, event times,
            predicted event times
        """
        batch = {
            'time_since_start': [_seq.to(self.device) for _seq in batch['time_since_start']],
            'time_since_last_event': [_seq.to(self.device) for _seq in batch['time_since_last_event']],
            'type_event': [_seq.to(self.device) for _seq in batch['type_event']],
            'type_text': batch['type_text'],
        }

        if phase == 'train':
            self.model.train()

            # Get the loss terms
            batch_event_nums, batch_nll_losses, batch_type_losses, batch_time_losses = self.model.compute_loss(batch)
            batch_loss = torch.sum(
                batch_nll_losses + self.beta_type * batch_type_losses + self.beta_time * batch_time_losses)

            # Optimize the model parameters
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # Convert tensors
            batch_event_nums = batch_event_nums.numpy(force=True)
            batch_log_likelihoods = - batch_nll_losses.numpy(force=True)
            print(
                f"batch_loss: {batch_loss.numpy(force=True):.4f}, "
                f"batch_event_nums: {np.sum(batch_event_nums):d}, "
                f"batch_log_likelihood: {np.sum(batch_log_likelihoods) / np.sum(batch_event_nums):.4f}")

            return batch_event_nums, batch_log_likelihoods, [], [], [], []

        elif phase == 'eval':
            self.model.eval()

            # Get the event predictions and labels
            batch_event_times = batch['time_since_start']
            batch_event_types = batch['type_event']
            batch_event_nums, batch_log_likelihoods, batch_next_event_types, batch_next_event_times = \
                self.model.predict_next_events(batch)

            # Shift and convert tensors
            batch_event_nums = batch_event_nums.numpy(force=True)
            batch_log_likelihoods = batch_log_likelihoods.numpy(force=True)
            batch_event_types_shifted = self.move_to_numpy_prev(batch_event_types)
            batch_event_times_shifted = self.move_to_numpy_prev(batch_event_times)
            batch_next_event_types_shifted = self.move_to_numpy_next(batch_next_event_types)
            batch_next_event_times_shifted = self.move_to_numpy_next(batch_next_event_times)

            return batch_event_nums, batch_log_likelihoods, batch_event_types_shifted, batch_next_event_types_shifted, \
                   batch_event_times_shifted, batch_next_event_times_shifted

        else:
            raise KeyError(f'Unknown phase: {phase}.')

    def run_epoch(self, dataloader: DataLoader, phase: str):
        """
        Run an epoch with batches of event sequences

        :param dataloader: data loader
        :param phase: running phase (train or eval)
        :return:
        """

        total_log_likelihood = 0
        total_num_events = 0
        all_event_types = []
        all_event_type_preds = []
        all_event_times = []
        all_event_time_preds = []
        metrics = {}

        for batch in tqdm(dataloader):
            event_nums, log_likelihoods, event_types, event_type_preds, event_times, event_time_preds = \
                self.run_batch(batch=batch, phase=phase)
            total_log_likelihood += np.sum(log_likelihoods)
            total_num_events += np.sum(event_nums)

            if phase == 'eval':
                all_event_types.append(np.concatenate(event_types))
                all_event_type_preds.append(np.concatenate(event_type_preds))
                all_event_times.append(np.concatenate(event_times))
                all_event_time_preds.append(np.concatenate(event_time_preds))

        avg_log_likelihood = total_log_likelihood / total_num_events
        metrics['log_likelihood'] = float(avg_log_likelihood)
        metrics['num_events'] = int(total_num_events)

        if phase == 'eval':
            all_event_types = np.concatenate(all_event_types)
            all_event_type_preds = np.concatenate(all_event_type_preds)
            all_event_times = np.concatenate(all_event_times)
            all_event_time_preds = np.concatenate(all_event_time_preds)
            accuracy = np.mean(all_event_types == all_event_type_preds)
            rmse = np.sqrt(np.mean((all_event_times - all_event_time_preds) ** 2))
            metrics['accuracy'] = float(accuracy)
            metrics['rmse'] = float(rmse)

        return metrics

    def run(
        self, dataloader_train: DataLoader = None, dataloader_val: DataLoader = None,
        dataloader_test: DataLoader = None, num_epochs: int = 1) -> None:
        """
        Run the training, validation, and testing for the model

        :param dataloader_train: data loader of the training set
        :param dataloader_val: data loader of the validation set
        :param dataloader_test: data loader of the testing set
        :param num_epochs: number of epochs
        """
        metrics_val_best = None

        for epoch in range(num_epochs):
            print(f'epoch: {epoch}')

            if dataloader_train:
                metrics_train = self.run_epoch(dataloader_train, phase='train')
                print(f'train metrics of epoch {epoch}: {metrics_train}')

            if dataloader_val:
                metrics_val = self.run_epoch(dataloader_val, phase='eval')
                print(f'validation metrics of epoch {epoch}: {metrics_val}')

                if metrics_val_best is None:
                    metrics_val_best = metrics_val
                    print(f'new best validation metrics')
                elif metrics_val['log_likelihood'] > metrics_val_best['log_likelihood']:
                    metrics_val_best = metrics_val
                    print(f'new best validation metrics')

            if dataloader_test:
                metrics_test = self.run_epoch(dataloader_test, phase='eval')
                print(f'test metrics of epoch {epoch}: {metrics_test}')

    def save(self, model_path: str):
        pass

    def load(self, model_path: str):
        pass
