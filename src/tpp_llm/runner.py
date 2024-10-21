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

from tpp_llm.model import TPPLLMModel


class TPPLLMRunner(object):
    """
    TPP-LLM Runner
    """

    def __init__(
        self, model: TPPLLMModel, beta_type: float = 1, beta_time: float = 1, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the TPP-LLM runner

        :param model: TPP-LLM model
        :param beta_type: coefficient for the event type prediction loss
        :param beta_time: coefficient for the event time prediction loss
        :param device: device
        """
        self.model = model
        self.beta_type = beta_type
        self.beta_time = beta_time
        self.device = torch.device(device)

        self.num_train_epochs = 0
        self.num_training_steps = 0
        self.num_warmup_steps = 0
        self.global_step = 0
        self.scheduler = None
        self.optimizer = None

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

            # Print metrics
            batch_event_nums = batch_event_nums.numpy(force=True)
            batch_log_likelihoods = - batch_nll_losses.numpy(force=True)
            metrics = {
                'batch_loss': batch_loss.float().cpu().item(),
                'batch_event_nums': batch_event_nums.sum().item(),
                'batch_log_likelihood': batch_log_likelihoods.sum().item() / batch_event_nums.sum().item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch': self.global_step / self.num_training_steps * self.num_train_epochs,
            }
            print(metrics)

            # Optimize the model parameters
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # Update the learning rate
            self.scheduler.step()
            self.global_step += 1

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

    def run_epoch(self, dataloader: DataLoader, phase: str) -> dict:
        """
        Run an epoch with batches of event sequences

        :param dataloader: data loader
        :param phase: running phase (train or eval)
        :return: metrics
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
        dataloader_test: DataLoader = None, learning_rate: float = 5e-4, lr_scheduler_type: str = 'constant',
        num_train_epochs: int = 1, warmup_ratio: float = 0) -> None:
        """
        Run the training, validation, and testing for the model

        :param dataloader_train: data loader of the training set
        :param dataloader_val: data loader of the validation set
        :param dataloader_test: data loader of the testing set
        :param learning_rate: learning rate
        :param lr_scheduler_type: learning rate scheduler type
        :param num_train_epochs: number of training epochs
        :param warmup_ratio: warmup ratio
        """
        # Calculate the number of training steps and the number of warmup steps
        self.num_train_epochs = num_train_epochs
        self.num_training_steps = self.num_train_epochs * len(dataloader_train)
        self.num_warmup_steps = int(warmup_ratio * self.num_training_steps)
        self.global_step = 0
        self.optimizer = Adam(params=self.model.parameters(), lr=learning_rate)

        # Initialize the learning rate scheduler
        if lr_scheduler_type == 'constant':
            self.scheduler = transformers.get_constant_schedule(
                optimizer=self.optimizer,
            )
        elif lr_scheduler_type == 'constant_with_warmup':
            self.scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.num_warmup_steps,
            )
        elif lr_scheduler_type == 'linear':
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
            )
        elif lr_scheduler_type == 'cosine':
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
                num_cycles=0.5,
            )
        else:
            raise KeyError(f'Unknown learning rate scheduler type: {lr_scheduler_type}')

        # Initial validation
        metrics_val_best = None

        if dataloader_val:
            metrics_val = self.run_epoch(dataloader_val, phase='eval')
            print(f'validation metrics: {metrics_val}')
            metrics_val_best = metrics_val

        if dataloader_test:
            metrics_test = self.run_epoch(dataloader_test, phase='eval')
            print(f'test metrics: {metrics_test}')

        # Start training loop
        for epoch in range(num_train_epochs):
            print(f'epoch: {epoch}')

            if dataloader_train:
                metrics_train = self.run_epoch(dataloader_train, phase='train')
                print(f'train metrics of epoch {epoch}: {metrics_train}')

            if dataloader_val:
                metrics_val = self.run_epoch(dataloader_val, phase='eval')
                print(f'validation metrics of epoch {epoch}: {metrics_val}')

                if metrics_val_best is not None and metrics_val['log_likelihood'] > metrics_val_best['log_likelihood']:
                    metrics_val_best = metrics_val
                    print(f'new best validation metrics')

            if dataloader_test:
                metrics_test = self.run_epoch(dataloader_test, phase='eval')
                print(f'test metrics of epoch {epoch}: {metrics_test}')

    def save(self, model_path: str):
        pass

    def load(self, model_path: str):
        pass
