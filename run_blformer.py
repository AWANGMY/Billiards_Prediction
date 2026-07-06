import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ClassesData.BLFormerDataset import BLFormerDataset, blformer_collate
from ClassesML.BLFormer import BLFormer


TASKS = ['clear', 'win', 'potted_after_break']
CLEAN_TARGET_ACCURACY = {'clear': 71.477663,
                         'win': 67.268041,
                         'potted_after_break': 61.254296}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-path',
                        default=os.path.join('Dataset', 'processed', 'billiards_layout.pt'))
    parser.add_argument('--output-dir',
                        default=os.path.join('Output', 'blformer'))
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--allow-cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--log-every-epochs', type=int, default=1)
    parser.add_argument('--disable-augmentation', action='store_true')

    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--clear-weight', type=float, default=1.0)
    parser.add_argument('--win-weight', type=float, default=1.0)
    parser.add_argument('--potted-weight', type=float, default=1.0)
    parser.add_argument('--potted-head',
                        choices=['ordinal', 'class', 'hybrid'],
                        default='ordinal')
    parser.add_argument('--potted-ordinal-weight', type=float, default=1.0)
    parser.add_argument('--potted-ce-weighting',
                        choices=['none', 'effective'],
                        default='none')
    parser.add_argument('--potted-ce-beta', type=float, default=0.99)
    parser.add_argument('--potted-label-smoothing', type=float, default=0.0)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--ffn-dim', type=int, default=128)
    parser.add_argument('--bias-hidden-dim', type=int, default=32)
    parser.add_argument('--pooling', choices=['cls', 'cls_mean'], default='cls')
    parser.add_argument('--use-paper-features', action='store_true')

    parser.add_argument('--search',
                        choices=['none', 'doc_small', 'paper40_improve',
                                 'paper40_clear_focus', 'paper40_paper_fusion'],
                        default='doc_small')
    parser.add_argument('--max-trials', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--use-stored-val', action='store_true')
    parser.add_argument('--selection-metric',
                        choices=['mean_macro_f1', 'mean_accuracy',
                                 'clean_target_min_margin',
                                 'clear_accuracy', 'win_accuracy',
                                 'potted_after_break_accuracy'],
                        default='mean_macro_f1')
    parser.add_argument('--final-train-on-full-train', action='store_true')

    return parser.parse_args()


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_processed_data(processed_path):

    if not os.path.exists(processed_path):
        raise FileNotFoundError('Processed data not found: ' + processed_path)

    try:
        return torch.load(processed_path, weights_only=False)
    except TypeError:
        return torch.load(processed_path)


def resolve_device(args):

    requested_device = torch.device(args.device)

    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        if args.allow_cpu:
            return torch.device('cpu')
        raise RuntimeError('CUDA is not available. Run this script on an interact-g GPU node '
                           'or pass --allow-cpu only for smoke tests.')

    if requested_device.type == 'cpu' and not args.allow_cpu:
        raise RuntimeError('CPU execution is disabled by default. Pass --allow-cpu only for '
                           'smoke tests; formal BLFormer training should use CUDA.')

    return requested_device


def to_index_list(indices):

    if isinstance(indices, torch.Tensor):
        return indices.cpu().long().tolist()

    return list(indices)


def build_splits(data, args):

    stored_splits = data['split_indices']
    test_indices = to_index_list(stored_splits['test'])

    if args.use_stored_val:
        train_indices = to_index_list(stored_splits['train'])
        val_indices = to_index_list(stored_splits.get('val', []))
        final_train_indices = train_indices + val_indices
        notes = ['uses train/val/test indices stored in processed data']
    else:
        pool_indices = (to_index_list(stored_splits['train']) +
                        to_index_list(stored_splits.get('val', [])))
        final_train_indices = list(pool_indices)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(pool_indices)
        val_size = int(np.floor(args.val_ratio * len(pool_indices)))
        if val_size <= 0:
            raise ValueError('Validation split is empty; increase --val-ratio.')
        val_indices = pool_indices[:val_size]
        train_indices = pool_indices[val_size:]
        notes = ['re-splits stored train+val with validation ratio ' + str(args.val_ratio),
                 'uses stored test indices as held-out test set']

    if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
        raise ValueError('Empty split found: train=' + str(len(train_indices)) +
                         ' val=' + str(len(val_indices)) +
                         ' test=' + str(len(test_indices)))

    return {'train': train_indices,
            'val': val_indices,
            'final_train': final_train_indices,
            'test': test_indices,
            'notes': notes}


def make_loader(data, indices, batch_size, shuffle, augment, seed, num_workers, device):

    dataset = BLFormerDataset(data, indices, augment=augment)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=blformer_collate,
                      generator=generator,
                      pin_memory=(device.type == 'cuda'))


def move_batch(batch, device):

    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value

    return moved


def base_hyperparameters(args, trial_config):

    return {'d_model': trial_config.get('d_model', args.d_model),
            'num_heads': trial_config.get('num_heads', args.num_heads),
            'num_layers': trial_config.get('num_layers', args.num_layers),
            'ffn_dim': trial_config.get('ffn_dim', args.ffn_dim),
            'dropout_rate': trial_config['dropout_rate'],
            'bias_hidden_dim': trial_config.get('bias_hidden_dim', args.bias_hidden_dim),
            'pooling': trial_config.get('pooling', args.pooling),
            'use_paper_features': trial_config.get('use_paper_features',
                                                   args.use_paper_features),
            'num_paper_features': 27,
            'paper_feature_vocab_size': 200,
            'pair_feature_dim': 10,
            'num_token_types': 4,
            'num_ball_ids': 11,
            'num_potted_thresholds': 9,
            'num_potted_classes': 10}


def trial_configs(args):

    base = {'trial_name': 'base',
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'dropout_rate': args.dropout_rate,
            'clear_weight': args.clear_weight,
            'win_weight': args.win_weight,
            'potted_weight': args.potted_weight,
            'potted_head': args.potted_head,
            'potted_ordinal_weight': args.potted_ordinal_weight,
            'potted_ce_weighting': args.potted_ce_weighting,
            'potted_ce_beta': args.potted_ce_beta,
            'potted_label_smoothing': args.potted_label_smoothing,
            'pooling': args.pooling,
            'use_paper_features': args.use_paper_features}

    if args.search == 'none':
        configs = [base]
    elif args.search == 'doc_small':
        configs = [
            base,
            override_trial(base, 'lr_1e-4', learning_rate=1e-4),
            override_trial(base, 'lr_1e-3', learning_rate=1e-3),
            override_trial(base, 'dropout_0.05', dropout_rate=0.05),
            override_trial(base, 'dropout_0.2', dropout_rate=0.2),
            override_trial(base, 'weight_decay_1e-3', weight_decay=1e-3),
            override_trial(base, 'potted_weight_0.5', potted_weight=0.5),
            override_trial(base, 'potted_weight_2.0', potted_weight=2.0),
            override_trial(base, 'lr_1e-4_dropout_0.2',
                           learning_rate=1e-4, dropout_rate=0.2),
            override_trial(base, 'lr_1e-3_dropout_0.2',
                           learning_rate=1e-3, dropout_rate=0.2),
            override_trial(base, 'wd_1e-3_dropout_0.05',
                           weight_decay=1e-3, dropout_rate=0.05),
            override_trial(base, 'dropout_0.2_potted_weight_2.0',
                           dropout_rate=0.2, potted_weight=2.0),
        ]
    elif args.search == 'paper40_improve':
        configs = [
            override_trial(base, 'class_cls_lr3e-4',
                           potted_head='class'),
            override_trial(base, 'class_cls_lr1e-4',
                           potted_head='class', learning_rate=1e-4),
            override_trial(base, 'class_cls_wd1e-3',
                           potted_head='class', weight_decay=1e-3),
            override_trial(base, 'class_cls_smooth0.05',
                           potted_head='class', potted_label_smoothing=0.05),
            override_trial(base, 'class_cls_effective',
                           potted_head='class', potted_ce_weighting='effective'),
            override_trial(base, 'class_clsmean_lr3e-4',
                           potted_head='class', pooling='cls_mean'),
            override_trial(base, 'class_clsmean_lr1e-4',
                           potted_head='class', pooling='cls_mean',
                           learning_rate=1e-4),
            override_trial(base, 'hybrid_cls_lr3e-4',
                           potted_head='hybrid'),
            override_trial(base, 'hybrid_cls_ord0.25',
                           potted_head='hybrid', potted_ordinal_weight=0.25),
            override_trial(base, 'hybrid_clsmean_ord0.25',
                           potted_head='hybrid', pooling='cls_mean',
                           potted_ordinal_weight=0.25),
            override_trial(base, 'class_d96_clsmean',
                           potted_head='class', pooling='cls_mean',
                           d_model=96, ffn_dim=192),
            override_trial(base, 'class_d96_clsmean_wd1e-3',
                           potted_head='class', pooling='cls_mean',
                           d_model=96, ffn_dim=192, weight_decay=1e-3),
        ]
    elif args.search == 'paper40_clear_focus':
        configs = [
            override_trial(base, 'class_clsmean_clear1.5',
                           potted_head='class', pooling='cls_mean',
                           clear_weight=1.5),
            override_trial(base, 'class_clsmean_clear2',
                           potted_head='class', pooling='cls_mean',
                           clear_weight=2.0),
            override_trial(base, 'class_clsmean_clear3',
                           potted_head='class', pooling='cls_mean',
                           clear_weight=3.0),
            override_trial(base, 'class_clsmean_clear4',
                           potted_head='class', pooling='cls_mean',
                           clear_weight=4.0),
            override_trial(base, 'class_clsmean_clear2_pot0.75',
                           potted_head='class', pooling='cls_mean',
                           clear_weight=2.0, potted_weight=0.75),
            override_trial(base, 'class_clsmean_clear3_pot0.75',
                           potted_head='class', pooling='cls_mean',
                           clear_weight=3.0, potted_weight=0.75),
            override_trial(base, 'class_clsmean_clear2_lr1e-4',
                           potted_head='class', pooling='cls_mean',
                           clear_weight=2.0, learning_rate=1e-4),
            override_trial(base, 'hybrid_clsmean_clear2_ord0.25',
                           potted_head='hybrid', pooling='cls_mean',
                           clear_weight=2.0, potted_ordinal_weight=0.25),
            override_trial(base, 'class_d96_clsmean_clear2',
                           potted_head='class', pooling='cls_mean',
                           d_model=96, ffn_dim=192, clear_weight=2.0),
            override_trial(base, 'class_d96_clsmean_clear3',
                           potted_head='class', pooling='cls_mean',
                           d_model=96, ffn_dim=192, clear_weight=3.0),
        ]
    else:
        configs = [
            override_trial(base, 'paper_class_cls',
                           potted_head='class', use_paper_features=True),
            override_trial(base, 'paper_class_clsmean',
                           potted_head='class', pooling='cls_mean',
                           use_paper_features=True),
            override_trial(base, 'paper_class_clsmean_clear1.5',
                           potted_head='class', pooling='cls_mean',
                           use_paper_features=True, clear_weight=1.5),
            override_trial(base, 'paper_class_clsmean_clear2',
                           potted_head='class', pooling='cls_mean',
                           use_paper_features=True, clear_weight=2.0),
            override_trial(base, 'paper_hybrid_clsmean_ord0.25',
                           potted_head='hybrid', pooling='cls_mean',
                           use_paper_features=True, potted_ordinal_weight=0.25),
            override_trial(base, 'paper_hybrid_clsmean_clear1.5',
                           potted_head='hybrid', pooling='cls_mean',
                           use_paper_features=True, clear_weight=1.5,
                           potted_ordinal_weight=0.25),
            override_trial(base, 'paper_class_clsmean_wd1e-3',
                           potted_head='class', pooling='cls_mean',
                           use_paper_features=True, weight_decay=1e-3),
            override_trial(base, 'paper_class_d96_clsmean',
                           potted_head='class', pooling='cls_mean',
                           use_paper_features=True, d_model=96, ffn_dim=192),
        ]

    if args.max_trials is not None:
        configs = configs[:args.max_trials]

    return configs


def override_trial(base, name, **kwargs):

    config = dict(base)
    config.update(kwargs)
    config['trial_name'] = name

    return config


def coral_targets(y, num_thresholds=9):

    thresholds = torch.arange(num_thresholds, device=y.device)

    return (y[:, None] > thresholds[None, :]).float()


def build_loss_config(data, train_indices, trial_config, device):

    loss_config = {'potted_class_weights': None}

    if trial_config['potted_ce_weighting'] == 'effective':
        y = data['potted_after_break'][train_indices].long()
        counts = torch.bincount(y, minlength=10).float()
        beta = float(trial_config['potted_ce_beta'])
        effective_num = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        weights = (1.0 - beta) / effective_num.clamp_min(1.0e-12)
        weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
        weights = weights / weights.sum().clamp_min(1.0e-12) * len(weights)
        loss_config['potted_class_weights'] = weights.to(device)

    return loss_config


def compute_loss(outputs, batch, trial_config, loss_config):

    clear_target = batch['clear'].float()
    win_target = batch['win'].float()
    potted_target = coral_targets(batch['potted_after_break'],
                                  outputs['potted_logits'].shape[1])

    clear_loss = F.binary_cross_entropy_with_logits(outputs['clear_logit'],
                                                    clear_target)
    win_loss = F.binary_cross_entropy_with_logits(outputs['win_logit'],
                                                  win_target)
    potted_head = trial_config['potted_head']
    ordinal_loss = F.binary_cross_entropy_with_logits(outputs['potted_logits'],
                                                      potted_target,
                                                      reduction='none')
    ordinal_loss = ordinal_loss.sum(dim=1).mean()

    class_loss = F.cross_entropy(outputs['potted_class_logits'],
                                 batch['potted_after_break'],
                                 weight=loss_config['potted_class_weights'],
                                 label_smoothing=trial_config['potted_label_smoothing'])

    if potted_head == 'ordinal':
        potted_loss = ordinal_loss
    elif potted_head == 'class':
        potted_loss = class_loss
    elif potted_head == 'hybrid':
        potted_loss = class_loss + trial_config['potted_ordinal_weight'] * ordinal_loss
    else:
        raise ValueError('Unknown potted head: ' + str(potted_head))

    total_loss = (trial_config['clear_weight'] * clear_loss +
                  trial_config['win_weight'] * win_loss +
                  trial_config['potted_weight'] * potted_loss)

    return {'loss': total_loss,
            'clear_loss': clear_loss.detach(),
            'win_loss': win_loss.detach(),
            'potted_loss': potted_loss.detach(),
            'potted_ordinal_loss': ordinal_loss.detach(),
            'potted_class_loss': class_loss.detach()}


def predictions_from_outputs(outputs, potted_head='ordinal'):

    clear_pred = (outputs['clear_logit'] > 0.0).long()
    win_pred = (outputs['win_logit'] > 0.0).long()

    if potted_head == 'ordinal':
        potted_pred = (outputs['potted_logits'] > 0.0).sum(dim=1).long()
    else:
        potted_pred = torch.argmax(outputs['potted_class_logits'], dim=1)

    return {'clear': clear_pred,
            'win': win_pred,
            'potted_after_break': potted_pred}


def train_one_epoch(model, loader, optimizer, scheduler, device, trial_config,
                    loss_config, args):

    model.train()
    accumulator = MetricAccumulator(potted_head=trial_config['potted_head'])

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch)
        losses = compute_loss(outputs, batch, trial_config, loss_config)
        losses['loss'].backward()

        if args.grad_clip is not None and args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        accumulator.update(batch, outputs, losses)

    scheduler.step()

    return accumulator.metrics()


def evaluate(model, loader, device, trial_config, loss_config, return_predictions=False):

    model.eval()
    accumulator = MetricAccumulator(potted_head=trial_config['potted_head'],
                                    keep_predictions=return_predictions)

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch)
            losses = compute_loss(outputs, batch, trial_config, loss_config)
            accumulator.update(batch, outputs, losses)

    return accumulator.metrics()


class MetricAccumulator:

    def __init__(self, potted_head='ordinal', keep_predictions=False):

        self.potted_head = potted_head
        self.keep_predictions = keep_predictions
        self.loss_sums = {'loss': 0.0,
                          'clear_loss': 0.0,
                          'win_loss': 0.0,
                          'potted_loss': 0.0,
                          'potted_ordinal_loss': 0.0,
                          'potted_class_loss': 0.0}
        self.count = 0
        self.y_true = {task: [] for task in TASKS}
        self.y_pred = {task: [] for task in TASKS}
        self.sample_indices = []

    def update(self, batch, outputs, losses):

        batch_size = int(batch['clear'].shape[0])
        self.count += batch_size

        for key in self.loss_sums:
            self.loss_sums[key] += float(losses[key].item()) * batch_size

        predictions = predictions_from_outputs(outputs, self.potted_head)
        for task in TASKS:
            self.y_true[task].append(batch[task].detach().cpu())
            self.y_pred[task].append(predictions[task].detach().cpu())

        if self.keep_predictions:
            self.sample_indices.append(batch['sample_indices'].detach().cpu())

    def metrics(self):

        metrics = {}
        for key, value in self.loss_sums.items():
            metrics[key] = value / self.count if self.count > 0 else None

        task_metrics = {}
        for task in TASKS:
            y_true = torch.cat(self.y_true[task], dim=0)
            y_pred = torch.cat(self.y_pred[task], dim=0)
            labels = [0, 1] if task in ['clear', 'win'] else list(range(10))
            task_metrics[task] = classification_metrics(y_true, y_pred, labels)

        metrics['task_metrics'] = task_metrics
        metrics['mean_macro_f1'] = sum([task_metrics[task]['macro_f1']
                                        for task in TASKS]) / len(TASKS)
        metrics['mean_accuracy'] = sum([task_metrics[task]['accuracy']
                                        for task in TASKS]) / len(TASKS)
        margins = [task_metrics[task]['accuracy'] - CLEAN_TARGET_ACCURACY[task]
                   for task in TASKS]
        metrics['clean_target_min_margin'] = min(margins)
        metrics['clean_target_mean_margin'] = sum(margins) / len(margins)

        if self.keep_predictions:
            metrics['predictions'] = self.prediction_rows()

        return metrics

    def prediction_rows(self):

        rows = []
        sample_indices = torch.cat(self.sample_indices, dim=0)
        y_true = {task: torch.cat(self.y_true[task], dim=0) for task in TASKS}
        y_pred = {task: torch.cat(self.y_pred[task], dim=0) for task in TASKS}

        for row_index in range(int(sample_indices.numel())):
            row = {'sample_index': int(sample_indices[row_index].item())}
            for task in TASKS:
                row[task + '_true'] = int(y_true[task][row_index].item())
                row[task + '_pred'] = int(y_pred[task][row_index].item())
            rows.append(row)

        return rows


def classification_metrics(y_true, y_pred, labels):

    y_true = y_true.long()
    y_pred = y_pred.long()
    accuracy = float((y_true == y_pred).float().mean().item()) * 100.0

    f1_values = []
    precision_values = []
    recall_values = []

    for label in labels:
        label_tensor = torch.tensor(label, dtype=torch.long)
        true_positive = int(((y_true == label_tensor) & (y_pred == label_tensor)).sum().item())
        false_positive = int(((y_true != label_tensor) & (y_pred == label_tensor)).sum().item())
        false_negative = int(((y_true == label_tensor) & (y_pred != label_tensor)).sum().item())

        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    return {'accuracy': accuracy,
            'macro_precision': sum(precision_values) / len(precision_values),
            'macro_recall': sum(recall_values) / len(recall_values),
            'macro_f1': sum(f1_values) / len(f1_values)}


def safe_divide(numerator, denominator):

    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def clone_state_dict(model):

    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def is_better_metrics(candidate, current_best, selection_metric):

    if current_best is None:
        return True

    if selection_metric in ['clear_accuracy', 'win_accuracy',
                            'potted_after_break_accuracy']:
        task = selection_metric.replace('_accuracy', '')
        candidate_key = (candidate['task_metrics'][task]['accuracy'],
                         candidate['task_metrics'][task]['macro_f1'],
                         -candidate['loss'])
        current_key = (current_best['task_metrics'][task]['accuracy'],
                       current_best['task_metrics'][task]['macro_f1'],
                       -current_best['loss'])
    elif selection_metric == 'clean_target_min_margin':
        candidate_key = (candidate['clean_target_min_margin'],
                         candidate['mean_accuracy'],
                         candidate['mean_macro_f1'],
                         -candidate['loss'])
        current_key = (current_best['clean_target_min_margin'],
                       current_best['mean_accuracy'],
                       current_best['mean_macro_f1'],
                       -current_best['loss'])
    elif selection_metric == 'mean_accuracy':
        candidate_key = (candidate['mean_accuracy'],
                         candidate['mean_macro_f1'],
                         -candidate['loss'])
        current_key = (current_best['mean_accuracy'],
                       current_best['mean_macro_f1'],
                       -current_best['loss'])
    else:
        candidate_key = (candidate['mean_macro_f1'],
                         candidate['mean_accuracy'],
                         -candidate['loss'])
        current_key = (current_best['mean_macro_f1'],
                       current_best['mean_accuracy'],
                       -current_best['loss'])

    return candidate_key > current_key


def run_trial(data, splits, args, device, trial_config, trial_index):

    set_seed(args.seed)

    train_loader = make_loader(data, splits['train'], args.batch_size, True,
                               not args.disable_augmentation,
                               args.seed + trial_index, args.num_workers, device)
    val_loader = make_loader(data, splits['val'], args.batch_size, False,
                             False, args.seed, args.num_workers, device)
    test_loader = make_loader(data, splits['test'], args.batch_size, False,
                              False, args.seed, args.num_workers, device)

    hparams = base_hyperparameters(args, trial_config)
    model = BLFormer(hparams).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=trial_config['learning_rate'],
                                  weight_decay=trial_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)
    loss_config = build_loss_config(data, splits['train'], trial_config, device)

    best_state = None
    best_epoch = None
    best_val_metrics = None
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                        scheduler, device, trial_config,
                                        loss_config, args)
        val_metrics = evaluate(model, val_loader, device, trial_config, loss_config)
        learning_rate = scheduler.get_last_lr()[0]

        history_row = flatten_epoch_metrics(trial_config['trial_name'],
                                            epoch,
                                            learning_rate,
                                            train_metrics,
                                            val_metrics)
        history.append(history_row)

        if is_better_metrics(val_metrics, best_val_metrics, args.selection_metric):
            best_val_metrics = without_predictions(val_metrics)
            best_state = clone_state_dict(model)
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        if should_log_epoch(args, epoch):
            print(format_progress(trial_config['trial_name'],
                                  epoch,
                                  args.epochs,
                                  train_metrics,
                                  val_metrics,
                                  best_epoch))

        if patience_count >= args.patience:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, trial_config, loss_config,
                            return_predictions=True)

    parameter_count = sum([parameter.numel() for parameter in model.parameters()])

    summary = {'trial_name': trial_config['trial_name'],
               'trial_index': trial_index,
               'best_epoch': best_epoch,
               'epochs_ran': len(history),
               'parameter_count': parameter_count,
               'learning_rate': trial_config['learning_rate'],
               'weight_decay': trial_config['weight_decay'],
               'dropout_rate': trial_config['dropout_rate'],
               'clear_weight': trial_config['clear_weight'],
               'win_weight': trial_config['win_weight'],
               'potted_weight': trial_config['potted_weight'],
               'potted_head': trial_config['potted_head'],
               'potted_ordinal_weight': trial_config['potted_ordinal_weight'],
               'potted_ce_weighting': trial_config['potted_ce_weighting'],
               'potted_ce_beta': trial_config['potted_ce_beta'],
               'potted_label_smoothing': trial_config['potted_label_smoothing'],
               'pooling': trial_config['pooling'],
               'use_paper_features': trial_config['use_paper_features'],
               'selection_metric': args.selection_metric,
               'best_val_metrics': best_val_metrics,
               'test_metrics': without_predictions(test_metrics),
               'hyperparameters': hparams}

    return {'summary': summary,
            'trial_config': dict(trial_config),
            'history': history,
            'best_state': best_state,
            'test_predictions': test_metrics['predictions']}


def flatten_epoch_metrics(trial_name, epoch, learning_rate, train_metrics, val_metrics):

    row = {'trial_name': trial_name,
           'epoch': epoch,
           'learning_rate': learning_rate}

    add_flat_metrics(row, 'train', train_metrics)
    add_flat_metrics(row, 'val', val_metrics)

    return row


def add_flat_metrics(row, prefix, metrics):

    for key in ['loss', 'clear_loss', 'win_loss', 'potted_loss',
                'potted_ordinal_loss', 'potted_class_loss',
                'mean_macro_f1', 'mean_accuracy',
                'clean_target_min_margin', 'clean_target_mean_margin']:
        row[prefix + '_' + key] = metrics[key]

    for task in TASKS:
        task_prefix = prefix + '_' + task + '_'
        for key, value in metrics['task_metrics'][task].items():
            row[task_prefix + key] = value


def without_predictions(metrics):

    clean_metrics = dict(metrics)
    clean_metrics.pop('predictions', None)

    return clean_metrics


def should_log_epoch(args, epoch):

    return (args.log_every_epochs > 0 and
            (epoch % args.log_every_epochs == 0 or epoch == args.epochs))


def format_progress(trial_name, epoch, epochs, train_metrics, val_metrics, best_epoch):

    return (trial_name + ' epoch ' + str(epoch) + '/' + str(epochs) +
            ' train_f1=' + format_float(train_metrics['mean_macro_f1']) +
            ' val_f1=' + format_float(val_metrics['mean_macro_f1']) +
            ' val_acc=' + format_float(val_metrics['mean_accuracy']) +
            ' best_epoch=' + str(best_epoch))


def format_float(value):

    if value is None:
        return 'NA'

    return '{:.4f}'.format(value)


def write_csv(path, rows):

    if len(rows) == 0:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, data):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as output_file:
        json.dump(json_safe(data), output_file, indent=2, sort_keys=True)


def json_safe(value):

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)

    return value


def make_run_dir(args):

    run_name = args.run_name
    if run_name is None:
        run_name = 'BLFormer_' + datetime.now().strftime('%Y%m%d_%H%M%S')

    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def best_trial_key(result, selection_metric):

    metrics = result['summary']['best_val_metrics']

    if selection_metric in ['clear_accuracy', 'win_accuracy',
                            'potted_after_break_accuracy']:
        task = selection_metric.replace('_accuracy', '')
        return (metrics['task_metrics'][task]['accuracy'],
                metrics['task_metrics'][task]['macro_f1'],
                -metrics['loss'])
    if selection_metric == 'clean_target_min_margin':
        return (metrics['clean_target_min_margin'],
                metrics['mean_accuracy'],
                metrics['mean_macro_f1'],
                -metrics['loss'])
    if selection_metric == 'mean_accuracy':
        return (metrics['mean_accuracy'],
                metrics['mean_macro_f1'],
                -metrics['loss'])

    return (metrics['mean_macro_f1'],
            metrics['mean_accuracy'],
            -metrics['loss'])


def run_fixed_epoch_training(data, splits, args, device, trial_config, epochs):

    set_seed(args.seed + 1000)

    train_loader = make_loader(data, splits['final_train'], args.batch_size, True,
                               not args.disable_augmentation,
                               args.seed + 1000, args.num_workers, device)
    test_loader = make_loader(data, splits['test'], args.batch_size, False,
                              False, args.seed, args.num_workers, device)

    hparams = base_hyperparameters(args, trial_config)
    model = BLFormer(hparams).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=trial_config['learning_rate'],
                                  weight_decay=trial_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=max(1, epochs))
    loss_config = build_loss_config(data, splits['final_train'], trial_config, device)

    final_train_metrics = None
    for epoch in range(1, epochs + 1):
        final_train_metrics = train_one_epoch(model, train_loader, optimizer,
                                              scheduler, device, trial_config,
                                              loss_config, args)
        if should_log_epoch(args, epoch):
            print('final_full_train epoch ' + str(epoch) + '/' + str(epochs) +
                  ' train_acc=' + format_float(final_train_metrics['mean_accuracy']) +
                  ' train_f1=' + format_float(final_train_metrics['mean_macro_f1']))

    test_metrics = evaluate(model, test_loader, device, trial_config, loss_config,
                            return_predictions=True)
    state = clone_state_dict(model)
    parameter_count = sum([parameter.numel() for parameter in model.parameters()])

    summary = {'trial_name': trial_config['trial_name'],
               'epochs': epochs,
               'train_size': len(splits['final_train']),
               'parameter_count': parameter_count,
               'learning_rate': trial_config['learning_rate'],
               'weight_decay': trial_config['weight_decay'],
               'dropout_rate': trial_config['dropout_rate'],
               'clear_weight': trial_config['clear_weight'],
               'win_weight': trial_config['win_weight'],
               'potted_weight': trial_config['potted_weight'],
               'potted_head': trial_config['potted_head'],
               'potted_ordinal_weight': trial_config['potted_ordinal_weight'],
               'potted_ce_weighting': trial_config['potted_ce_weighting'],
               'potted_ce_beta': trial_config['potted_ce_beta'],
               'potted_label_smoothing': trial_config['potted_label_smoothing'],
               'pooling': trial_config['pooling'],
               'use_paper_features': trial_config['use_paper_features'],
               'train_metrics': final_train_metrics,
               'test_metrics': without_predictions(test_metrics),
               'hyperparameters': hparams}

    return {'summary': summary,
            'state': state,
            'test_predictions': test_metrics['predictions']}


def main():

    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args)
    data = load_processed_data(args.processed_path)
    splits = build_splits(data, args)
    configs = trial_configs(args)
    run_dir = make_run_dir(args)

    print('device:', device)
    print('processed_path:', args.processed_path)
    print('run_dir:', run_dir)
    print('search:', args.search)
    print('split_sizes:', {name: len(splits[name])
                           for name in ['train', 'val', 'final_train', 'test']})
    print('split_notes:', '; '.join(splits['notes']))

    write_json(os.path.join(run_dir, 'config.json'),
               {'args': vars(args),
                'splits': {key: value for key, value in splits.items()
                           if key in ['train', 'val', 'final_train', 'test']},
                'split_notes': splits['notes']})

    all_history = []
    search_summaries = []
    best_result = None

    for trial_index, config in enumerate(configs):
        print('trial:', config['trial_name'], '(' + str(trial_index + 1) +
              '/' + str(len(configs)) + ')')
        result = run_trial(data, splits, args, device, config, trial_index)
        search_summaries.append(result['summary'])
        all_history.extend(result['history'])

        if (best_result is None or
                best_trial_key(result, args.selection_metric) >
                best_trial_key(best_result, args.selection_metric)):
            best_result = result

        write_csv(os.path.join(run_dir, 'history.csv'), all_history)
        write_json(os.path.join(run_dir, 'search_results.json'), search_summaries)
        write_csv(os.path.join(run_dir, 'search_results.csv'),
                  [flatten_trial_summary(summary) for summary in search_summaries])

    validation_checkpoint_path = os.path.join(run_dir, 'BLFormer_selected_val.pt')
    torch.save({'model_state_dict': best_result['best_state'],
                'hyperparameters': best_result['summary']['hyperparameters'],
                'trial': best_result['summary'],
                'args': vars(args),
                'split_notes': splits['notes']},
               validation_checkpoint_path)

    validation_predictions_path = os.path.join(run_dir, 'test_predictions_selected_val.csv')
    write_csv(validation_predictions_path, best_result['test_predictions'])

    final_result = None
    final_checkpoint_path = None
    final_predictions_path = None

    if args.final_train_on_full_train:
        print('final_full_train: trial=' + best_result['summary']['trial_name'] +
              ' epochs=' + str(best_result['summary']['best_epoch']))
        final_result = run_fixed_epoch_training(data,
                                                splits,
                                                args,
                                                device,
                                                best_result['trial_config'],
                                                best_result['summary']['best_epoch'])
        final_checkpoint_path = os.path.join(run_dir, 'BLFormer_final_full_train.pt')
        torch.save({'model_state_dict': final_result['state'],
                    'hyperparameters': final_result['summary']['hyperparameters'],
                    'trial': final_result['summary'],
                    'selected_trial': best_result['summary'],
                    'args': vars(args),
                    'split_notes': splits['notes']},
                   final_checkpoint_path)
        final_predictions_path = os.path.join(run_dir,
                                              'test_predictions_final_full_train.csv')
        write_csv(final_predictions_path, final_result['test_predictions'])

    final_summary = {'best_trial': best_result['summary'],
                     'validation_checkpoint_path': validation_checkpoint_path,
                     'validation_predictions_path': validation_predictions_path,
                     'final_full_train': None if final_result is None else final_result['summary'],
                     'final_checkpoint_path': final_checkpoint_path,
                     'final_predictions_path': final_predictions_path,
                     'search_results_path': os.path.join(run_dir, 'search_results.json'),
                     'history_path': os.path.join(run_dir, 'history.csv'),
                     'split_sizes': {name: len(splits[name])
                                     for name in ['train', 'val', 'final_train', 'test']},
                     'split_notes': splits['notes']}
    write_json(os.path.join(run_dir, 'summary.json'), final_summary)

    print('best_trial:', best_result['summary']['trial_name'])
    print('best_val_mean_macro_f1:',
          format_float(best_result['summary']['best_val_metrics']['mean_macro_f1']))
    print('selected_val_test_mean_macro_f1:',
          format_float(best_result['summary']['test_metrics']['mean_macro_f1']))
    print('selected_val_test_mean_accuracy:',
          format_float(best_result['summary']['test_metrics']['mean_accuracy']))
    if final_result is not None:
        print('final_full_train_test_mean_macro_f1:',
              format_float(final_result['summary']['test_metrics']['mean_macro_f1']))
        print('final_full_train_test_mean_accuracy:',
              format_float(final_result['summary']['test_metrics']['mean_accuracy']))
    print('checkpoint:', final_checkpoint_path or validation_checkpoint_path)
    print('summary:', os.path.join(run_dir, 'summary.json'))


def flatten_trial_summary(summary):

    row = {'trial_name': summary['trial_name'],
           'trial_index': summary['trial_index'],
           'best_epoch': summary['best_epoch'],
           'epochs_ran': summary['epochs_ran'],
           'parameter_count': summary['parameter_count'],
           'learning_rate': summary['learning_rate'],
           'weight_decay': summary['weight_decay'],
           'dropout_rate': summary['dropout_rate'],
           'clear_weight': summary['clear_weight'],
           'win_weight': summary['win_weight'],
           'potted_weight': summary['potted_weight'],
           'potted_head': summary['potted_head'],
           'potted_ordinal_weight': summary['potted_ordinal_weight'],
           'potted_ce_weighting': summary['potted_ce_weighting'],
           'potted_ce_beta': summary['potted_ce_beta'],
           'potted_label_smoothing': summary['potted_label_smoothing'],
           'pooling': summary['pooling'],
           'use_paper_features': summary['use_paper_features'],
           'selection_metric': summary['selection_metric'],
           'best_val_loss': summary['best_val_metrics']['loss'],
           'best_val_mean_macro_f1': summary['best_val_metrics']['mean_macro_f1'],
           'best_val_mean_accuracy': summary['best_val_metrics']['mean_accuracy'],
           'best_val_clean_target_min_margin': (
               summary['best_val_metrics']['clean_target_min_margin']),
           'test_loss': summary['test_metrics']['loss'],
           'test_mean_macro_f1': summary['test_metrics']['mean_macro_f1'],
           'test_mean_accuracy': summary['test_metrics']['mean_accuracy'],
           'test_clean_target_min_margin': (
               summary['test_metrics']['clean_target_min_margin'])}

    for task in TASKS:
        row['test_' + task + '_accuracy'] = (
            summary['test_metrics']['task_metrics'][task]['accuracy'])
        row['test_' + task + '_macro_f1'] = (
            summary['test_metrics']['task_metrics'][task]['macro_f1'])

    return row


if __name__ == '__main__':
    main()
