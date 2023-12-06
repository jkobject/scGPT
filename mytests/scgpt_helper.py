from scgpt.tokenizer import tokenize_and_pad_batch, GeneVocab, random_mask_value

from scgpt.trainer import prepare_data, prepare_dataloader
from scgpt import logger
from scgpt.trainer import train as scgpt_train
from scgpt.trainer import evaluate as scgpt_validate
from scgpt.utils import set_seed, add_file_handler

import time
import wandb
import scanpy as sc
from pathlib import Path
import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
import copy
import gc
import torch
from typing import Union


def prepare_dataset(
    dataset,
    vocab,
    batch_size,
    epoch=5,
    mask_ratio=0.4,
    mask_value=-1,
    n_hvg=2000,
    pad_token="<pad>",
    pad_value=-2,
    per_seq_batch_sample=True,
    test_size=0.2,
):
    all_counts = (
        dataset.layers["X_binned"]
        if issparse(dataset.layers["X_binned"])
        else dataset.layers["X_binned"]
    )
    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
        all_counts,
        dataset.obs["cell_type"].values,
        dataset.obs["batch_id"].values,
        test_size=test_size,
        shuffle=True,
    )
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        dataset.var["gene_ids"].values,
        max_len=n_hvg + 1,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        dataset.var["gene_ids"].values,
        max_len=n_hvg + 1,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    train_data_pt, valid_data_pt = prepare_data(
        masked_values_train,
        masked_values_valid,
        train_batch_labels,
        valid_batch_labels,
        task="annotation",
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
        epoch=epoch,
        train_celltype_labels=train_celltype_labels,
        valid_celltype_labels=valid_celltype_labels,
        sort_seq_batch=per_seq_batch_sample,
    )

    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )
    return train_loader, valid_loader


def setup(dataset_name, save_path, config, seed=42):
    save_dir = Path(f"{save_path}_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    add_file_handler(logger, save_dir / "run.log")
    logger.info(f"save to {save_dir}")

    return save_dir


def load_dataset(dataset, vocab):
    dataset.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in dataset.var["gene_symbols"]
    ]
    gene_ids_in_vocab = np.array(dataset.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )

    dataset = dataset[:, dataset.var["id_in_vocab"] >= 0]
    dataset.var["gene_ids"] = vocab(dataset.var["gene_symbols"].tolist())
    return dataset


def fine_tune(model, train_loader, valid_loader, epochs, scheduler, save_folder):
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        scgpt_train()
        val_loss, val_mre = scgpt_validate()
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")
        scheduler.step()
    # TODO:
    logger.info(f"Saving model to {save_folder}")
    torch.save(
        best_model.state_dict(),
        self.save_folder / f"model_e{best_model_epoch}.pt",
    )


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def random_mask_value_weighted(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    do_not_pad_index: Union[torch.Tensor, np.ndarray] = np.array([]),
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        non_padding_idx = np.setdiff1d(non_padding_idx, do_not_pad_index)
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()


config = {
    "GEPC": True,  # Gene expression modelling for cell objective
    "ecs_thres": 0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    "dab_weight": 1.0,  # DAR objective weight for batch correction
    "mask_ratio": 0.4,  # Default mask ratio
    "epochs": 15,  # Default number of epochs for fine-tuning
    "n_bins": 51,  # Default number of bins for value binning in data pre-processing
    "lr": 1e-4,  # Default learning rate for fine-tuning
    "batch_size": 64,  # Default batch size for fine-tuning
    "layer_size": 128,
    "nhead": 8,  # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    "dropout": 0.2,  # Default dropout rate during model fine-tuning
    "schedule_ratio": 0.9,  # Default rate for learning rate decay
    "save_eval_interval": 5,  # Default model evaluation interval
    "log_interval": 100,  # Default log interval
    "pre_norm": False,  # Default setting
    "dsbn": True,  # Default setting
    "amp": True,  # # Default setting: Automatic Mixed Precision
    "scheduler_interval": 100,
    "scheduler_factor": 0.99,
    "warmup_ratio_or_step": 10000.0,
    "no_cls": True,
    "no_cce": True,
    "fp16": True,
    "fast_transformer": True,
    "nlayers": 12,
    "embsize": 512,
    "d_hid": 512,
    "n_layers_cls": 3,
    "max_seq_len": 1200,
}
