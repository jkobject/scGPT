from scgpt.tokenizer import tokenize_and_pad_batch

from scgpt.trainer import prepare_data, prepare_dataloader
from scgpt import logger
from scgpt.trainer import train as scgpt_train
from scgpt.trainer import evaluate as scgpt_validate
from scgpt.utils import set_seed, add_file_handler
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
import time
import wandb
from pathlib import Path
import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
import copy
import torch
from torch import nn
import gc


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
    task="annotation",
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
        dataset.obs["cell_type"].astype("category").cat.codes.values,
        dataset.obs["batch_id"].astype("category").cat.codes.values,
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
    train_data_pt, valid_data_pt = prepare_data(
        tokenized_train,
        tokenized_valid,
        train_batch_labels,
        valid_batch_labels,
        task=task,
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


def fine_tune(
    model,
    config,
    dataset,
    vocab,
    epochs,
    batch_size,
    save_folder,
    device,
    task="annotation",
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        eps=1e-4 if config["amp"] else 1e-8,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 1, gamma=config["schedule_ratio"]
    )
    config["task"] = "annotation"
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])

    run = wandb.init(
        config=config,
        project="scGPT",
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    wandb.watch(model)

    best_val_loss = float("inf")

    best_model = None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        data_loader, valid_loader = prepare_dataset(
            dataset,
            vocab,
            batch_size,
            epoch=epoch,
            n_hvg=config["n_hvg"],
            test_size=0.2,
            mask_ratio=config["mask_ratio"],
        )
        scgpt_train(
            model,
            data_loader,
            vocab,
            criterion_gep_gepc=masked_mse_loss,
            criterion_dab=torch.nn.CrossEntropyLoss(),
            criterion_cls=nn.CrossEntropyLoss(),
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            logger=logger,
            epoch=epoch,
        )
        val_loss = scgpt_validate(
            model,
            valid_loader,
            vocab,
            criterion_gep_gepc=masked_mse_loss,
            criterion_dab=torch.nn.CrossEntropyLoss(),
            criterion_cls=nn.CrossEntropyLoss(),
            device=device,
            config=config,
            logger=logger,
            epoch=epoch,
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")
        scheduler.step()
    logger.info(f"Saving model to {save_folder}")

    torch.save(
        best_model.state_dict(),
        save_folder + f"model_e{best_model_epoch}.pt",
    )
    wandb.use_artifact(save_folder + f"model_e{best_model_epoch}.pt", type="model")

    wandb.finish()
    gc.collect()
    return best_model


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")



config = {
    "GEPC": False,  # Gene expression modelling for cell objective
    "MVC": True,
    "GEP": True,
    "mask_ratio": 0.4,  # Default mask ratio
    "DSBN": True,  # Default setting
    "ECS": False,
    "ecs_thres": 0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    "CLS": True,
    "USE_CLS": False,
    "USE_GENERATIVE_TRAINING": True,
    "DAR": False,
    "dab_weight": 1.0,  # DAR objective weight for batch correction
    "USE_CCE": False,
    "explicit_zero_prob": False,
    #
    "epochs": 6,  # Default number of epochs for fine-tuning
    "lr": 1e-4,  # Default learning rate for fine-tuning
    "schedule_ratio": 0.9,  # Default rate for learning rate decay
    "save_eval_interval": 5,  # Default model evaluation interval
    "log_interval": 100,  # Default log interval
    "pre_norm": False,  # Default setting
    "amp": True,  # # Default setting: Automatic Mixed Precision
    "dropout": 0.2,  # Default dropout rate during model fine-tuning
    "batch_size": 6,
    "eval_batch_size": 8,
    "log_interval": 9000,
    "save_interval": 27000,
    #
    "scheduler_interval": 100,
    "scheduler_factor": 0.99,
    "warmup_ratio_or_step": 10000.0,
    #
    "no_cce": True,
    #
    "fast_transformer": True,
    "n_layers_cls": 3,
    "fp16": True,
    "nlayers": 12,
    "embsize": 512,
    "d_hid": 512,
    "nhead": 8,  # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    "layer_size": 128,
    #
    "max_seq_len": 1200,
    "n_hvg": 2000,
    "n_bins": 51,  # Default number of bins for value binning in data pre-processing
    "mask_value": -1,
    "pad_value": -2,
    "pad_token": "<pad>",
    "input_style": "binned",
    #
    "valid_size_or_ratio": 0.003,
    "dist_backend": "nccl",
    "grad_accu_steps": 1,
    "input_emb_style": "continuous",
    "training_tasks": "both",
    "trunc_by_sample": True,
    "rank": 0,
    #
    "world_size": 16,
    "distributed": True,
    "local_rank": 0,
    "gpu": 0,
    "task": "annotation",
    "use_batch_labels": True,
    "use_mod": False,
}
