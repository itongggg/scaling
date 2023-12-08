import os
import sys
import math
import glob
import time
from pathlib import Path
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from functools import partial
from pathlib import Path
from typing import Tuple, Optional
from lit_llama.utils import save_model_checkpoint, get_logps
from lit_llama.packed_dataset import PackedDataset, CombinedDataset
from lit_llama.model import Block, LLaMA, LLaMAConfig
import lightning as L
from lightning.fabric.strategies import DDPStrategy, FSDPStrategy

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import numpy as np
from loguru import logger
import colorful as cf
from utils.tsdb_assistant import InfluxDBHelper
from utils.util import load_yaml


HOST_NAME = os.uname()[1]
DISABLE_IF = False
IBH = None
eval_iters = 100
def main(
        config_file: str = None
) -> None:
    config = load_yaml(file_path=config_file, advanced=True, show=True)
    logger.info(cf.blue(config))

    if config.training_config.strategy.lower() == "ddp":
        strategy = DDPStrategy(
            process_group_backend='nccl'
        )
    elif config.training_config.strategy.lower() == "fsdp":
        auto_wrap_policy = partial(transformer_auto_wrap_policy,
                                   transformer_layer_cls={Block})
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block}
        )
    else:
        raise Exception("Only support FSDP and DDP strategy now.")
    
    fabric = L.Fabric(
        accelerator=config.training_config.accelerator,
        strategy=strategy,
        devices=config.training_config.devices,
        num_nodes=config.training_config.num_nodes,
        precision=config.training_config.precision,
    )

    # Uncomments it for single node run with python.
    # Comments it for lightning run.
    fabric.launch()
    fabric.seed_everything(config.training_config.fabric_seed)
    model_config = LLaMAConfig.from_name(config.training_config.model_name)

    fabric.print(f"node rank:{fabric.node_rank}, \
                local rank:{fabric.local_rank}, \
                global rank:{fabric.global_rank}, \
                world size:{fabric.world_size}, \
                host name:{HOST_NAME}")
    if fabric.global_rank == 0:
        os.makedirs(config.training_config.output_dir, exist_ok=True)

        try:
            if not DISABLE_IF and fabric.global_rank == 0:
                hparams = {}

                hparams["world_size"] = fabric.world_size
                hparams["node_rank"] = fabric.node_rank
                hparams["global_rank"] = fabric.global_rank
                hparams["local_rank"] = fabric.local_rank
                hparams["host_name"] = HOST_NAME
                hparams["train_data_path"] = str(config.training_config.train_data_dir)
                hparams["val_data_path"] = str(config.training_config.val_data_dir)
                for k, v in config.hyper_parameters.items():
                    hparams[k] = v
                for k, v in config.data_config.items():
                    hparams[f"data_{k}"] = v
                global IBH
                IBH = InfluxDBHelper(config_file=config.log_config.config_file,
                                     exp_name=config.exp_name,
                                     hparams=hparams) 
        except Exception as e:
            fabric.print(f"Init ibh error: {e}")

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=config.hyper_parameters.micro_batch_size,
        block_size=model_config.block_size,
        fabric=fabric,
        train_data_dir=config.training_config.train_data_dir,
        val_data_dir=config.training_config.val_data_dir,
        seed=config.training_config.dataloader_seed,
        n_chunks=config.training_config.dataloader_chunks,
        data_config=config.data_config,
        match_pattern=config.training_config.dataloader_match_pattern,
        num_files=config.training_config.num_files
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(
            train_dataloader, val_dataloader
        )

    with fabric.device:
        # torch.set_default_dtype(torch.bfloat16)
        torch.set_default_dtype(torch.float32)
        model = LLaMA(model_config)
        state_dict = fabric.load(config.training_config.checkpoint_path)
        model.load_state_dict(state_dict)
        new_model_config = LLaMAConfig.from_name(config.training_config.new_model_name)
        model.grow_model(new_model_config)
        model.freeze_old_params()
        model._init_new_weights()
        # model.apply(model._init_weights)
        

    model = fabric.setup_module(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.hyper_parameters.learning_rate,
        weight_decay=config.hyper_parameters.weight_decay,
        betas=(config.hyper_parameters.beta1, config.hyper_parameters.beta2),
    )
    optimizer = fabric.setup_optimizers(optimizer)
    devices = config.training_config.devices if type(config.training_config.devices) == int else len(config.training_config.devices)
    process_batch_size = config.hyper_parameters.batch_size // devices
    gradient_accumulation_iters = process_batch_size // config.hyper_parameters.micro_batch_size
    logger.info("start training")
    train(fabric=fabric,
          model=model,
          optimizer=optimizer,
          train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          grad_accum_steps=gradient_accumulation_iters,
          grad_clip=config.hyper_parameters.grad_clip,
          decay_lr=config.hyper_parameters.decay_lr,
          learning_rate=config.hyper_parameters.learning_rate,
          warmup_iters=config.hyper_parameters.warmup_iters,
          lr_decay_iters=config.hyper_parameters.lr_decay_iters,
          min_lr=config.hyper_parameters.min_lr,
          devices=devices,
          num_nodes=config.training_config.num_nodes,
          save_interval=config.training_config.save_interval,
          eval_interval=config.training_config.eval_interval,
          log_interval=config.training_config.log_interval,
          micro_batch_size=config.hyper_parameters.micro_batch_size,
          max_iters=config.hyper_parameters.max_iters,
          out_dir=config.training_config.output_dir)


def train(
        fabric: L.Fabric,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,   
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        grad_accum_steps: int,
        grad_clip: float,
        decay_lr: bool,
        learning_rate: float,
        warmup_iters: int,
        lr_decay_iters: int,
        min_lr: float,
        devices: int,
        num_nodes: int,
        save_interval: int,
        eval_interval: int,
        log_interval: int,
        micro_batch_size: int,
        max_iters: int,
        out_dir: str,
        stage1: bool = True,
        kl_ctl: float = 0.01
) -> None:

    step_count = 0

    step_time = 0.0
    tokens = 0
    tokens_sec = 0.0
    prev_t1 = time.time()
    total_time = 0.0
    for iter_num, train_data in enumerate(train_dataloader):
        t0 = time.time()

        lr = get_lr(it=iter_num, learning_rate=learning_rate, warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters, min_lr=min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        input_ids = train_data[:, 0: model.config.block_size].contiguous()
        targets = train_data[:, 1: model.config.block_size + 1].contiguous()

        is_accumulating = (iter_num + 1) % grad_accum_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            if stage1:
                with torch.no_grad():
                    orig_logits = model(input_ids, stage1=stage1)
                    kl_penalty = kl_ctl * (get_logps(orig_logits, targets).view(-1) - get_logps(logits, targets).view(-1)).mean()
            else:
                kl_penalty = 0
                model.unfreeze_old_params()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            ) + kl_penalty
            fabric.backward(loss / grad_accum_steps)
        t1 = time.time()

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer=optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            step_count += 1
            t1 = time.time()

            if val_dataloader is not None and step_count % eval_interval == 0:
                val_loss = validate(fabric=fabric, 
                                    model=model, 
                                    val_dataloader=val_dataloader)
                fabric.print(f"step {iter_num}: Validation loss: {val_loss}")
                fabric.barrier()
                fabric.log_dict(
                    {"iter": iter_num, "val_loss": val_loss, "step": step_count, "lr": lr}
                )
            if step_count % save_interval == 0:
                fabric.print(f"Saving model at step {step_count} to {out_dir}")
                save_model_checkpoint(
                    fabric,
                    model,
                    os.path.join(out_dir, f"iter-{iter_num:06d}.pt"),
                )
        
        dt = t1 - t0
        tokens += micro_batch_size * model.config.block_size
        step_time += t1 - prev_t1
        prev_t1 = t1

        if iter_num % log_interval == 0:
            tokens_per_sec = f"{tokens / step_time:.0f}" if not is_accumulating else "-"
            fabric.log_dict(
                {
                    "iter": iter_num,
                    "lr": lr,
                    "train_loss": loss.item(),
                    "step": step_count,
                    "tokens/sec": tokens_per_sec,
                }
            )
            total_time += dt
            if IBH is not None:
                fabric.print("IBH track metrics")
                IBH.track_metrics(metrics={"iter": iter_num,
                                           "train_loss": loss.item(),
                                           "step": step_count,
                                           "lr": lr,
                                           "time": dt*1000,
                                           "total_time": total_time/3600,
                                           "speed": (tokens * devices * num_nodes) / step_time,
                                           "train_progress": iter_num/max_iters,
                                           "kl_penalty": kl_penalty,
                                           },
                                  subset="train")
            fabric.print(
                f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, speed: {tokens_per_sec} toks/s/device"
            )

        if not is_accumulating:
            tokens = 0
            step_time = 0.0
        if abs(kl_penalty) <= 1e-4 or iter_num > 30000:
            stage1 = False
            fabric.print("Stage 1 finished.")
        if iter_num > max_iters:
            break


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    eval_iters: int = eval_iters
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        input_ids = val_data[:, 0: model.config.block_size].contiguous()
        targets = val_data[:, 1: model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: str,
    fabric,
    shuffle: bool = True,
    seed: int = 12345,
    n_chunks: int = 4,
    data_config: object = None,
    match_pattern: str = "*",
    is_validate: bool = False,
    num_files: int = 8000
) -> DataLoader:
    datasets = []
    global config
    print(data_config.items())
    for prefix, _ in data_config.items():
        filenames = glob.glob(os.path.join(data_dir, prefix+match_pattern))
        if is_validate:
            filenames = filenames[-50:]
        else:
            filenames =filenames[5000:num_files]
        logger.info(f"Total filenames: {len(filenames)}")
        # Wrap is True, means allow repeat sampling
        dataset = PackedDataset(
            filenames, n_chunks=n_chunks, block_size=block_size, shuffle=shuffle, seed=seed,
            num_processes=fabric.world_size, process_rank=fabric.global_rank, wrap=True
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config.items()]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: str = "data/lit-redpajama",
    val_data_dir: Optional[str] = None,
    seed: int = 12345,
    n_chunks: int = 4,
    data_config: object = None,
    match_pattern: str = "*",
    num_files: int = 5000
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well

    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        n_chunks=n_chunks,
        data_config=data_config,
        match_pattern=match_pattern,
        num_files=num_files
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            n_chunks=n_chunks,
            data_config=data_config,
            match_pattern=match_pattern,
            is_validate=True,
            num_files=num_files
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader
# learning rate decay scheduler (cosine with warmup)
def get_lr(it, learning_rate, warmup_iters, min_lr, lr_decay_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)
    