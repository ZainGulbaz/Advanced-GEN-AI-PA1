import os
from pathlib import Path
import numpy as np
import torch


def load_token_array(path) -> np.memmap:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size % 2 != 0:
        raise ValueError("token stream has an odd byte length")
    arr = np.memmap(path, mode="r", dtype=np.dtype("<u2"))
    if arr.ndim != 1:
        raise ValueError("token stream must be one-dimensional")
    return arr


def get_batch(dataset, batch_size, sequence_length, device, generator):
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    if dataset.ndim != 1:
        raise ValueError("dataset must be one-dimensional")
    if len(dataset) <= sequence_length:
        raise ValueError("dataset does not contain enough tokens for one input/target window")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")

    max_start = len(dataset) - sequence_length
    starts = torch.randint(0, max_start, (batch_size,), generator=generator, device="cpu")
    windows = [dataset[int(s): int(s) + sequence_length + 1].astype(np.int64, copy=False)
               for s in starts.tolist()]
    batch = torch.from_numpy(np.stack(windows, axis=0)).to(torch.long)
    return batch[:, :-1].to(device), batch[:, 1:].to(device)


def save_checkpoint(model, optimizer, next_step, train_generator, val_generator, out) -> None:
    if next_step < 0:
        raise ValueError("next_step must be non-negative")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_step": next_step,
        "train_generator": train_generator.get_state(),
        "val_generator": val_generator.get_state(),
    }, out)


def load_checkpoint(src, model, optimizer, train_generator, val_generator):
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "next_step", "train_generator", "val_generator"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"checkpoint missing keys: {sorted(missing)}")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    train_generator.set_state(checkpoint["train_generator"])
    val_generator.set_state(checkpoint["val_generator"])
    return checkpoint["next_step"]
