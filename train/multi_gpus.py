import torch
import math
import numpy as np
from torch_geometric.data import Data
from graph_batch import TriBatch
from copy import deepcopy


def move_batch_to_device(batch_data, device):
    """Move all tensors and TriBatch to the target device."""

    out = {}
    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif hasattr(value, "to"):  # For graph_data (TriBatch)
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def split_batches_for_devices(batch_list, num_devices):
    """Split input_ids, attention_mask, and graph_data into N parts."""
    # Calculate batch size per device
    total_batches = len(batch_list)
    batches_per_device = math.ceil(total_batches / num_devices)

    # Split batches into sub-batches for each device
    sub_batches = []
    for i in range(num_devices):
        start_idx = i * batches_per_device
        end_idx = min((i + 1) * batches_per_device, total_batches)

        if start_idx >= total_batches:
            # Add empty batch if we've run out of data
            sub_batches.append([])
            continue

        device_batch = batch_list[start_idx:end_idx]

        sub_batches.append(device_batch)

    return sub_batches


def multi_gpu_inference(model, batch_data):
    """
    Runs inference on multiple GPUs by assigning each GPU a sub-batch.
    Assumes model is already on default device (cuda:0).
    """
    num_devices = torch.cuda.device_count()
    if num_devices < 2:
        # Fallback to single device inference
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            return model(**batch_data)["logits"].detach().cpu()

    model.eval()
    sub_batches = split_batches_for_devices(
        batch_list=batch_data, num_devices=num_devices
    )

    outputs = [None] * num_devices

    # Parallelism via threading (simpler than multiprocessing for GPU tasks)
    import threading

    threads = []

    def worker(i):
        device = torch.device(f"cuda:{i}")
        local_model = deepcopy(model).to(device)
        local_batch = move_batch_to_device(sub_batches[i][0], device)

        with torch.no_grad():
            outputs[i] = local_model(**local_batch)["logits"].detach().cpu()

    for i in range(num_devices):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return torch.cat(outputs, dim=0)
