# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, DINO, and BIOT code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/ycq091044/BIOT
# ---------------------------------------------------------

import io
import hashlib
import os
import math
import time
import json
import glob
import warnings
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path
import argparse
from typing import Optional

import torch
import torch.distributed as dist
from torch import inf
import h5py

from tensorboardX import SummaryWriter
from data_processor.dataset import ShockDataset
import pickle
from scipy.signal import resample
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

TUEV_BIPOLAR_16_CH = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

ISRUC_BIPOLAR_CH = ["F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1"]
ISRUC_LABRAM_CH = ["F3", "C3", "O1", "F4", "C4", "O2"]

# Exact channel order emitted by shared EEG preprocessing/preprocessing/preprocessing_physio.py,
# normalized to the LaBraM standard_1020 vocabulary. The serialized PhysioNet-MI
# LMDB does not carry channel metadata, so this in-repository constant is the
# authoritative manifest for the stored tensor rows.
PHYSIONET_MI_LABRAM_CH = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2",
    "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
    "F6", "F8", "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8", "P7", "P5",
    "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2", "IZ",
]


class ISRUCSequenceLoader(torch.utils.data.Dataset):
    """Subject-wise ISRUC sequences with shape [20, 6, 30, 200]."""

    def __init__(self, root, subjects):
        self.root = Path(root)
        self.subjects = tuple(int(subject) for subject in subjects)
        self.samples = self._index_samples()
        if not self.samples:
            raise ValueError(f"ISRUC split is empty under {self.root}")

    @staticmethod
    def _numeric_files(directory):
        files = {}
        for path in Path(directory).glob("*.npy"):
            stem = path.stem
            try:
                index = int(stem.rsplit("-", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"ISRUC filename must end in a numeric sequence index: {path}") from exc
            if index in files:
                raise ValueError(f"Duplicate ISRUC sequence index {index} in {directory}")
            files[index] = path
        return files

    def _index_samples(self):
        samples = []
        for subject in self.subjects:
            seq_dir = self.root / "seq" / f"ISRUC-group1-{subject}"
            label_dir = self.root / "labels" / f"ISRUC-group1-{subject}"
            if not seq_dir.is_dir() or not label_dir.is_dir():
                raise FileNotFoundError(f"Missing ISRUC directories for subject {subject}")
            signals = self._numeric_files(seq_dir)
            labels = self._numeric_files(label_dir)
            if signals.keys() != labels.keys():
                raise RuntimeError(f"ISRUC signal/label indices differ for subject {subject}")
            samples.extend((subject, index, signals[index], labels[index]) for index in sorted(signals))
        return samples

    def __len__(self):
        return len(self.samples)

    def get_ch_names(self):
        return list(ISRUC_LABRAM_CH)

    def split_metadata(self):
        return {
            "subjects": list(self.subjects),
            "sample_count": len(self.samples),
            "shape": [20, 6, 30, 200],
            "bipolar_channels": list(ISRUC_BIPOLAR_CH),
            "labram_channel_names": list(ISRUC_LABRAM_CH),
        }

    def __getitem__(self, index):
        subject, sequence_index, signal_path, label_path = self.samples[index]
        signal = np.load(signal_path)
        label = np.load(label_path)
        if tuple(signal.shape) == (20, 6, 6000):
            signal = signal.reshape(20, 6, 30, 200)
        if tuple(signal.shape) != (20, 6, 30, 200):
            raise ValueError(f"ISRUC subject={subject} index={sequence_index} signal shape={signal.shape}")
        if tuple(label.shape) != (20,):
            raise ValueError(f"ISRUC subject={subject} index={sequence_index} label shape={label.shape}")
        if not np.isfinite(signal).all():
            raise ValueError(f"ISRUC subject={subject} index={sequence_index} has non-finite signal values")
        if not np.isin(label, [0, 1, 2, 3, 4]).all():
            raise ValueError(f"ISRUC subject={subject} index={sequence_index} has unexpected labels")
        return torch.FloatTensor(signal), torch.LongTensor(label)


def prepare_ISRUC_dataset(root):
    train_dataset = ISRUCSequenceLoader(root, range(1, 81))
    val_dataset = ISRUCSequenceLoader(root, range(81, 91))
    test_dataset = ISRUCSequenceLoader(root, range(91, 101))
    print(
        f"[ISRUC] train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
        f"shape={[20, 6, 30, 200]} channels={ISRUC_BIPOLAR_CH}"
    )
    return train_dataset, test_dataset, val_dataset


class PhysioNetMILoader(torch.utils.data.Dataset):
    """LaBraM-only reader for the shared EEG preprocessing PhysioNet-MI serialized LMDB.

    The preprocessing contract stores each example as [64, 4, 200] in raw
    microvolt scale. The LaBraM engine performs the single /100 input scaling;
    this loader deliberately returns the raw tensor so scaling cannot happen
    twice.
    """

    expected_shape = (64, 4, 200)
    label_min = 0
    label_max = 3

    def __init__(self, root, mode="train"):
        self.root = os.fspath(root)
        self.mode = str(mode)
        self.channel_names = list(PHYSIONET_MI_LABRAM_CH)
        if len(self.channel_names) != self.expected_shape[0]:
            raise RuntimeError(
                f"PhysioNet-MI manifest has {len(self.channel_names)} channels; "
                f"expected {self.expected_shape[0]}"
            )
        unknown_channels = [name for name in self.channel_names if name not in standard_1020]
        if unknown_channels or len(set(self.channel_names)) != len(self.channel_names):
            raise RuntimeError(
                f"Invalid PhysioNet-MI LaBraM channel manifest: unknown={unknown_channels} "
                f"duplicates={len(self.channel_names) - len(set(self.channel_names))}"
            )
        try:
            import lmdb  # local import so non-LMDB datasets do not require it
        except ImportError as exc:
            raise ImportError("PhysioNet-MI loading requires the 'lmdb' package") from exc
        self._lmdb = lmdb
        self.db = None
        with lmdb.open(
            self.root, readonly=True, lock=False, readahead=False, meminit=False
        ).begin(write=False) as txn:
            raw_keys = txn.get(b"__keys__")
        if raw_keys is None:
            raise KeyError(f"PhysioNet-MI LMDB missing '__keys__' in {self.root}")
        split_index = pickle.loads(raw_keys)
        if not isinstance(split_index, dict) or self.mode not in split_index:
            available = list(split_index) if isinstance(split_index, dict) else type(split_index).__name__
            raise KeyError(f"PhysioNet-MI LMDB missing split {self.mode!r}; available: {available}")
        self.keys = list(split_index[self.mode])
        if not self.keys:
            raise ValueError(f"PhysioNet-MI split {self.mode!r} is empty in {self.root}")

        first_key = self.keys[0]
        encoded_key = first_key.encode() if isinstance(first_key, str) else first_key
        with lmdb.open(
            self.root, readonly=True, lock=False, readahead=False, meminit=False
        ).begin(write=False) as txn:
            first_raw = txn.get(encoded_key)
        if first_raw is None:
            raise KeyError(f"PhysioNet-MI LMDB key not found: {first_key!r}")
        self.sample_shape = self._validate_sample(pickle.loads(first_raw), first_key)

    def __len__(self):
        return len(self.keys)

    def get_ch_names(self):
        return list(self.channel_names)

    def split_metadata(self):
        digest = hashlib.sha256()
        for key in self.keys:
            encoded = key.encode("utf-8") if isinstance(key, str) else bytes(key)
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        subjects = sorted({str(key).split("R", 1)[0] for key in self.keys})
        return {
            "mode": self.mode,
            "sample_count": len(self.keys),
            "key_sha256": digest.hexdigest(),
            "sample_shape": list(self.sample_shape),
            "subject_count": len(subjects),
            "subjects": subjects,
            "channel_names": list(self.channel_names),
        }

    def _validate_sample(self, sample, key):
        if not isinstance(sample, dict) or "sample" not in sample or "label" not in sample:
            raise ValueError(f"PhysioNet-MI record {key!r} must contain sample and label fields")
        signal = np.asarray(sample["sample"])
        if tuple(signal.shape) != self.expected_shape:
            raise ValueError(
                f"PhysioNet-MI record {key!r} expected shape {self.expected_shape}, "
                f"got {tuple(signal.shape)}"
            )
        if not np.isfinite(signal).all():
            raise ValueError(f"PhysioNet-MI record {key!r} contains NaN or Inf")
        label_values = np.asarray(sample["label"]).reshape(-1)
        if label_values.size != 1:
            raise ValueError(f"PhysioNet-MI record {key!r} must contain one scalar label")
        label = int(label_values[0])
        if not self.label_min <= label <= self.label_max:
            raise ValueError(
                f"PhysioNet-MI record {key!r} label must be in "
                f"[{self.label_min}, {self.label_max}], got {label}"
            )
        return tuple(signal.shape)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["db"] = None
        return state

    def _get_db(self):
        if self.db is None:
            self.db = self._lmdb.open(
                self.root,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
            )
        return self.db

    def __getitem__(self, index):
        key = self.keys[index]
        encoded_key = key.encode() if isinstance(key, str) else key
        with self._get_db().begin(write=False) as txn:
            raw = txn.get(encoded_key)
        if raw is None:
            raise KeyError(f"PhysioNet-MI LMDB key not found: {key!r}")
        sample = pickle.loads(raw)
        self._validate_sample(sample, key)
        # Keep raw storage scale here. engine_for_finetuning.py owns /100.
        return torch.FloatTensor(np.asarray(sample["sample"])), int(sample["label"])


def prepare_PhysioNet_MI_dataset(root):
    train_dataset = PhysioNetMILoader(root, mode="train")
    val_dataset = PhysioNetMILoader(root, mode="val")
    test_dataset = PhysioNetMILoader(root, mode="test")
    train_subjects = set(train_dataset.split_metadata()["subjects"])
    val_subjects = set(val_dataset.split_metadata()["subjects"])
    test_subjects = set(test_dataset.split_metadata()["subjects"])
    overlaps = {
        "train_val": sorted(train_subjects & val_subjects),
        "train_test": sorted(train_subjects & test_subjects),
        "val_test": sorted(val_subjects & test_subjects),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"PhysioNet-MI subject overlap detected: {overlaps}")
    print(
        f"[PhysioNet-MI] train={len(train_dataset)} val={len(val_dataset)} "
        f"test={len(test_dataset)} shape={list(PhysioNetMILoader.expected_shape)} "
        f"channels={len(PHYSIONET_MI_LABRAM_CH)} scaling=engine_divide_100"
    )
    print(f"[PhysioNet-MI] subject-overlap checks: {overlaps}")
    return train_dataset, test_dataset, val_dataset


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model
            
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=_distributed_device())
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)
    
    def update_image(self, head='images', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.writer.add_image(head + "/" + k, v, self.step if step is None else step)
            
    def flush(self):
        self.writer.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False):
    world_size = get_world_size()

    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op, async_op=async_op)

    return tensor

def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])


def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = _get_world_size_env()  # int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = _get_local_rank_env()
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier(device_ids=[args.gpu])
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

    # Keep the loader's diagnostics machine-readable.  The original LaBraM
    # helper only printed these lists, which made it difficult to distinguish
    # an intentional task-head mismatch from an accidentally partial backbone
    # load in a completed Slurm job.
    return {
        'missing_keys': list(missing_keys),
        'ignored_missing_keys': list(ignore_missing_keys),
        'unexpected_keys': list(unexpected_keys),
        'error_messages': list(error_msgs),
    }

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")
        self.last_grad_finite = True
        self.last_step_skipped = False
        self.last_nonfinite_param_count = 0
        self.last_nonfinite_group_names = []
        self.last_scale_before = self._scaler.get_scale()
        self.last_scale_after = self.last_scale_before

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, layer_names=None):
        self.last_grad_finite = True
        self.last_step_skipped = False
        self.last_nonfinite_param_count = 0
        self.last_nonfinite_group_names = []
        self.last_scale_before = self._scaler.get_scale()
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            nonfinite_group_names = set()
            nonfinite_param_count = 0
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            for group in optimizer.param_groups:
                group_name = group.get("group_name", "unknown")
                for parameter in group["params"]:
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        nonfinite_param_count += 1
                        nonfinite_group_names.add(group_name)
            self.last_nonfinite_param_count = nonfinite_param_count
            self.last_nonfinite_group_names = sorted(nonfinite_group_names)
            self.last_grad_finite = nonfinite_param_count == 0
            self._scaler.step(optimizer)
            self._scaler.update()
            self.last_scale_after = self._scaler.get_scale()
            self.last_step_skipped = self.last_scale_after < self.last_scale_before
        else:
            norm = None
            self.last_scale_after = self._scaler.get_scale()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict): 
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
        
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))
        
        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")
        
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    total_iters = epochs * niter_per_ep
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    warmup_iters = min(warmup_iters, total_iters)
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(total_iters - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == total_iters
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None, save_ckpt_freq=1):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    if not getattr(args, 'enable_deepspeed', False):
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if epoch in {'best', 'best-ba'}:
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name),]
        elif (epoch + 1) % save_ckpt_freq == 0:
            checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if loss_scaler is not None:
                to_save['scaler'] = loss_scaler.state_dict()

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)
                
            if optimizer_disc is not None:
                to_save['optimizer_disc'] = optimizer_disc.state_dict()

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)           

def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None):
    output_dir = Path(args.output_dir)
    
    if not getattr(args, 'enable_deepspeed', False):
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
            if len(all_checkpoints) > 0:
                args.resume = os.path.join(output_dir, 'checkpoint.pth')
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model']) # strict: bool=True, , strict=False
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
                args.start_epoch = 1#checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'optimizer_disc' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])

def create_ds_config(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "latest"), mode="w") as f:
        pass

    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))


def build_pretraining_dataset(datasets: list, time_window: list, stride_size=200, start_percentage=0, end_percentage=1):
    shock_dataset_list = []
    ch_names_list = []
    for dataset_list, window_size in zip(datasets, time_window):
        dataset = ShockDataset([Path(file_path) for file_path in dataset_list], window_size * 200, stride_size, start_percentage, end_percentage)
        shock_dataset_list.append(dataset)
        ch_names_list.append(dataset.get_ch_names())
    return shock_dataset_list, ch_names_list


def get_input_chans(ch_names):
    input_chans = [0] # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans


def _distributed_device():
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y
    

class TUEVLoader(torch.utils.data.Dataset):
    """LaBraM reader for the shared EEG preprocessing-preprocessed TUEV events.

    The serialized event stores raw microvolt data as [16, 1000] with labels
    in [1, 6].  The LaBraM engine performs the single /100 scaling, so this
    loader deliberately returns raw values and reshapes only to the native
    [16, 5, 200] token grid.
    """

    expected_raw_shape = (16, 1000)
    expected_token_shape = (16, 5, 200)
    label_min = 1
    label_max = 6

    def __init__(self, root, files, sampling_rate=200):
        self.root = os.fspath(root)
        self.files = tuple(sorted(os.fspath(name) for name in files))
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.channel_names = list(TUEV_BIPOLAR_16_CH)
        if len(self.channel_names) != self.expected_raw_shape[0]:
            raise RuntimeError("TUEV bipolar channel manifest must contain 16 channels")
        if not self.files:
            raise ValueError(f"TUEV split is empty: {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.files[index])
        with open(path, "rb") as handle:
            sample = pickle.load(handle)
        if not isinstance(sample, dict) or "signal" not in sample or "label" not in sample:
            raise ValueError(f"TUEV record must contain signal and label fields: {path}")
        X = np.asarray(sample["signal"])
        if tuple(X.shape) != self.expected_raw_shape:
            raise ValueError(f"TUEV record {path} shape={X.shape}, expected {self.expected_raw_shape}")
        if not np.isfinite(X).all():
            raise ValueError(f"TUEV record {path} contains non-finite signal values")
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X.reshape(self.expected_token_shape)
        raw_label = int(np.asarray(sample["label"]).reshape(-1)[0])
        if not self.label_min <= raw_label <= self.label_max:
            raise ValueError(f"TUEV record {path} label={raw_label}, expected 1..6")
        Y = raw_label - 1
        X = torch.FloatTensor(X)
        return X, Y

    def get_ch_names(self):
        return list(self.channel_names)

    def split_metadata(self):
        digest = hashlib.sha256()
        record_prefixes = set()
        for name in self.files:
            encoded = os.fsencode(name)
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            record_prefixes.add(name.split("_", 1)[0])
        return {
            "root": os.path.abspath(self.root),
            "file_count": len(self.files),
            "file_manifest_sha256": digest.hexdigest(),
            "record_prefix_count": len(record_prefixes),
            "record_prefixes": sorted(record_prefixes),
            "stored_shape": list(self.expected_raw_shape),
            "token_shape": list(self.expected_token_shape),
            "label_range_stored": [self.label_min, self.label_max],
            "label_range_model": [self.label_min - 1, self.label_max - 1],
            "channel_names": list(self.channel_names),
            "input_scale_divisor": 100.0,
        }
    

def prepare_TUEV_dataset(root):
    root = os.fspath(root)
    processed_root = root
    if not all(os.path.isdir(os.path.join(processed_root, name)) for name in (
        "processed_train", "processed_eval", "processed_test"
    )):
        processed_root = os.path.join(root, "processed")
    train_files = sorted(
        name for name in os.listdir(os.path.join(processed_root, "processed_train"))
        if name.endswith(".pkl")
    )
    val_files = sorted(
        name for name in os.listdir(os.path.join(processed_root, "processed_eval"))
        if name.endswith(".pkl")
    )
    test_files = sorted(
        name for name in os.listdir(os.path.join(processed_root, "processed_test"))
        if name.endswith(".pkl")
    )

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(processed_root, "processed_train"), train_files
    )
    test_dataset = TUEVLoader(
        os.path.join(processed_root, "processed_test"), test_files
    )
    val_dataset = TUEVLoader(
        os.path.join(processed_root, "processed_eval"), val_files
    )
    print(
        f"[TUEV] train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
        f"stored_shape={TUEVLoader.expected_raw_shape} token_shape={TUEVLoader.expected_token_shape} "
        f"channels={TUEV_BIPOLAR_16_CH} scale=/100"
    )
    return train_dataset, test_dataset, val_dataset


class SEEDVLoader(torch.utils.data.Dataset):
    expected_shape = None
    label_min = 0
    label_max = 4

    def __init__(self, root, mode="train", channel_manifest=None, expected_shape=None):
        self.root = root
        self.mode = mode
        self.expected_shape = tuple(expected_shape) if expected_shape is not None else None
        self.channel_names = read_seedv_channel_names(root, manifest_path=channel_manifest)
        if self.channel_names is None:
            raise RuntimeError(
                "SEED-V requires a validated channel manifest matching the stored tensor. "
                "Provide channel_names.json beside the LMDB or pass an explicit manifest path."
            )
        try:
            import lmdb  # local import so non-LMDB datasets do not require it
        except ImportError as exc:
            raise ImportError("SEED-V loading requires the 'lmdb' package in the active environment") from exc
        self._lmdb = lmdb
        self.db = None
        # Read split metadata once in the parent process, then reopen LMDB lazily per
        # worker. Sharing an inherited LMDB handle across DataLoader worker processes
        # can trigger intermittent segfaults on cluster filesystems.
        with lmdb.open(root, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False) as txn:
            raw_keys = txn.get(b"__keys__")
        if raw_keys is None:
            raise KeyError(f"SEED-V LMDB missing '__keys__' in {root}")
        split_index = pickle.loads(raw_keys)
        if mode not in split_index:
            raise KeyError(f"SEED-V LMDB missing split {mode!r}; available: {list(split_index.keys())}")
        self.keys = split_index[mode]
        if not self.keys:
            raise ValueError(f"SEED-V split {mode!r} is empty in {root}")

        first_key = self.keys[0]
        first_key = first_key.encode() if isinstance(first_key, str) else first_key
        with lmdb.open(root, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False) as txn:
            first_raw = txn.get(first_key)
        if first_raw is None:
            raise KeyError(f"SEED-V LMDB key not found: {self.keys[0]!r}")
        first_sample = pickle.loads(first_raw)
        first_shape = self._validate_sample(first_sample, self.keys[0])
        self.channel_count = first_shape[0]
        self.sample_shape = first_shape
        if len(self.channel_names) != self.channel_count:
            raise RuntimeError(
                f"SEED-V manifest has {len(self.channel_names)} channels but stored samples have "
                f"{self.channel_count}; provide a manifest for the actual tensor order."
            )

    def __len__(self):
        return len(self.keys)

    def get_ch_names(self):
        return self.channel_names

    def _validate_sample(self, sample, key):
        if "sample" not in sample or "label" not in sample:
            raise ValueError(f"SEED-V record {key!r} must contain sample and label fields")
        X = np.asarray(sample["sample"])
        expected_shape = self.expected_shape
        if expected_shape is None:
            expected_shape = (X.shape[0], 1, 200)
        if tuple(X.shape) != expected_shape:
            raise ValueError(
                f"SEED-V record {key!r} expected shape {expected_shape}, got {tuple(X.shape)}"
            )
        if not np.isfinite(X).all():
            raise ValueError(f"SEED-V record {key!r} contains NaN or Inf")
        label_values = np.asarray(sample["label"]).reshape(-1)
        if label_values.size != 1:
            raise ValueError(f"SEED-V record {key!r} must contain one scalar label, got {label_values.shape}")
        label = int(label_values[0])
        if not self.label_min <= label <= self.label_max:
            raise ValueError(
                f"SEED-V record {key!r} label must be in [{self.label_min}, {self.label_max}], got {label}"
            )
        return tuple(X.shape)

    def split_metadata(self):
        """Return cheap immutable split metadata without reading every sample."""
        digest = hashlib.sha256()
        for key in self.keys:
            encoded = key.encode("utf-8") if isinstance(key, str) else bytes(key)
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return {
            "mode": self.mode,
            "sample_count": len(self.keys),
            "key_sha256": digest.hexdigest(),
            "sample_shape": list(self.sample_shape),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["db"] = None
        return state

    def _get_db(self):
        if self.db is None:
            self.db = self._lmdb.open(
                self.root,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
            )
        return self.db

    def __getitem__(self, index):
        key = self.keys[index]
        enc_key = key.encode() if isinstance(key, str) else key
        with self._get_db().begin(write=False) as txn:
            raw = txn.get(enc_key)
        if raw is None:
            raise KeyError(f"SEED-V LMDB key not found: {key!r}")
        sample = pickle.loads(raw)
        self._validate_sample(sample, key)
        X = np.asarray(sample["sample"])
        Y = int(np.asarray(sample["label"]).reshape(-1)[0])
        # The finetuning engine owns the native LaBraM input scaling. Keeping
        # the dataset raw prevents applying the divisor twice.
        X = torch.FloatTensor(X)
        return X, Y


def prepare_SEEDV_dataset(root, channel_manifest=None):
    expected_shape = (62, 1, 200)
    train_dataset = SEEDVLoader(root, mode="train", channel_manifest=channel_manifest, expected_shape=expected_shape)
    val_dataset = SEEDVLoader(root, mode="val", channel_manifest=channel_manifest, expected_shape=expected_shape)
    test_dataset = SEEDVLoader(root, mode="test", channel_manifest=channel_manifest, expected_shape=expected_shape)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    return train_dataset, test_dataset, val_dataset


class FACEDLoader(torch.utils.data.Dataset):
    expected_shape = (32, 10, 200)

    def __init__(self, root, mode="train"):
        self.root = root
        self.mode = mode
        try:
            import lmdb  # local import so non-LMDB datasets do not require it
        except ImportError as exc:
            raise ImportError("FACED loading requires the 'lmdb' package in the active environment") from exc
        self._lmdb = lmdb
        self.db = None
        with lmdb.open(root, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False) as txn:
            raw_keys = txn.get(b"__keys__")
        if raw_keys is None:
            raise KeyError(f"FACED LMDB missing '__keys__' in {root}")
        split_index = pickle.loads(raw_keys)
        if mode not in split_index:
            raise KeyError(f"FACED LMDB missing split {mode!r}; available: {list(split_index.keys())}")
        self.keys = split_index[mode]
        self.channel_names = read_faced_channel_names(root)
        if self.channel_names is None or len(self.channel_names) != 32:
            raise RuntimeError(
                "FACED LaBraM runs require a validated 32-channel manifest."
            )

    def __len__(self):
        return len(self.keys)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["db"] = None
        return state

    def _get_db(self):
        if self.db is None:
            self.db = self._lmdb.open(
                self.root,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
            )
        return self.db

    def __getitem__(self, index):
        key = self.keys[index]
        enc_key = key.encode() if isinstance(key, str) else key
        with self._get_db().begin(write=False) as txn:
            raw = txn.get(enc_key)
        if raw is None:
            raise KeyError(f"FACED LMDB key not found: {key!r}")
        sample = pickle.loads(raw)
        X = sample["sample"]
        if tuple(X.shape) != self.expected_shape:
            raise ValueError(
                f"FACED expected {self.expected_shape}, got {tuple(X.shape)} "
                f"for sample key {key!r}"
            )
        if not np.isfinite(X).all():
            raise ValueError(f"FACED sample {key!r} contains NaN or Inf")
        Y = int(sample["label"])
        X = torch.FloatTensor(X)
        return X, Y

    def get_ch_names(self):
        return self.channel_names


def _faced_subject_id(key):
    text = key.decode("utf-8") if isinstance(key, bytes) else str(key)
    return text.split(".pkl", 1)[0].split("-", 1)[0]


def _faced_sample_stats(dataset):
    key = dataset.keys[0]
    enc_key = key.encode() if isinstance(key, str) else key
    with dataset._get_db().begin(write=False) as txn:
        raw = txn.get(enc_key)
    if raw is None:
        raise KeyError(f"FACED LMDB key not found while collecting diagnostics: {key!r}")
    sample = pickle.loads(raw)
    x = np.asarray(sample["sample"])
    if tuple(x.shape) != FACEDLoader.expected_shape:
        raise ValueError(
            f"FACED expected {FACEDLoader.expected_shape}, got {tuple(x.shape)} "
            f"for sample key {key!r}"
        )
    scaled = x / 100.0
    return key, x.shape, (
        float(x.mean()), float(x.std()), float(x.min()), float(x.max())
    ), (
        float(scaled.mean()), float(scaled.std()), float(scaled.min()), float(scaled.max())
    )


def _print_faced_contract(train_dataset, val_dataset, test_dataset):
    key, shape, raw_stats, scaled_stats = _faced_sample_stats(train_dataset)
    split_subjects = {
        "train": sorted({_faced_subject_id(k) for k in train_dataset.keys}),
        "val": sorted({_faced_subject_id(k) for k in val_dataset.keys}),
        "test": sorted({_faced_subject_id(k) for k in test_dataset.keys}),
    }
    train_val_overlap = sorted(set(split_subjects["train"]) & set(split_subjects["val"]))
    train_test_overlap = sorted(set(split_subjects["train"]) & set(split_subjects["test"]))
    val_test_overlap = sorted(set(split_subjects["val"]) & set(split_subjects["test"]))
    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError(
            "FACED split subject overlap detected: "
            f"train-val={train_val_overlap}, train-test={train_test_overlap}, val-test={val_test_overlap}"
        )
    input_chans = get_input_chans(train_dataset.channel_names)
    print(f"[FACED] diagnostic sample key: {key!r}")
    print(f"[FACED] raw sample shape: {tuple(shape)}")
    print(
        "[FACED] raw stats mean/std/min/max: "
        f"{raw_stats[0]:.8g} {raw_stats[1]:.8g} {raw_stats[2]:.8g} {raw_stats[3]:.8g}"
    )
    print(
        "[FACED] post-division-by-100 stats mean/std/min/max: "
        f"{scaled_stats[0]:.8g} {scaled_stats[1]:.8g} {scaled_stats[2]:.8g} {scaled_stats[3]:.8g}"
    )
    print(f"[FACED] channel names ({len(train_dataset.channel_names)}): {train_dataset.channel_names}")
    print(f"[FACED] mapped LaBraM input_chans ({len(input_chans)}): {input_chans}")
    print(
        "[FACED] sample counts: "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )
    print(f"[FACED] train subject IDs ({len(split_subjects['train'])}): {split_subjects['train']}")
    print(f"[FACED] val subject IDs ({len(split_subjects['val'])}): {split_subjects['val']}")
    print(f"[FACED] test subject IDs ({len(split_subjects['test'])}): {split_subjects['test']}")
    print("[FACED] split-overlap checks: train-val=0 train-test=0 val-test=0")


def prepare_FACED_dataset(root):
    train_dataset = FACEDLoader(root, mode="train")
    val_dataset = FACEDLoader(root, mode="val")
    test_dataset = FACEDLoader(root, mode="test")
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    _print_faced_contract(train_dataset, val_dataset, test_dataset)
    return train_dataset, test_dataset, val_dataset


def _normalize_labram_channel_name(ch_name: str) -> str:
    key = str(ch_name).strip().upper()
    alias_map = {
        "A1": "TP9",
        "A2": "TP10",
        "M1": "TP9",
        "M2": "TP10",
        "LEFT_MASTOID": "TP9",
        "RIGHT_MASTOID": "TP10",
    }
    return alias_map.get(key, key)


def _validate_channel_names(channel_names, root, dataset_name: str, expected_count: Optional[int] = None):
    if channel_names is None:
        return None
    if not isinstance(channel_names, (list, tuple)):
        warnings.warn(f"{dataset_name} channel manifest at {root} is not a list/tuple; ignoring it.")
        return None
    normalized = [_normalize_labram_channel_name(ch) for ch in channel_names if str(ch).strip()]
    if expected_count is not None and len(normalized) != expected_count:
        warnings.warn(
            f"{dataset_name} channel manifest for {root} has {len(normalized)} channels instead of {expected_count}; ignoring it."
        )
        return None
    if len(set(normalized)) != len(normalized):
        warnings.warn(
            f"{dataset_name} channel manifest for {root} contains duplicate channel names; ignoring it."
        )
        return None
    missing = [ch for ch in normalized if ch not in standard_1020]
    if missing:
        warnings.warn(
            f"{dataset_name} channel manifest for {root} contains channels not present in LaBraM standard_1020: {missing[:8]}; ignoring it."
        )
        return None
    return normalized


def _validate_seedv_channel_names(channel_names, root, expected_count=None):
    return _validate_channel_names(channel_names, root, "SEED-V", expected_count)


def _extract_channel_names_payload(payload):
    if isinstance(payload, dict):
        if "labram_channel_names" in payload:
            return payload["labram_channel_names"]
        for field in ("channel_names", "ch_names", "channels", "stored_channel_names"):
            if field in payload:
                return payload[field]
    elif isinstance(payload, (list, tuple)):
        return payload
    return None


def read_seedv_channel_names(root, manifest_path=None, expected_count=None) -> Optional[list]:
    candidate_keys = [
        b"__channel_names__",
        b"channel_names",
        b"ch_names",
        b"channels",
    ]
    candidate_files = [
        manifest_path,
        root + "_channel_names.json",
        root + "_channel_names.pkl",
        root + "_metadata.json",
        root + "_meta.json",
        os.path.join(root, "channel_names.json"),
        os.path.join(root, "channel_names.pkl"),
    ]

    try:
        import lmdb
    except ImportError:
        lmdb = None

    if lmdb is not None:
        try:
            db = lmdb.open(root, readonly=True, lock=False, readahead=True, meminit=False)
            with db.begin(write=False) as txn:
                for key in candidate_keys:
                    raw = txn.get(key)
                    if raw is None:
                        continue
                    try:
                        channel_names = pickle.loads(raw)
                    except Exception:
                        try:
                            channel_names = json.loads(raw.decode("utf-8"))
                        except Exception:
                            channel_names = None
                    channel_names = _validate_seedv_channel_names(
                        _extract_channel_names_payload(channel_names), root, expected_count
                    )
                    if channel_names is not None:
                        return channel_names
        except Exception:
            pass

    for path in candidate_files:
        if not path or not os.path.isfile(path):
            continue
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
        except Exception:
            continue

        channel_names = _extract_channel_names_payload(payload)

        channel_names = _validate_seedv_channel_names(channel_names, root, expected_count)
        if channel_names is not None:
            return channel_names

    return None


def _validate_faced_channel_names(channel_names, root):
    return _validate_channel_names(channel_names, root, "FACED", 32)


def read_faced_channel_names(root) -> Optional[list]:
    candidate_keys = [
        b"__channel_manifest__",
        b"__channel_names__",
        b"channel_names",
        b"ch_names",
        b"channels",
    ]
    candidate_files = [
        root + "_channel_manifest.json",
        root + "_channel_names.json",
        os.path.join(root, "channel_manifest.json"),
        os.path.join(root, "channel_names.json"),
    ]

    try:
        import lmdb
    except ImportError:
        lmdb = None

    if lmdb is not None:
        try:
            db = lmdb.open(root, readonly=True, lock=False, readahead=True, meminit=False)
            with db.begin(write=False) as txn:
                for key in candidate_keys:
                    raw = txn.get(key)
                    if raw is None:
                        continue
                    try:
                        payload = pickle.loads(raw)
                    except Exception:
                        try:
                            payload = json.loads(raw.decode("utf-8"))
                        except Exception:
                            payload = None
                    channel_names = _validate_faced_channel_names(_extract_channel_names_payload(payload), root)
                    if channel_names is not None:
                        return channel_names
        except Exception:
            pass

    for path in candidate_files:
        if not os.path.isfile(path):
            continue
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
        except Exception:
            continue
        channel_names = _validate_faced_channel_names(_extract_channel_names_payload(payload), root)
        if channel_names is not None:
            return channel_names

    return None


def prepare_TUAB_dataset(root):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if is_binary:
        y_true = np.asarray(target).reshape(-1)
        y_score = np.asarray(output).reshape(-1)
        y_pred = (y_score >= threshold).astype(int)
        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = accuracy_score(y_true, y_pred)
            elif metric == "balanced_accuracy":
                results[metric] = balanced_accuracy_score(y_true, y_pred)
            elif metric == "pr_auc":
                # average_precision_score is the standard PR-AUC summary statistic.
                results[metric] = average_precision_score(y_true, y_score)
            elif metric == "roc_auc":
                if len(np.unique(y_true)) < 2:
                    results[metric] = 0.0
                else:
                    results[metric] = roc_auc_score(y_true, y_score)
    else:
        y_true = np.asarray(target).reshape(-1)
        y_score = np.asarray(output)
        y_pred = np.argmax(y_score, axis=-1)
        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = accuracy_score(y_true, y_pred)
            elif metric == "balanced_accuracy":
                results[metric] = balanced_accuracy_score(y_true, y_pred)
            elif metric == "cohen_kappa":
                results[metric] = cohen_kappa_score(y_true, y_pred)
            elif metric == "f1_weighted":
                results[metric] = f1_score(y_true, y_pred, average="weighted")
    return results


def classification_diagnostics_from_confusion(confusion, logit_sum=None, logit_sq_sum=None, logit_count=None):
    """Return class-count, recall/F1, and prediction-concentration diagnostics.

    ``confusion`` uses rows for targets and columns for predictions.  The
    helper accepts either a NumPy array or a tensor and deliberately returns
    JSON-friendly Python values for run logs.
    """
    if torch.is_tensor(confusion):
        confusion = confusion.detach().cpu().numpy()
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(f"Expected square confusion matrix, got {confusion.shape}")

    num_classes = confusion.shape[0]
    target_counts = confusion.sum(axis=1)
    prediction_counts = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    recall = np.divide(
        true_positive,
        target_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=target_counts > 0,
    )
    precision = np.divide(
        true_positive,
        prediction_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=prediction_counts > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    total = int(target_counts.sum())
    prediction_fraction = (
        prediction_counts.astype(np.float64) / total if total else np.zeros(num_classes, dtype=np.float64)
    )
    diagnostics = {
        **{f"target_count_class_{i}": int(value) for i, value in enumerate(target_counts)},
        **{f"prediction_count_class_{i}": int(value) for i, value in enumerate(prediction_counts)},
        **{f"prediction_fraction_class_{i}": float(value) for i, value in enumerate(prediction_fraction)},
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
        "majority_prediction_fraction": float(prediction_fraction.max()) if total else 0.0,
        "number_of_predicted_classes": int(np.count_nonzero(prediction_counts)),
    }
    if logit_sum is not None and logit_sq_sum is not None and logit_count:
        mean = float(logit_sum / logit_count)
        variance = max(float(logit_sq_sum / logit_count) - mean * mean, 0.0)
        diagnostics["logit_mean"] = mean
        diagnostics["logit_std"] = variance ** 0.5
    return diagnostics
