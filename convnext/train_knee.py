#!/usr/bin/env python3
"""cur_best_eval94_dense96_convnext: ConvNeXt-Small DDP+EMA on Dense-96.

8 GPUs: torchrun --standalone --nproc_per_node=8 train_knee.py
1 GPU:  python train_knee.py
Default: ConvNeXt-Small (ImageNet-22k -> ImageNet-1k), 384px, k=12/k_eval=94,
20 epochs, batch size 2 PER GPU, gradient checkpointing, EMA decay 0.999,
validation batch size 1 and 24-window backbone chunks.  Up to 94 unique
validation windows are concatenated before one MIL head call, matching
submit.py while keeping the larger bag out of one monolithic backbone batch.

Training normally obtains the timm pretrained weights online/from its cache.
For offline training, --pretrained_path points to the official timm model's
pytorch_model.bin or model.safetensors for the exact CONVNEXT_ARCH below.
CoAtNet, ConvNeXt-Tiny/Base, DINOv2 and full RSNA checkpoints are not compatible
with --pretrained_path. Full RSNA weights are used only by the submission.
The backbone and twelve-label head are trained together from the first epoch.
The saved .pt contains all fine-tuned weights and inference settings.
"""
import os, sys, time, json, math, random, argparse, copy
from datetime import timedelta
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from sklearn.metrics import roc_auc_score

# --- ROI localizer (anatomical joint crop). Optional so the no-ROI path is untouched. ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from roi_localize import square_box as _roi_square_box, compartments as _roi_compartments
except Exception:
    _roi_square_box = _roi_compartments = None

HERE = os.path.dirname(os.path.abspath(__file__))
RSNA = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# Input discovery. On Kaggle the corpus arrives as two read-only datasets and the
# competition data as a third, so nothing lives beside this script. Everything in
# this block is discovery only - the training code further down is unchanged.
# ---------------------------------------------------------------------------
def _find(*names, root="/kaggle/input"):
    """First path under root whose basename matches one of names."""
    for d, _, fs in os.walk(root):
        for n in names:
            if n in fs:
                return os.path.join(d, n)
    return None


class _TwoPartVols:
    """Presents the two corpus parts as one logical memory-mapped array.

    Both parts stay memory-mapped and are never concatenated on disk: copying 22 GB would be
    pointless when every read is a single study. Row order is part 1 then part 2, matching the
    order the id files concatenate in. That ordering is the contract between volumes, masks and
    ids, so do not sort any of them independently.
    """
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


def _corpus_files(corpus_dir=None, corpus_ext_dir=None):
    """Resolve matching volume/mask/id files once, also usable in spawned workers."""
    if corpus_dir or corpus_ext_dir:
        if not (corpus_dir and corpus_ext_dir):
            raise ValueError("Provide both --corpus_dir and --corpus_ext_dir.")
        parts = [(corpus_dir, "all"), (corpus_ext_dir, "extra")]
    elif os.path.isfile(os.path.join(HERE, "all_vols.npy")):
        parts = [(HERE, "all")]
        if os.path.isfile(os.path.join(HERE, "extra_vols.npy")):
            parts.append((HERE, "extra"))
    else:
        av, bv = _find("all_vols.npy"), _find("extra_vols.npy")
        if not (av and bv):
            raise FileNotFoundError("Attach both corpus parts or provide --corpus_dir and --corpus_ext_dir.")
        parts = [(os.path.dirname(av), "all"), (os.path.dirname(bv), "extra")]
    files = [tuple(os.path.join(folder, f"{prefix}_{kind}.npy")
                   for kind in ("vols", "masks", "ids")) for folder, prefix in parts]
    for part in files:
        for path in part:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
    return files


def _open_corpus(corpus_files=None):
    """Return (vols, masks), from a local single-file corpus or the two public parts."""
    parts = corpus_files if corpus_files is not None else _corpus_files()
    volumes, masks = [], []
    for vp, mp, ip in parts:
        v, m = np.load(vp, mmap_mode="r"), np.load(mp)
        ids = np.load(ip, allow_pickle=False)
        if v.ndim != 4 or v.dtype != np.uint8 or v.shape[1] < 3:
            raise ValueError(f"Expected uint8 [N,D,H,W], D>=3: {vp}: {v.shape}, {v.dtype}")
        if m.shape != v.shape[:2] or ids.shape != (v.shape[0],):
            raise ValueError(f"Volume/mask/id shapes do not align in {vp}.")
        volumes.append(v); masks.append(m)
    if len(volumes) == 2 and volumes[0].shape[1:] != volumes[1].shape[1:]:
        raise ValueError("Corpus parts have different slice/image dimensions.")
    vols = volumes[0] if len(volumes) == 1 else _TwoPartVols(*volumes)
    masks = masks[0] if len(masks) == 1 else np.concatenate(masks, axis=0)
    return vols, masks


def _open_ids(corpus_files=None):
    parts = corpus_files if corpus_files is not None else _corpus_files()
    ids = np.concatenate([np.load(p[2], allow_pickle=False).astype(str) for p in parts])
    if len(set(ids)) != len(ids) or any(not u.strip() for u in ids):
        raise ValueError("Corpus IDs must be unique and nonempty across both parts.")
    return ids
LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
       "Effusion","Synovitis","Baker's","Contusion","Fracture"]
DENSE_CONTRACT = "maxspan_dense96_capacity_aware_unique_first_series"
DENSE_SHAPE = (96, 336, 336)
DENSE_SPAN = (0.02, 0.98)
DENSE_SLOTS = (
    ("Sagittal", 1, 27), ("Sagittal", 0, 21),
    ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18),
)


_NO_LABELS = """
No training labels found, so there is nothing to train against.

The competition labels only 58 of the 4,407 studies. The other 4,349 carry a free-text
radiology report instead, so before this can train you have to turn those reports into
twelve probabilities per study and pass the result with --labels.

Expected format: a parquet with a StudyInstanceUID column plus the columns
  {cols}
with values between 0 and 1. Soft values work better than hard 0/1 here: the loss is built
for them, and hedged reports are common.

Everything else in this notebook is ready to run once that file exists.
"""


# ------------------------------- data ----------------------------------------
class StudyWindows(Dataset):
    """Per-study bag of 2.5D windows sampled from the Dense-96 cache.
    Each window = 3 adjacent cached indices -> channels, resized to `res`.
    Dense valid planes are packed contiguously and eval never repeats a center.
    As in the baseline, a window at a packed slot boundary can span two series.
    Normalization is configurable."""
    def __init__(self, root, ids, id2row, labels, res, k, train, aug=True, norm="none",
                 roi=False, roi_mode="tight", roi_pad=0.06, roi_overlap=0.12,
                 roi_sbox=None, roi_cen=None, corpus_files=None):
        self.root = root
        self.ids = ids
        self.id2row = id2row
        self.labels = labels              # dict uid -> np.float32[12]
        self.res, self.k, self.train, self.aug = res, k, train, aug
        self.norm = norm                  # auto resolves to ImageNet normalization for ConvNeXt
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.vols = None; self.masks = None
        self.corpus_files = corpus_files
        # --- ROI anatomical joint-crop config ---
        self.roi = bool(roi)
        self.roi_mode, self.roi_pad, self.roi_overlap = roi_mode, roi_pad, roi_overlap
        self.roi_sbox = roi_sbox          # (N,D,4) int16 per-slice tissue bbox, or None
        self.roi_cen = roi_cen            # (N,D,2) int16 per-slice joint centroid, or None
        if self.roi and (_roi_square_box is None or roi_sbox is None):
            raise RuntimeError("roi=True but roi_localize or roi_boxes not available")

    def __len__(self): return len(self.ids)

    def _ensure(self):
        if self.vols is None:
            self.vols, self.masks = _open_corpus(self.corpus_files)

    def _centers(self, valid, count):
        # Dense builder packs all real planes contiguously before masked padding.
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs: cs = [max(1, min((lo + hi) // 2, self._D - 2))]
        if self.train:
            reps = count // len(cs) + 1
            pool = (cs * reps)
            random.shuffle(pool)
            return pool[:count]
        # Eval uses every available center up to k_eval, never repeated. Studies
        # with fewer than 96 real planes therefore produce fewer than 94 windows.
        count = min(count, len(cs))
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        if len(np.unique(idx)) != count:
            raise AssertionError("Dense eval center selection produced duplicates.")
        return [cs[i] for i in idx]

    def _resize(self, tri):
        """tri: (3,h,w) float32 [0,1] -> (3,res,res) float32."""
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear",
                              align_corners=False)[0]
        return t.numpy()

    def __getitem__(self, i):
        self._ensure()
        uid = self.ids[i]; row = self.id2row[uid]
        self._D = self.vols.shape[1]
        m = self.masks[row]
        valid = np.where(m > 0)[0]
        if len(valid) < 3: valid = np.arange(min(3, self._D))
        # compartment mode emits 2 crops/center -> sample ceil(k/2) centers to keep #windows==k
        compart = self.roi and self.roi_mode == "compartment"
        n_centers = (self.k + 1) // 2 if compart else self.k
        cs = self._centers(valid, n_centers)
        vol = self.vols[row]  # (D,H,W) u8  (single study read)
        tiles = []            # list of (3,res,res) float32
        for c in cs:
            c = max(1, min(c, self._D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0  # (3,H,W)
            H, W = tri.shape[-2], tri.shape[-1]
            if not self.roi:
                tiles.append(self._resize(tri))
                continue
            # one joint box from the CENTER slice, applied to all 3 slices (keeps RGB registered)
            sq_mode = "tight" if compart else self.roi_mode
            sq = _roi_square_box(tuple(int(v) for v in self.roi_sbox[row, c]), W=W, H=H,
                                 pad=self.roi_pad, mode=sq_mode,
                                 centroid=tuple(float(v) for v in self.roi_cen[row, c]))
            if compart:
                left, right = _roi_compartments(sq, overlap=self.roi_overlap)
                for box in (left, right):
                    x0, y0, x1, y1 = box
                    tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
            else:
                x0, y0, x1, y1 = sq
                tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
        if len(tiles) > self.k:
            tiles = tiles[:self.k]
        wins = np.stack(tiles, 0)          # (K,3,res,res)
        x = torch.from_numpy(wins)
        if self.train and self.aug:
            # light medical-safe aug: NO flips (laterality is signal); mild intensity jitter
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        if self.norm == "imagenet":        # each backbone at its correct input distribution
            x = (x - self._mean) / self._std
        elif self.norm == "coatnet":
            x = (x - 0.5) / 0.5
        y = torch.from_numpy(self.labels[uid])
        return x, y


def collate(batch):
    xs = torch.stack([b[0] for b in batch])   # (B,K,3,res,res)
    ys = torch.stack([b[1] for b in batch])   # (B,12)
    return xs, ys


# ------------------------------- model ---------------------------------------
def _amp_dtype(device, precision):
    if device != "cuda":
        if precision in ("fp16", "bf16"):
            raise ValueError("fp16/bf16 training requires CUDA; use --precision fp32 on CPU.")
        return None
    native_bf16 = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    if precision == "auto":
        return torch.bfloat16 if native_bf16 else torch.float16
    if precision == "bf16" and not native_bf16:
        raise ValueError("GPU has no native bfloat16 support; use --precision auto or fp16.")
    return {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


CONVNEXT_ARCH = "convnext_small.fb_in22k_ft_in1k_384"


def build_backbone(arch=CONVNEXT_ARCH, pretrained=False, res=384, pretrained_path=None):
    import timm
    if arch != CONVNEXT_ARCH:
        raise ValueError(f"This ConvNeXt version uses {CONVNEXT_ARCH}; got {arch}.")
    if res < 32 or res % 32:
        raise ValueError("ConvNeXt resolution must be a positive multiple of 32, e.g. 224 or 384.")
    # ConvNeXt is fully convolutional: resize data to res, do not pass img_size.
    kw = dict(pretrained=pretrained, in_chans=3, global_pool="avg",
              num_classes=1000 if pretrained else 0)
    if pretrained_path:
        if not pretrained:
            raise ValueError("A local pretrained file requires pretrained=True.")
        path = os.path.abspath(pretrained_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing ConvNeXt pretrained file: {path}")
        # Load the official 1000-class timm checkpoint strictly before removing its classifier.
        kw["pretrained_cfg_overlay"] = {"file": path, "hf_hub_id": None, "url": ""}
    bb = timm.create_model(arch, **kw)
    if pretrained:
        bb.reset_classifier(0, global_pool="avg")
    return bb


class RaptorClassifier(nn.Module):
    """ConvNeXt encoder + unchanged per-diagnosis attention-MIL head (12 findings)."""
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


class ModelEMA:
    """Exponential moving average of the complete model state on the active GPU.

    Each DDP rank owns an identical EMA copy. Floating parameters and buffers are
    averaged; integer buffers such as BatchNorm counters are copied directly.
    """
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for name, averaged in self.module.state_dict().items():
            current = source[name].detach()
            if averaged.is_floating_point():
                averaged.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                averaged.copy_(current)
        self.updates += 1


# ------------------------------- distributed helpers ------------------------
class DistributedContext:
    """torchrun owns process creation; this class binds one process to one device."""
    def __init__(self, timeout_minutes=120):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        self.is_main = self.rank == 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            if self.local_rank >= torch.cuda.device_count():
                raise ValueError(f"LOCAL_RANK={self.local_rank}, but only "
                                 f"{torch.cuda.device_count()} CUDA devices are visible.")
            torch.cuda.set_device(self.local_rank)
        elif self.distributed:
            raise RuntimeError("Multi-GPU training requested, but CUDA is unavailable in this process.")
        if self.distributed:
            dist.init_process_group(backend="nccl", init_method="env://",
                                    timeout=timedelta(minutes=timeout_minutes))
        self.log(f"[DDP] world_size={self.world_size}, backend="
                 f"{'nccl' if self.distributed else 'single-process'}")

    def log(self, *args, **kwargs):
        if self.is_main:
            kwargs.setdefault("flush", True)
            print(*args, **kwargs)

    def gather(self, value):
        if not self.distributed:
            return [value]
        gathered = [None] * self.world_size
        # Only small CPU predictions/metadata, never images or model state.
        dist.all_gather_object(gathered, value)
        return gathered

    def close(self):
        # No barrier here: a failing worker must be allowed to exit so torchrun
        # can stop its peers, instead of waiting forever in an error path.
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _loader_kwargs(workers, seed, device, prefetch_factor=1, persistent=False):
    kw = dict(num_workers=workers, pin_memory=device == "cuda",
              worker_init_fn=_seed_worker,
              generator=torch.Generator().manual_seed(seed))
    if workers:
        # Forking after NCCL initialization is unsafe. Spawn also leaves the
        # corpus lazy-opened independently inside each DataLoader worker.
        kw.update(multiprocessing_context="spawn", prefetch_factor=prefetch_factor,
                  persistent_workers=persistent)
    return kw


def _eval_indices(size, rank, world_size):
    """No padding: every validation/OOF study belongs to exactly one rank."""
    if size < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid evaluation shard size/rank/world_size.")
    return list(range(rank, size, world_size))


def _merge_predictions(parts, size, n_labels=12):
    """Restore original study order and reject duplicates, omissions or NaNs."""
    indices = np.concatenate([np.asarray(p[0], dtype=np.int64) for p in parts])
    predictions = np.concatenate([np.asarray(p[1], dtype=np.float32) for p in parts], axis=0)
    if predictions.shape != (len(indices), n_labels):
        raise ValueError(f"Invalid gathered prediction shape: {predictions.shape}.")
    order = np.argsort(indices)
    if not np.array_equal(indices[order], np.arange(size)):
        raise ValueError("Distributed evaluation has missing, duplicate or out-of-range study indices.")
    predictions = predictions[order]
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Non-finite validation probabilities; try --precision bf16 or fp32.")
    return predictions


@torch.no_grad()
def _sync_buffers(model, ctx):
    if ctx.distributed:
        # Keep any registered buffers identical to the rank-0 checkpoint.
        # ConvNeXt uses LayerNorm, so there are no BatchNorm running statistics.
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)


@torch.no_grad()
def _load_shared_state(model, state, ctx):
    if ctx.is_main:
        model.load_state_dict(state, strict=True)
    if ctx.distributed:
        # Copy into existing tensors, preserving DDP's registered parameters.
        for value in model.state_dict().values():
            dist.broadcast(value, src=0)


def _make_eval_loader(dataset, a, ctx):
    indices = _eval_indices(len(dataset), ctx.rank, ctx.world_size)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=a.eval_bs, shuffle=False,
        collate_fn=collate,
        **_loader_kwargs(a.eval_workers, a.seed + ctx.rank + 10000, ctx.device,
                         a.prefetch_factor, persistent=a.eval_workers > 0))
    return loader, indices


def _eval_forward_chunked(model, x, window_batch):
    """Encode windows in chunks, then apply MIL once to every study's full bag.

    With eval_bs=1 and k_eval=94 this is the same computation layout used by
    submit.py: up to 24/24/24/22 backbone images followed by one MIL head.
    Splitting predictions and averaging probabilities would not be equivalent.
    """
    if x.ndim != 5 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"Expected validation input [B,K,3,H,W], got {tuple(x.shape)}")
    batch, windows = x.shape[:2]
    flat = x.flatten(0, 1)
    chunk = len(flat) if window_batch == 0 else min(int(window_batch), len(flat))
    features = torch.cat(
        [model.backbone(part) for part in flat.split(chunk)], dim=0
    )
    return model.head(features.reshape(batch, windows, -1))


@torch.no_grad()
def _predict_distributed(model, loader, indices, size, ctx, amp_dtype,
                         eval_window_batch, name="val"):
    # IMPORTANT: model is the unwrapped RaptorClassifier, not DDP.
    # Ranks have unequal evaluation lengths (some may be empty). DDP forward
    # buffer broadcasts would otherwise mismatch collectives and hang.
    model.eval()
    _sync_buffers(model, ctx)
    local = []
    for step, (x, _) in enumerate(loader):
        with torch.autocast(ctx.device, dtype=amp_dtype or torch.float32,
                            enabled=amp_dtype is not None):
            logits = _eval_forward_chunked(
                model, x.to(ctx.device, non_blocking=True), eval_window_batch
            )
            probabilities = torch.sigmoid(logits.float())
        local.append(probabilities.cpu().numpy())
        if step == 0:
            ctx.log(f"[{name} shapes] rank0 input={tuple(x.shape)}, "
                    f"backbone_window_batch={eval_window_batch or 'all'}, "
                    f"probabilities={tuple(probabilities.shape)}")
    pred = np.concatenate(local) if local else np.empty((0, len(LAB)), dtype=np.float32)
    return _merge_predictions(ctx.gather((indices, pred)), size, len(LAB))


def _auc_metrics(truth, predictions):
    aucs = {}
    for j, name in enumerate(LAB):
        if len(np.unique(truth[:, j].astype(int))) > 1:
            aucs[name] = float(roc_auc_score(truth[:, j], predictions[:, j]))
    if not aucs:
        raise ValueError("Gold validation has no label with both positive and negative studies.")
    return float(np.mean(list(aucs.values()))), aucs


def _atomic_json(path, payload):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def _atomic_checkpoint(path, state):
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def _validate_dense_manifests(corpus_files):
    """Reject another 96-slice cache whose sampling contract only looks similar."""
    checked = set()
    for volume_path, _, _ in corpus_files:
        folder = os.path.dirname(volume_path)
        if folder in checked:
            continue
        checked.add(folder)
        path = os.path.join(folder, "corpus_manifest.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Dense-96 manifest is required: {path}. "
                "Build this cache with build_corpus_dense96.py."
            )
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        observed_slots = tuple(tuple(slot) for slot in manifest.get("slots", []))
        if (manifest.get("corpus_kind") != DENSE_CONTRACT
                or int(manifest.get("depth", -1)) != DENSE_SHAPE[0]
                or int(manifest.get("img", -1)) != DENSE_SHAPE[1]
                or tuple(manifest.get("span", [])) != DENSE_SPAN
                or observed_slots != DENSE_SLOTS
                or manifest.get("unique_source_slices") is not True):
            raise ValueError(f"Incompatible Dense-96 manifest: {path}")


# ------------------------------- train ---------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", default="/root/RSNA/train_data/knee-raptor-corpus-96-dense", help="Dense-96 part 1 directory containing all_vols/masks/ids.npy")
    ap.add_argument("--corpus_ext_dir", default="/root/RSNA/train_data/knee-raptor-corpus-96-dense-ext", help="Dense-96 part 2 directory containing extra_vols/masks/ids.npy")
    ap.add_argument("--competition_dir", default="/root/RSNA/rsna-knee-abnormality-detection", help="Official competition directory containing train.csv")
    ap.add_argument("--output_dir", default=os.path.join(HERE, "output7"), help="Directory for checkpoints and run metadata")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--ckpt", default="timm", choices=["timm", "pretrained", "none"],
                    help="timm/pretrained: ImageNet weights; none: explicit random-initialization control")
    ap.add_argument("--pretrained_path", default=None,
                    help="Optional official timm ConvNeXt-Small 384 .bin/.safetensors for offline initialization")
    ap.add_argument("--arch", default=CONVNEXT_ARCH, choices=[CONVNEXT_ARCH])
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k_eval", type=int, default=94,
                    help="Maximum unique validation windows per study; 94 matches Dense-96 submit.py")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=2, help="Studies PER GPU; global batch = bs * world_size")
    ap.add_argument("--bb_lr", type=float, default=5e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4, help="Training DataLoader workers PER GPU")
    ap.add_argument("--eval_bs", type=int, default=1,
                    help="Validation studies per GPU; keep 1 because Dense-96 bags can have variable K")
    ap.add_argument("--eval_window_batch", type=int, default=24,
                    help="Backbone windows per validation chunk; 0 encodes B*K at once")
    ap.add_argument("--eval_workers", type=int, default=0, help="Validation/OOF DataLoader workers per GPU")
    ap.add_argument("--prefetch_factor", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20, help="Training progress interval in optimizer steps")
    ap.add_argument("--ddp_timeout_minutes", type=int, default=120)
    ap.add_argument("--local-rank", "--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--freeze_blocks", type=int, default=0)
    # Each kept epoch is a full ConvNeXt checkpoint in system RAM.
    # Lower to 1 to disable weight averaging and reduce host memory usage.
    ap.add_argument("--topk", type=int, default=3,
                    help="top epochs kept for weight averaging (1 disables averaging)")
    ap.add_argument("--norm", default="auto", choices=["auto", "none", "imagenet", "coatnet"],
                    help="auto/imagenet: pretrained ConvNeXt ImageNet mean/std; coatnet/none are explicit ablations")
    ap.add_argument("--grad_ckpt", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable gradient checkpointing (default); --no-grad_ckpt disables it")
    ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True,
                    help="Validate and save exponential-moving-average weights (default); --no-ema disables it")
    ap.add_argument("--ema_decay", type=float, default=0.999,
                    help="EMA decay applied after each successful optimizer step")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--labels", default="/root/RSNA/rsna-knee-llm-labels/teacher0949_other_model_weighted_blend_50_50.parquet",
                    help="parquet of training labels: StudyInstanceUID + the twelve finding "
                         "columns, values 0..1.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # ---- CV fold hook ----
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--fold_file", default=None)
    # ---- anatomical ROI joint-crop (A/B lever). Default OFF -> identical to baseline. ----
    ap.add_argument("--roi", action="store_true")
    ap.add_argument("--roi_mode", default="compartment", choices=["tight", "safe", "compartment"])
    ap.add_argument("--roi_pad", type=float, default=0.06)
    ap.add_argument("--roi_overlap", type=float, default=0.12)
    ap.add_argument("--roi_boxes", default=os.path.join(HERE, "roi_boxes.npz"))
    a = ap.parse_args()
    if a.pretrained_path and a.ckpt == "none":
        ap.error("--pretrained_path cannot be combined with --ckpt none.")
    if a.res < 32 or a.res % 32:
        ap.error("--res must be a positive multiple of 32, e.g. 224 or 384.")
    if a.roi:
        ap.error("This ConvNeXt submit.py supports roi=False; leave --roi disabled.")
    if a.norm == "auto":
        a.norm = "imagenet"
    if min(a.res, a.k, a.k_eval, a.bs, a.epochs) < 1 or a.workers < 0:
        ap.error("res/k/k_eval/bs/epochs must be positive and workers must be nonnegative.")
    if (a.eval_bs < 0 or a.eval_window_batch < 0 or a.eval_workers < 0
            or a.prefetch_factor < 1 or a.log_every < 1):
        ap.error("eval_bs/eval_window_batch/eval_workers must be nonnegative; "
                 "prefetch_factor/log_every must be positive.")
    if a.ddp_timeout_minutes < 1 or a.limit < 0 or a.topk < 1 or a.freeze_blocks < 0:
        ap.error("Invalid timeout/limit/topk/freeze_blocks.")
    if not 0.0 <= a.ema_decay < 1.0:
        ap.error("--ema_decay must be in [0,1).")
    if a.folds and (not a.fold_file or not 0 <= a.fold < a.folds):
        ap.error("--folds requires --fold_file and a valid --fold.")
    if a.smoke:
        a.epochs, a.bs, a.k, a.k_eval, a.limit, a.workers, a.eval_workers = 2, min(a.bs, 2), 4, 6, 40, 0, 0
    a.eval_bs = a.eval_bs or 1
    return a


def train(a, ctx):
    a.output_dir = os.path.abspath(a.output_dir)
    if ctx.is_main:
        os.makedirs(a.output_dir, exist_ok=True)
    # The sampler uses a shared seed; augmentation/dropout streams differ by rank.
    random.seed(a.seed + ctx.rank)
    np.random.seed(a.seed + ctx.rank)
    torch.manual_seed(a.seed + ctx.rank)
    dev = ctx.device
    amp_dtype = _amp_dtype(dev, a.precision)
    devices = ctx.gather(dict(rank=ctx.rank, local_rank=ctx.local_rank,
                              gpu=torch.cuda.get_device_name() if dev == "cuda" else "CPU",
                              precision=str(amp_dtype or torch.float32)))
    if len({d["precision"] for d in devices}) != 1:
        raise ValueError("Ranks selected different AMP dtypes; set a precision supported by every GPU.")
    for info in devices:
        ctx.log(f"[device] {info}")
    amp_enabled = amp_dtype is not None
    amp_type = amp_dtype or torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    global_bs = a.bs * ctx.world_size
    ctx.log(f"[config] res={a.res}, k={a.k}/{a.k_eval}, per_gpu_bs={a.bs}, "
            f"global_bs={global_bs}, train_workers={a.workers * ctx.world_size}, "
            f"eval_bs_per_gpu={a.eval_bs}, eval_window_batch={a.eval_window_batch}, "
            f"norm={a.norm}, tag={a.tag}")
    if a.k_eval == 94 and (a.eval_bs != 1 or a.eval_window_batch != 24):
        raise ValueError("Submission-aligned Dense-96 validation requires --eval_bs 1 "
                         "and --eval_window_batch 24.")
    ctx.log(f"[lr] backbone={a.bb_lr}, head={a.head_lr}; no automatic LR scaling")

    # ---- ids / labels ----
    corpus_files = _corpus_files(a.corpus_dir, a.corpus_ext_dir)
    _validate_dense_manifests(corpus_files)
    ids = _open_ids(corpus_files)
    corpus_vols, corpus_masks = _open_corpus(corpus_files)
    corpus_shape = tuple(corpus_vols.shape)
    if corpus_shape[1:] != DENSE_SHAPE:
        raise ValueError(
            "cur_best_eval94_dense96_convnext requires a (N,96,336,336) cache; "
            f"got {corpus_shape}. Build it with build_corpus_dense96.py."
        )
    ctx.log(f"[data] corpus={corpus_shape}, ids={len(ids)}, parts={len(corpus_files)}", flush=True)
    del corpus_vols, corpus_masks
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)
    _tcsv = os.path.join(a.competition_dir, "train.csv") if a.competition_dir else os.path.join(RSNA, "train.csv")
    if not a.competition_dir and not os.path.isfile(_tcsv):
        _tcsv = _find("train.csv")
    if not _tcsv or not os.path.isfile(_tcsv):
        raise FileNotFoundError(f"Could not find train.csv at {_tcsv}; check --competition_dir.")
    tr = pd.read_csv(_tcsv); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    if tr["StudyInstanceUID"].duplicated().any():
        raise ValueError("Duplicate StudyInstanceUID in official train.csv.")
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
    gold_ids = [u for u in gold_df.index if u in idset]
    if not gold_ids:
        raise ValueError("No fully labelled official studies match the corpus; validation would be empty.")
    _lab = a.labels or _find("labels_llm_soft.parquet")
    if not _lab or not os.path.exists(_lab):
        # Exit cleanly rather than as a failure: running this notebook as published, with no
        # labels attached, is the expected path and should read as an explanation, not a crash.
        ctx.log(_NO_LABELS.format(cols=", ".join(LAB)), flush=True)
        raise SystemExit(1)
    soft = pd.read_parquet(_lab)
    missing = [c for c in ["StudyInstanceUID"] + LAB if c not in soft.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")
    if soft["StudyInstanceUID"].isna().any() or soft["StudyInstanceUID"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing or blank StudyInstanceUID in label parquet.")
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str); soft = soft.set_index("StudyInstanceUID")
    if not soft.index.is_unique:
        raise ValueError("Duplicate StudyInstanceUID in label parquet.")
    label_values = soft[LAB].to_numpy(dtype=np.float32)
    if not np.isfinite(label_values).all() or np.any((label_values < 0) | (label_values > 1)):
        raise ValueError("All twelve training labels must be finite numbers in [0,1].")
    goldset = set(gold_ids)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    # ---- CV fold hook: hold out fold `a.fold`, train on the rest ----
    oof_ids = []
    if a.folds > 0:
        assert 0 <= a.fold < a.folds, f"--fold must be in [0,{a.folds}) when --folds>0"
        fmap = json.load(open(a.fold_file))["folds"]
        held = set(u for u in train_ids if fmap.get(u, -1) == a.fold)
        oof_ids = [u for u in train_ids if u in held]
        train_ids = [u for u in train_ids if u not in held]
        ctx.log(f"[cv] fold {a.fold}/{a.folds}: train {len(train_ids)} | OOF held-out {len(oof_ids)} "
              f"| fold_file {os.path.basename(a.fold_file)}", flush=True)
    if a.limit: train_ids = train_ids[:a.limit]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in train_ids}
    for u in oof_ids: labels[u] = soft.loc[u, LAB].values.astype(np.float32)
    for u in gold_ids: labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)
    ctx.log(f"train {len(train_ids)} | gold-val {len(gold_ids)}"
          + (f" | oof {len(oof_ids)}" if oof_ids else ""), flush=True)

    if not train_ids:
        raise ValueError("No training studies remain after ID matching and validation/fold exclusion.")
    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)

    # ---- ROI boxes (only when --roi) ----
    roi_sbox = roi_cen = None
    if a.roi:
        rb = np.load(a.roi_boxes)
        rb_ids = rb["ids"].astype(str)
        if not np.array_equal(rb_ids, ids):
            rmap = {u: i for i, u in enumerate(rb_ids)}
            missing = [u for u in ids if u not in rmap]
            if missing:
                raise RuntimeError(f"roi_boxes missing {len(missing)} corpus ids (e.g. {missing[:2]})")
            order = np.array([rmap[u] for u in ids])
            roi_sbox = rb["sbox"][order]; roi_cen = rb["cen"][order]
        else:
            roi_sbox = rb["sbox"]; roi_cen = rb["cen"]
        ctx.log(f"[roi] ENABLED mode={a.roi_mode} pad={a.roi_pad} overlap={a.roi_overlap} "
              f"boxes={os.path.basename(a.roi_boxes)} sbox={roi_sbox.shape}", flush=True)
    _roi_kw = dict(corpus_files=corpus_files, roi=a.roi, roi_mode=a.roi_mode, roi_pad=a.roi_pad, roi_overlap=a.roi_overlap,
                   roi_sbox=roi_sbox, roi_cen=roi_cen)

    tds = StudyWindows(HERE, train_ids, id2row, labels, a.res, a.k, train=True, norm=a.norm, **_roi_kw)
    vds = StudyWindows(HERE, gold_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)
    sampler = DistributedSampler(tds, num_replicas=ctx.world_size, rank=ctx.rank,
                                 shuffle=True, seed=a.seed, drop_last=True) if ctx.distributed else None
    tl = DataLoader(
        tds, batch_size=a.bs, shuffle=sampler is None, sampler=sampler, drop_last=True,
        collate_fn=collate,
        **_loader_kwargs(a.workers, a.seed + ctx.rank, dev, a.prefetch_factor,
                         persistent=a.workers > 0))
    vl, val_indices = _make_eval_loader(vds, a, ctx)
    if len(tl) == 0:
        raise ValueError(f"{len(train_ids)} training studies cannot form a global batch of {global_bs}; "
                         "reduce --bs / process count or check labels and corpus IDs.")
    used_per_epoch = len(tl) * global_bs
    ctx.log(f"[data] steps/epoch={len(tl)}, used/epoch={used_per_epoch}, "
            f"dropped tail/epoch={len(train_ids) - used_per_epoch}; sampler reshuffles every epoch")

    import timm
    use_pretrained = a.ckpt in ("timm", "pretrained")
    ctx.log("[backbone] Building ConvNeXt-Small; rank 0 initializes pretrained weights, then DDP synchronizes.")
    bb = build_backbone(a.arch, pretrained=use_pretrained and ctx.is_main, res=a.res,
                        pretrained_path=a.pretrained_path if ctx.is_main else None)
    src = (os.path.abspath(a.pretrained_path) if a.pretrained_path else
           f"timm-pretrained:{a.arch}" if use_pretrained else "random-init")
    backbone_meta = dict(backbone_backend="timm", timm_version=timm.__version__,
                         backbone_num_features=int(bb.num_features),
                         baseline_version="cur_best_eval94_dense96_convnext",
                         dense_corpus=DENSE_CONTRACT,
                         dense_span=list(DENSE_SPAN),
                         dense_slots=[list(slot) for slot in DENSE_SLOTS])
    parallel_meta = dict(world_size=ctx.world_size, per_gpu_bs=a.bs, global_bs=global_bs,
                         steps_per_epoch=len(tl), used_studies_per_epoch=used_per_epoch,
                         batchnorm="not-present (ConvNeXt LayerNorm); rank0-buffers-for-eval")
    checkpoint_meta = dict(backbone_meta, **parallel_meta, norm=a.norm, k=a.k, k_eval=a.k_eval,
                           eval_window_batch=a.eval_window_batch,
                           corpus_shape=list(corpus_shape), roi=a.roi, roi_mode=a.roi_mode,
                           roi_pad=a.roi_pad, roi_overlap=a.roi_overlap,
                           ema_enabled=a.ema, ema_decay=a.ema_decay,
                           weight_source="ema" if a.ema else "online")
    if ctx.is_main:
        _atomic_json(os.path.join(a.output_dir, f"run_{a.tag}.json"),
                     dict(vars(a), **backbone_meta, **parallel_meta, devices=devices,
                          corpus_shape=list(corpus_shape), n_train=len(train_ids), n_gold=len(gold_ids),
                          weight_source="ema" if a.ema else "online"))

    if a.grad_ckpt:
        # This architecture has a fixed training graph. static_graph=True below
        # supports timm versions using either reentrant or non-reentrant checkpointing.
        bb.set_grad_checkpointing(True)
        ctx.log("[raptor] gradient checkpointing ON")
    model = RaptorClassifier(bb, F_dim=bb.num_features).to(dev)
    if a.freeze_blocks > 0:
        prefixes = tuple(f"stages.{b}." for b in range(a.freeze_blocks))
        frozen = 0
        for name, parameter in model.backbone.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        ctx.log(f"[raptor] frozen backbone parameters: {frozen / 1e6:.2f}M")

    # Keep the raw model unwrapped; DDP trains it and EMA owns validation/checkpoint weights.
    train_model = DDP(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank,
                      broadcast_buffers=True, find_unused_parameters=False,
                      gradient_as_bucket_view=True, static_graph=True) if ctx.distributed else model
    # Construct EMA only after DDP has broadcast rank-0 initialization to every rank.
    ema = ModelEMA(model, decay=a.ema_decay) if a.ema else None
    eval_model = ema.module if ema is not None else model
    ctx.log(f"[ema] {'ON' if ema is not None else 'OFF'}"
            + (f" | decay={a.ema_decay}" if ema is not None else ""))
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    bb_params = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.bb_lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.bb_lr, a.head_lr], total_steps=len(tl) * a.epochs, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Y = np.stack([labels[u] for u in gold_ids]).astype(np.float32)

    def evaluate(target_model=None, name=None):
        target_model = eval_model if target_model is None else target_model
        name = ("val-ema" if ema is not None else "val") if name is None else name
        P = _predict_distributed(target_model, vl, val_indices, len(gold_ids), ctx, amp_dtype,
                                 a.eval_window_batch, name=name)
        au, aucs = _auc_metrics(Y, P)
        return au, aucs, P

    def snapshot(ep, au, aucs):
        # Non-main ranks retain only small metadata, never top-K CPU weight copies.
        state = ({key: value.detach().cpu().clone()
                  for key, value in eval_model.state_dict().items()} if ctx.is_main else None)
        return dict(checkpoint_meta, model=state, gold_auc=au, aucs=aucs, src=src,
                    res=a.res, arch=a.arch, lab=LAB, epoch=ep,
                    ema_updates=ema.updates if ema is not None else 0)

    def save_predictions(filename, P, truth=Y, ids=gold_ids, **extra):
        if ctx.is_main:
            np.savez(os.path.join(a.output_dir, filename), pred=P, truth=truth, ids=np.array(ids), **extra)

    best = -float("inf")
    best_state = None
    best_P = None
    topk = []
    hist = []
    t0 = time.time()

    def save_summary(partial):
        if ctx.is_main:
            _atomic_json(os.path.join(a.output_dir, f"raptor_ft_{a.tag}.json"),
                         dict(tag=a.tag, src=src, best_gold_auc=best, aucs=best_state["aucs"],
                              hist=hist, res=a.res, epochs=a.epochs, bb_lr=a.bb_lr, head_lr=a.head_lr,
                              k=a.k, k_eval=a.k_eval, eval_bs=a.eval_bs,
                              eval_window_batch=a.eval_window_batch,
                              n_train=len(train_ids), n_gold=len(gold_ids), roi=a.roi, roi_mode=a.roi_mode,
                              ema_enabled=ema is not None, ema_decay=a.ema_decay,
                              ema_updates=ema.updates if ema is not None else 0,
                              weight_source="ema" if ema is not None else "online",
                              partial=partial, epochs_done=len(hist), **parallel_meta))

    ctx.log("[train] Starting training.")
    for ep in range(a.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        train_model.train()
        totals = torch.zeros(2, dtype=torch.float64, device=dev)
        epoch_start = time.time()
        skipped_steps = 0
        for step, (x, y) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=amp_type, enabled=amp_enabled):
                logits = train_model(x)
                loss = lossf(logits.float(), y)
            if ep == 0 and step == 0:
                ctx.log(f"[train shapes] rank0 input={tuple(x.shape)}, labels={tuple(y.shape)}, "
                        f"logits={tuple(logits.shape)}; {ctx.world_size} GPUs have separate study batches")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 3.0)
            previous_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                sched.step()
                if ema is not None:
                    ema.update(model)
            else:
                skipped_steps += 1
            totals[0] += loss.detach().double() * len(y)
            totals[1] += len(y)
            if step == 0 or (step + 1) % a.log_every == 0 or step + 1 == len(tl):
                ctx.log(f"[train] ep{ep} step={step + 1}/{len(tl)} "
                        f"rank0_loss={loss.item():.4f}, elapsed={time.time() - epoch_start:.0f}s")
        if ctx.distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        train_loss = (totals[0] / totals[1]).item()
        ctx.log("[val] Predicting all validation shards and computing AUC on the combined studies.")
        au, aucs, P = evaluate()
        hist.append(dict(ep=ep, loss=train_loss, gold_auc=au,
                         weight_source="ema" if ema is not None else "online",
                         ema_updates=ema.updates if ema is not None else 0,
                         optimizer_steps=len(tl) - skipped_steps, amp_skipped_steps=skipped_steps))
        improved = au > best
        keep_epoch = len(topk) < a.topk or au > min(t["gold_auc"] for t in topk)
        candidate = snapshot(ep, au, aucs) if improved or keep_epoch else None
        if improved:
            best, best_P, best_state = au, P, candidate
            if ctx.is_main:
                _atomic_checkpoint(os.path.join(a.output_dir, f"raptor_ft_{a.tag}.pt"), best_state)
            save_predictions(f"raptor_gold_{a.tag}.npz", best_P)
            ctx.log(f"[ckpt] Saved best {'EMA' if ema is not None else 'online'} epoch {ep}: "
                    f"gold macro-AUC={best:.4f}")
        if keep_epoch:
            topk.append(dict(candidate, P=P))
            topk.sort(key=lambda t: -t["gold_auc"])
            del topk[a.topk:]
        save_summary(partial=True)
        ctx.log(f"ep{ep} global_loss={train_loss:.4f} | "
                f"GOLD {'EMA ' if ema is not None else ''}macro-AUC={au:.4f} "
                f"(best={best:.4f}) | total={time.time() - t0:.0f}s")

    save_summary(partial=False)
    ctx.log(f"DONE training | BEST GOLD {'EMA ' if a.ema else ''}macro-AUC={best:.4f}")
    for name, value in best_state["aucs"].items():
        ctx.log(f"  {name:18s} {value:.4f}")

    # No further backward passes: free optimizer/last-batch tensors before OOF/SWA.
    opt.zero_grad(set_to_none=True)
    del train_model, opt, sched, scaler, x, y, logits, loss, totals, candidate, eval_model, ema
    if dev == "cuda":
        torch.cuda.empty_cache()

    # CV OOF uses the same no-padding distributed prediction path as gold.
    ol = None
    if a.folds > 0 and oof_ids:
        ods = StudyWindows(HERE, oof_ids, id2row, labels, a.res, a.k_eval,
                           train=False, norm=a.norm, **_roi_kw)
        ol, oof_indices = _make_eval_loader(ods, a, ctx)
        Yo = np.stack([labels[u] for u in oof_ids]).astype(np.float32)
        oof_by_ep = {}
        for t in topk:
            _load_shared_state(model, t["model"], ctx)
            Pe = _predict_distributed(model, ol, oof_indices, len(oof_ids), ctx, amp_dtype,
                                      a.eval_window_batch, name="oof")
            epoch = int(t["epoch"])
            oof_by_ep[epoch] = Pe
            suffix = "" if epoch == best_state["epoch"] else f"_ep{epoch}"
            save_predictions(f"raptor_oof_{a.tag}{suffix}_fold{a.fold}.npz", Pe,
                             truth=Yo, ids=oof_ids, fold=a.fold, nfolds=a.folds)
            ctx.log(f"[cv] Saved OOF for epoch {epoch}; truth contains soft labels.")
        if len(oof_by_ep) > 1:
            save_predictions(f"raptor_oof_{a.tag}_epens_fold{a.fold}.npz",
                             np.mean(list(oof_by_ep.values()), axis=0),
                             truth=Yo, ids=oof_ids, fold=a.fold, nfolds=a.folds)

    # Preserve top-K checkpoint averaging; with EMA enabled these are top EMA epochs.
    # No extra BatchNorm recalibration is introduced.
    if len(topk) > 1:
        epochs = [int(t["epoch"]) for t in topk]
        avg = None
        if ctx.is_main:
            avg = {}
            for key, value in topk[0]["model"].items():
                if value.is_floating_point():
                    acc = torch.zeros_like(value, dtype=torch.float64)
                    for t in topk:
                        acc.add_(t["model"][key].double())
                    avg[key] = acc.div_(len(topk)).to(value.dtype)
                else:
                    avg[key] = value.clone()
        _load_shared_state(model, avg, ctx)
        au_swa, aucs_swa, P_swa = evaluate(model, name="val-swa")
        if ctx.is_main:
            swa_state = dict(checkpoint_meta, model=avg, gold_auc=au_swa, aucs=aucs_swa,
                             src=src, res=a.res, arch=a.arch, lab=LAB, epoch=epochs, swa_over=epochs)
            swa_state["weight_source"] = "ema_topk_swa" if a.ema else "online_topk_swa"
            _atomic_checkpoint(os.path.join(a.output_dir, f"raptor_ft_{a.tag}_swa.pt"), swa_state)
        save_predictions(f"raptor_gold_{a.tag}_swa.npz", P_swa)
        ctx.log(f"SWA over epochs {epochs} | gold={au_swa:.4f}, best-epoch={best:.4f}")
        if ol is not None:
            Ps = _predict_distributed(model, ol, oof_indices, len(oof_ids), ctx, amp_dtype,
                                      a.eval_window_batch, name="oof-swa")
            save_predictions(f"raptor_oof_{a.tag}_swa_fold{a.fold}.npz", Ps,
                             truth=Yo, ids=oof_ids, fold=a.fold, nfolds=a.folds)
        for t in topk[1:]:
            save_predictions(f"raptor_gold_{a.tag}_ep{t['epoch']}.npz", t["P"])
        Pens = np.mean([t["P"] for t in topk], axis=0)
        ensemble_auc, _ = _auc_metrics(Y, Pens)
        save_predictions(f"raptor_gold_{a.tag}_epens.npz", Pens)
        ctx.log(f"TOPK epochs {epochs} | best={best:.4f}, epoch-ensemble={ensemble_auc:.4f}")

    _load_shared_state(model, best_state["model"], ctx)
    ctx.log(f"[done] All outputs saved to {a.output_dir}")


def main():
    a = parse_args()
    ctx = DistributedContext(a.ddp_timeout_minutes)
    try:
        train(a, ctx)
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
