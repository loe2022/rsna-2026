"""cur_best_eval94_dense96：与训练验证对齐的Max-Span Dense-96双GPU提交代码。

将本文件全部复制到 Kaggle Notebook 的一个 Code 单元格，启用 GPU 后直接运行。
通过 WINDOW_COUNTS 设置一个整数窗口数，最终生成 submission.csv。每张GPU各加载
一份相同权重并处理不同study；不足两张GPU时自动降级。

相对 submit.py 的无损加速：权重只反序列化一次、每张GPU使用持久DICOM线程池、
CPU预处理与上一例GPU推理流水重叠、原生未压缩PixelData安全直读（其余自动回退）、
固定内存与non-blocking拷贝。测试集使用与build_corpus_dense96.py一致的
首个匹配series、2--98% Max-Span、容量感知配额转移和无重复真实切片；
切片、窗位、缩放、归一化、精度和MIL计算均与训练验证对齐。
"""

COMPETITION_DIR = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
# 训练完成并上传后，把这里改成Dense-96缓存训练所得的SWA权重路径。
WEIGHT_PATH = "/kaggle/input/datasets/renmingqian/dense96/raptor_ft_dev_swa.pt"
# 必须使用本目录train_knee.py在Dense-96缓存上训练的权重。
# 94是96张真实切片可形成的最大连续三切片窗口数；物理切片不足时自动减少。
WINDOW_COUNTS = 94
OUTPUT_DIR = "/kaggle/working"
WINDOW_BATCH = 24  # 与train_knee.py验证一致：分块编码，完整真实窗口集合只做一次MIL。
NUM_GPUS = 2      # 最多使用的GPU数量；只有一张GPU时自动按单卡运行。
DICOM_IO_THREADS = 8  # 每张GPU对应的持久DICOM读取线程数；双卡共16个。
PREFETCH_STUDIES = True  # CPU准备下一例时，当前例在GPU上推理。
PIN_MEMORY = True       # 锁页内存 + non-blocking CPU->GPU 拷贝。

import os, glob, time, gc, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
_CUDA_OOM_ERROR = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
# Fixed window shapes; mixed precision is selected for the active GPU below.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---- fixed config (must match training exactly) -----------------------------
IMG = 336
CROP_MM = 140.0
# Nominal proportions are 1.5x the public 18/14/12/8/12 Raptor layout.
SLOTS = [("Sagittal", 1, 27), ("Sagittal", 0, 21), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = 96
SPAN_LO, SPAN_HI = 0.02, 0.98
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ============================================================================
# Model -- mirrors train_knee.py (same parameter names and attention head)
# ============================================================================
COATNET_ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"


def build_backbone(arch, res):
    import timm
    # Construct the same backbone as training at the saved resolution.
    # Full fine-tuned weights are loaded afterwards; inference never downloads pretrained weights.
    return timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3,
                             img_size=res, global_pool="avg")


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=1024, n=12, drop=0.2):
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


def checkpoint_settings(ck):
    required = ["model", "arch", "res", "lab", "norm", "k_eval", "corpus_shape",
                "roi", "dense_corpus", "dense_span", "dense_slots"]
    missing = [key for key in required if key not in ck]
    if missing:
        raise ValueError(f"Checkpoint lacks {missing}; use the checkpoint from this train_knee.py.")
    if list(ck["lab"]) != LAB:
        raise ValueError(f"Checkpoint label order differs from LAB: {ck['lab']}")
    if ck["roi"]:
        raise ValueError("This submit.py handles --roi=False training only; ROI inference needs its localizer.")
    shape = list(ck["corpus_shape"])
    if len(shape) != 4 or shape[1:] != [MAXS, IMG, IMG]:
        raise ValueError(f"Expected a training corpus of (N,{MAXS},{IMG},{IMG}), got {shape}.")
    if ck["dense_corpus"] != "maxspan_dense96_capacity_aware_unique_first_series":
        raise ValueError("Checkpoint was not trained with this Dense-96 preprocessing contract.")
    if tuple(ck["dense_span"]) != (SPAN_LO, SPAN_HI):
        raise ValueError(f"Checkpoint Dense span differs: {ck['dense_span']}")
    if tuple(tuple(slot) for slot in ck["dense_slots"]) != tuple(SLOTS):
        raise ValueError(f"Checkpoint Dense slots differ: {ck['dense_slots']}")
    norm, res, k_eval = ck["norm"], int(ck["res"]), int(ck["k_eval"])
    if norm not in ("none", "imagenet", "coatnet") or min(res, k_eval) < 1:
        raise ValueError("Invalid checkpoint norm/res/k_eval.")
    backend = ck.get("backbone_backend")
    if backend != "timm" or ck["arch"] != COATNET_ARCH:
        raise ValueError("请使用 baseline_v0 对应的 CoAtNet 权重；DINOv2 权重不能加载到 CoAtNet。")
    if res < 32 or res % 32:
        raise ValueError("CoAtNet checkpoint res must be a positive multiple of 32.")
    if k_eval != WINDOW_COUNTS:
        raise ValueError(
            f"Expected submission-aligned k_eval={WINDOW_COUNTS}, got {k_eval}."
        )
    trained_window_batch = int(ck.get("eval_window_batch", -1))
    if trained_window_batch != WINDOW_BATCH:
        raise ValueError(
            f"Expected eval_window_batch={WINDOW_BATCH}, got {trained_window_batch}. "
            "Use a cur_best_eval94_dense96 checkpoint."
        )
    return dict(backend=backend, arch=ck["arch"], res=res, k_eval=k_eval, norm=norm)


def load_checkpoint(pt_path):
    import timm
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"找不到 CoAtNet 权重：{pt_path}，请检查 WEIGHT_PATH。")
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    settings = checkpoint_settings(ck)
    print(f"[checkpoint] {pt_path} | gold_auc={ck.get('gold_auc')} | "
          f"epoch={ck.get('epoch')}", flush=True)
    print(f"[timm] trained={ck.get('timm_version', 'unknown')}, inference={timm.__version__}", flush=True)
    return ck, settings


def load_model(ck, settings, device):
    bb = build_backbone(settings["arch"], res=settings["res"])
    if int(ck.get("backbone_num_features", bb.num_features)) != bb.num_features:
        raise ValueError("Checkpoint feature dimension does not match the CoAtNet backbone.")
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    gc.collect()
    return model


# ============================================================================
# Eval windowing -- matches train_knee.py StudyWindows(train=False, roi=False).
# Dense valid planes are contiguous; never repeat centers when fewer than 94 exist.
# ============================================================================
def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    k = min(k, len(cs))
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    if len(np.unique(idx)) != k:
        raise AssertionError("Dense eval center selection produced duplicates.")
    return [cs[i] for i in idx]


def windows_from_centers(vol, centers, res, norm):
    D = vol.shape[0]
    cs = list(centers)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res or t.shape[-2] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    elif norm == "coatnet":
        x = (x - 0.5) / 0.5
    return x


def eval_windows(vol, mask, k, res, norm):
    """保留单个 K 的原始接口，便于校验与旧 submit.py 一致。"""
    return windows_from_centers(
        vol, _eval_centers(mask, vol.shape[0], k), res=res, norm=norm
    )


def amp_dtype(device, precision):
    device = torch.device(device)
    if device.type == "cpu":
        if precision not in ("auto", "fp32"):
            raise ValueError("Use auto or fp32 on CPU.")
        return None
    with torch.cuda.device(device):
        native_bf16 = (torch.cuda.get_device_capability(device)[0] >= 8
                       and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if precision == "auto":
        return torch.bfloat16 if native_bf16 else torch.float16
    if precision == "bf16" and not native_bf16:
        raise ValueError("This GPU lacks native BF16; use --precision auto.")
    return {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def _effective_window_batch(model, xwins, window_batch):
    if window_batch < 0 or len(xwins) < 1:
        raise ValueError("window_batch must be nonnegative and windows must be nonempty.")
    requested = window_batch or len(xwins)  # 0 = encode the whole study together.
    remembered = getattr(model, "_inference_window_batch_limit", requested)
    return min(requested, remembered, len(xwins))


def _copy_to_device(x, device):
    """保持dtype/数值不变；锁页CPU张量在CUDA上使用异步拷贝。"""
    non_blocking = device.type == "cuda" and x.device.type == "cpu" and x.is_pinned()
    return x.to(device, non_blocking=non_blocking)


@torch.inference_mode()
def infer_probs(model, xwins, device, dtype, window_batch):
    device = torch.device(device)
    batch_size = _effective_window_batch(model, xwins, window_batch)

    def predict(active_dtype):
        amp_context = (torch.autocast(device_type=device.type, dtype=active_dtype)
                       if active_dtype is not None else nullcontext())
        with amp_context:
            if batch_size >= len(xwins):
                features = model.backbone(_copy_to_device(xwins, device))
            else:
                features = torch.cat([
                    model.backbone(_copy_to_device(windows, device))
                    for windows in xwins.split(batch_size)
                ], dim=0)
            # Preserve every window, in order. Apply MIL once to the complete study.
            logits = model.head(features.unsqueeze(0))
            probs = torch.sigmoid(logits.float())[0].cpu().numpy()
        if not np.isfinite(probs).all():
            raise FloatingPointError("Non-finite predictions.")
        return probs

    active_dtype = dtype
    while True:
        release_cuda_cache = False
        try:
            return predict(active_dtype)
        except _CUDA_OOM_ERROR as exc:
            if (device.type != "cuda" or batch_size <= 1
                    or "out of memory" not in str(exc).lower()):
                raise
            previous = batch_size
            batch_size = max(1, batch_size // 2)
            model._inference_window_batch_limit = batch_size
            print(f"[memory] CUDA OOM: window batch {previous} -> {batch_size}; retrying all windows. Future studies reuse this limit.", flush=True)
            release_cuda_cache = True
        except FloatingPointError:
            if active_dtype is None:
                raise
            print("[precision] Non-finite mixed-precision predictions; retrying this study in FP32.", flush=True)
            active_dtype = None
        # Leave the exception handler first so failed-forward tensors can be freed.
        if release_cuda_cache:
            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


@torch.inference_mode()
def infer_multi_probs(model, xwins, gather_indices, device, dtype, window_batch):
    """共享一次 backbone 特征提取，对每个 K 独立执行 MIL head。

    gather_indices[k] 保留该 K 原始的窗口顺序和重复项，所以结果与
    对每个 K 单独调用 eval_windows + infer_probs 等价。
    """
    device = torch.device(device)
    batch_size = _effective_window_batch(model, xwins, window_batch)

    def predict(active_dtype):
        amp_context = (torch.autocast(device_type=device.type, dtype=active_dtype)
                       if active_dtype is not None else nullcontext())
        with amp_context:
            if batch_size >= len(xwins):
                features = model.backbone(_copy_to_device(xwins, device))
            else:
                features = torch.cat([
                    model.backbone(_copy_to_device(windows, device))
                    for windows in xwins.split(batch_size)
                ], dim=0)

            result = {}
            for k, indices in gather_indices.items():
                index = torch.as_tensor(indices, dtype=torch.long, device=features.device)
                selected = features.index_select(0, index).unsqueeze(0)
                logits = model.head(selected)
                probs = torch.sigmoid(logits.float())[0].cpu().numpy()
                if not np.isfinite(probs).all():
                    raise FloatingPointError(f"Non-finite predictions for K={k}.")
                result[k] = probs
        return result

    active_dtype = dtype
    while True:
        release_cuda_cache = False
        try:
            return predict(active_dtype)
        except _CUDA_OOM_ERROR as exc:
            if (device.type != "cuda" or batch_size <= 1
                    or "out of memory" not in str(exc).lower()):
                raise
            previous = batch_size
            batch_size = max(1, batch_size // 2)
            model._inference_window_batch_limit = batch_size
            print(f"[memory] CUDA OOM: window batch {previous} -> {batch_size}; "
                  "retrying all unique windows.", flush=True)
            release_cuda_cache = True
        except FloatingPointError:
            if active_dtype is None:
                raise
            print("[precision] Non-finite mixed-precision predictions; "
                  "retrying this study in FP32.", flush=True)
            active_dtype = None
        if release_cuda_cache:
            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


# ============================================================================
# DICOM reader and Dense-96 sampling match build_corpus_dense96.py.
# ============================================================================
def _dicom_files(series_dir):
    files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not files and os.path.isdir(series_dir):
        files = [str(path) for path in Path(series_dir).iterdir() if path.is_file()]
    return files


def _normalize_plane(value):
    text = str(value).strip().lower()
    return {
        "sagittal": "Sagittal",
        "coronal": "Coronal",
        "axial": "Axial",
    }.get(text, str(value).strip())


def _normalize_fluid(value):
    import pandas as pd
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n"}:
        return 0
    return None


def _make_reader(io_executor=None):
    import pydicom, cv2
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_modality_lut

    def _map(func, items):
        return list(io_executor.map(func, items)) if io_executor is not None else list(map(func, items))

    def header_record(f):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            iop = getattr(h, 'ImageOrientationPatient', None)
            ipp = getattr(h, 'ImagePositionPatient', None)
            instance = float(getattr(h, 'InstanceNumber', 0) or 0)
            if iop is not None and ipp is not None and len(iop) == 6 and len(ipp) >= 3:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                n = np.cross(r, c); pos = float(np.dot(np.array(ipp[:3], float), n))
            else:
                pos = instance
            ps = getattr(h, 'PixelSpacing', None)
            ps = float(ps[0]) if ps is not None and len(ps) else 0.5
            if not np.isfinite(ps) or ps <= 0:
                ps = 0.5
            return (pos, instance, f, ps), ps
        except Exception:
            # Keep the original fallback and do not include it in the median.
            return (0.0, 0.0, f, 0.5), None

    def order_and_meta(sdir):
        fs = _dicom_files(sdir)
        records = _map(header_record, fs)
        recs = [record for record, _ in records]
        ps_list = [ps for _, ps in records if ps is not None]
        recs.sort(key=lambda x: (x[0], x[1], x[2]))
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, _, f, ps in recs], med_ps

    def native_uncompressed_pixels(d):
        """只处理能证明与pixel_array等价的简单原生单帧灰度格式。"""
        file_meta = getattr(d, 'file_meta', None)
        transfer = getattr(file_meta, 'TransferSyntaxUID', None)
        if transfer is None or bool(getattr(transfer, 'is_compressed', True)):
            return None
        rows = int(getattr(d, 'Rows', 0) or 0)
        cols = int(getattr(d, 'Columns', 0) or 0)
        samples = int(getattr(d, 'SamplesPerPixel', 1) or 1)
        frames = int(getattr(d, 'NumberOfFrames', 1) or 1)
        bits_allocated = int(getattr(d, 'BitsAllocated', 0) or 0)
        bits_stored = int(getattr(d, 'BitsStored', 0) or 0)
        high_bit = int(getattr(d, 'HighBit', -1))
        photometric = str(getattr(d, 'PhotometricInterpretation', ''))
        if (rows < 1 or cols < 1 or samples != 1 or frames != 1
                or bits_allocated not in (8, 16, 32)
                or bits_stored != bits_allocated or high_bit != bits_stored - 1
                or photometric not in ('MONOCHROME1', 'MONOCHROME2')):
            return None
        signed = int(getattr(d, 'PixelRepresentation', 0) or 0) == 1
        byte_order = '>' if str(transfer) == '1.2.840.10008.1.2.2' else '<'
        dtype = np.dtype(byte_order + ('i' if signed else 'u') + str(bits_allocated // 8))
        count = rows * cols
        payload = getattr(d, 'PixelData', None)
        if payload is None or len(payload) < count * dtype.itemsize:
            return None
        return np.frombuffer(payload, dtype=dtype, count=count).reshape(rows, cols)

    def read_px(f):
        d = pydicom.dcmread(f, force=True)
        # Compressed, multi-frame, colour and packed-bit cases retain pydicom's decoder.
        try:
            raw = native_uncompressed_pixels(d)
        except Exception:
            raw = None
        a = apply_modality_lut(raw if raw is not None else d.pixel_array, d).astype(np.float32)
        a = np.squeeze(a)
        if a.ndim != 2:
            raise ValueError(f"Expected a 2D DICOM image, got {a.shape}: {f}")
        if str(getattr(d, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
            a = a.max() - a
        return a

    def read_many(file_ps_pairs):
        pairs = list(file_ps_pairs)
        unique_files = list(dict.fromkeys(f for f, _ in pairs))

        def read_one(f):
            try:
                return read_px(f)
            except Exception:
                return None

        # Dense sampling is unique; the dictionary also guards against accidental duplicates.
        arrays = dict(zip(unique_files, _map(read_one, unique_files)))
        return [(arrays[f], ps) for f, ps in pairs]

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = max(1, min(cpx, min(h, w))); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_many, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        # pandas reads this official column as bool/numeric; NaN simply has no preference.
        pref = [r for r in cands if r.get('Fluid_Sensitive') == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _eligible_bounds(length):
    if length < 1:
        return 0, -1
    lo = min(int(length * SPAN_LO), length - 1)
    hi = min(max(lo, int(length * SPAN_HI) - 1), length - 1)
    return lo, hi


def _dense_allocate(capacities, proportions, target):
    """Largest-remainder allocation with capacity-aware redistribution."""
    caps = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(proportions, dtype=np.float64)
    if caps.ndim != 1 or weights.shape != caps.shape:
        raise ValueError("capacities/proportions must be one-dimensional and aligned.")
    if np.any(caps < 0) or np.any(weights < 0) or target < 0:
        raise ValueError("capacities, proportions and target must be nonnegative.")
    quota = np.zeros_like(caps)
    remaining = min(int(target), int(caps.sum()))
    while remaining:
        active = np.flatnonzero(quota < caps)
        if not len(active):
            break
        active_weights = weights[active]
        if not np.any(active_weights > 0):
            active_weights = np.ones(len(active), dtype=np.float64)
        raw = remaining * active_weights / active_weights.sum()
        base = np.floor(raw).astype(np.int64)
        base = np.minimum(base, caps[active] - quota[active])
        granted = int(base.sum())
        if granted:
            quota[active] += base
            remaining -= granted
            if not remaining:
                break

        active = np.flatnonzero(quota < caps)
        if not len(active):
            break
        active_weights = weights[active]
        if not np.any(active_weights > 0):
            active_weights = np.ones(len(active), dtype=np.float64)
        raw = remaining * active_weights / active_weights.sum()
        fractions = raw - np.floor(raw)
        order = sorted(
            range(len(active)),
            key=lambda j: (-fractions[j], -active_weights[j], int(active[j])),
        )
        for j in order:
            slot = int(active[j])
            if remaining < 1:
                break
            if quota[slot] < caps[slot]:
                quota[slot] += 1
                remaining -= 1
    if int(quota.sum()) != min(int(target), int(caps.sum())):
        raise AssertionError("Dense quota allocation did not reach the feasible target.")
    return quota


def _dense_unique_indices(length, count):
    lo, hi = _eligible_bounds(length)
    capacity = max(0, hi - lo + 1)
    if count < 0 or count > capacity:
        raise ValueError(f"Cannot select {count} unique slices from capacity {capacity}.")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    picks = np.linspace(lo, hi, count).round().astype(np.int64)
    if len(np.unique(picks)) != count:
        raise AssertionError("Dense linspace unexpectedly produced repeated indices.")
    return picks


def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_many, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    mask = np.zeros(MAXS, np.uint8)
    used = set()
    selected_slots = []
    capacities = []
    proportions = []
    for plane, fluid, nominal in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        files, med_ps = [], 0.5
        series_uid = None
        if r is not None:
            series_uid = str(r['SeriesInstanceUID'])
            used.add(series_uid)
            files, med_ps = order_and_meta(f"{tsdir}/{sid}/{series_uid}")
        lo, hi = _eligible_bounds(len(files))
        capacities.append(max(0, hi - lo + 1))
        proportions.append(nominal)
        selected_slots.append((plane, fluid, nominal, series_uid, files, med_ps))

    quotas = _dense_allocate(capacities, proportions, MAXS)
    packed = []
    for slot_index, slot in enumerate(selected_slots):
        plane, fluid, nominal, series_uid, files, med_ps = slot
        quota = int(quotas[slot_index])
        if quota == 0:
            continue
        picks = _dense_unique_indices(len(files), quota)
        selected = [files[int(pick)] for pick in picks]
        decoded = read_many(selected)
        arrs = [a for a, _ in decoded]
        pss = [ps if a is not None else med_ps for a, ps in decoded]
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
            del allpx
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if a is None:
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            stored = (aw * 255).astype(np.uint8)
            if np.any(stored != 0):
                packed.append(stored)

    if len(packed) < 3:
        raise RuntimeError(
            f"Study {sid} has only {len(packed)} usable Dense-96 slices."
        )
    stored_count = min(len(packed), MAXS)
    vol[:stored_count] = np.stack(packed[:stored_count], axis=0)
    mask[:stored_count] = 1
    return vol, mask


# ============================================================================
# Test-root discovery + weights + main
# ============================================================================
def make_submission(pred, test_ids, sub_cols):
    import pandas as pd
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Duplicate StudyInstanceUID in test.csv.")
    if len(sub_cols) != 13 or set(sub_cols) != set(["StudyInstanceUID"] + LAB):
        raise ValueError(f"Unexpected sample_submission.csv columns: {sub_cols}")
    if pred.shape != (len(test_ids), len(LAB)):
        raise ValueError(f"Prediction shape mismatch: {pred.shape}")
    if not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 1)):
        raise ValueError("Submission predictions must be finite probabilities in [0,1].")
    sub = pd.DataFrame(pred, columns=LAB)
    sub.insert(0, "StudyInstanceUID", test_ids)
    return sub[sub_cols]


def main():
    import pandas as pd
    if isinstance(NUM_GPUS, bool) or not isinstance(NUM_GPUS, (int, np.integer)) or NUM_GPUS < 1:
        raise ValueError("NUM_GPUS 必须是正整数。")
    if (isinstance(DICOM_IO_THREADS, bool)
            or not isinstance(DICOM_IO_THREADS, (int, np.integer))
            or DICOM_IO_THREADS < 1):
        raise ValueError("DICOM_IO_THREADS 必须是正整数。")
    if WINDOW_BATCH < 0:
        raise ValueError("WINDOW_BATCH 必须为 0（自动整例）或正整数。")
    if isinstance(WINDOW_COUNTS, bool) or not isinstance(WINDOW_COUNTS, (int, np.integer)):
        raise TypeError("请把 WINDOW_COUNTS 设置为单个整数，例如 WINDOW_COUNTS = 42。")
    window_count = int(WINDOW_COUNTS)
    if window_count < 1 or window_count > MAXS - 2:
        raise ValueError(f"WINDOW_COUNTS 必须在1到{MAXS - 2}之间，当前为 {window_count}。")
    window_counts = (window_count,)
    t0 = time.time()
    if torch.cuda.is_available():
        gpu_count = min(int(NUM_GPUS), torch.cuda.device_count())
        devices = [torch.device(f"cuda:{index}") for index in range(gpu_count)]
    else:
        devices = [torch.device("cpu")]
    dtypes = [amp_dtype(device, "auto") for device in devices]
    device_summary = ", ".join(
        f"{device}={dtype or 'fp32'}" for device, dtype in zip(devices, dtypes)
    )
    print(f"workers={len(devices)} | devices=[{device_summary}] | torch={torch.__version__}",
          flush=True)

    root = COMPETITION_DIR
    for name in ("test.csv", "test_series.csv"):
        if not os.path.isfile(os.path.join(root, name)):
            raise FileNotFoundError(os.path.join(root, name))
    tsdir = next((os.path.join(root, name) for name in ("test_series", "test_images")
                  if os.path.isdir(os.path.join(root, name))), None)
    if tsdir is None:
        raise FileNotFoundError(f"No test_series or test_images directory in {root}")
    test = pd.read_csv(os.path.join(root, "test.csv"), dtype={"StudyInstanceUID": str})
    if test["StudyInstanceUID"].isna().any() or test["StudyInstanceUID"].duplicated().any():
        raise ValueError("test.csv has missing or duplicate study IDs.")
    test_ids = test["StudyInstanceUID"].tolist()
    if not test_ids:
        raise ValueError("No test studies found.")
    tser = pd.read_csv(os.path.join(root, "test_series.csv"),
                       dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    required = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]
    if any(col not in tser.columns for col in required):
        raise ValueError(f"test_series.csv must contain {required}")
    if tser["SeriesInstanceUID"].duplicated().any():
        raise ValueError("test_series.csv contains duplicate SeriesInstanceUID rows.")
    tser["Anatomical_Plane"] = tser["Anatomical_Plane"].map(_normalize_plane)
    tser["Fluid_Sensitive"] = tser["Fluid_Sensitive"].map(_normalize_fluid)
    ser = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID", sort=False)}
    ssub = os.path.join(root, "sample_submission.csv")
    sub_cols = list(pd.read_csv(ssub, nrows=0).columns) if os.path.isfile(ssub) else ["StudyInstanceUID"] + LAB
    # Validate output schema before spending time on inference.
    make_submission(np.full((len(test_ids), len(LAB)), 0.5, np.float32), test_ids, sub_cols)
    # 权重文件只从磁盘反序列化一次；state_dict再复制到每张GPU的独立模型。
    checkpoint, settings = load_checkpoint(WEIGHT_PATH)
    models = []
    for device in devices:
        models.append(load_model(checkpoint, settings, device))
    del checkpoint
    gc.collect()
    print(f"[model] {settings} | test_k={window_counts} | "
          f"window_batch={WINDOW_BATCH or 'all (OOM auto-reduce)'}", flush=True)
    if settings["k_eval"] not in window_counts:
        print(f"[note] checkpoint k_eval={settings['k_eval']} 不在 WINDOW_COUNTS 中。", flush=True)
    print(f"[data] {len(test_ids)} studies | {tsdir} | slots={SLOTS} | span={SPAN_LO}:{SPAN_HI}", flush=True)
    predictions = {
        k: np.full((len(test_ids), len(LAB)), 0.5, np.float32)
        for k in window_counts
    }
    progress_lock = threading.Lock()
    completed = 0

    def predict_shard(worker_index, study_indices):
        nonlocal completed
        device = devices[worker_index]
        model = models[worker_index]
        dtype = dtypes[worker_index]
        pin_enabled = bool(PIN_MEMORY and device.type == "cuda")
        device_context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        print(f"[worker {worker_index}] device={device} | studies={len(study_indices)} | "
              f"dicom_threads={DICOM_IO_THREADS} | prefetch={bool(PREFETCH_STUDIES)}",
              flush=True)

        with ThreadPoolExecutor(max_workers=int(DICOM_IO_THREADS),
                                thread_name_prefix=f"dicom-{worker_index}") as io_executor:
            reader = _make_reader(io_executor)

            def prepare_study(i):
                nonlocal pin_enabled
                sid = test_ids[i]
                try:
                    vol, mask = build_study(sid, ser, tsdir, reader)
                    if not mask.any():
                        raise ValueError(
                            "No nonzero slices; check DICOM paths, metadata, and pixel decoders."
                        )
                    # 各K中心、顺序和重复项与原submit.py完全相同。
                    center_sets = {
                        k: _eval_centers(mask, vol.shape[0], k)
                        for k in window_counts
                    }
                    # This version accepts exactly one K. Preserve all 78 positions,
                    # including repeated centers, so the backbone batch sequence is
                    # identical to training validation rather than deduplicating first.
                    only_k = window_counts[0]
                    encoded_centers = list(center_sets[only_k])
                    gather_indices = {only_k: list(range(len(encoded_centers)))}
                    xw = windows_from_centers(
                        vol, encoded_centers, res=settings["res"], norm=settings["norm"]
                    )
                    if pin_enabled:
                        try:
                            xw = xw.pin_memory()
                        except RuntimeError as exc:
                            pin_enabled = False
                            print(f"[worker {worker_index}] pin_memory unavailable; "
                                  f"falling back to blocking copies: {exc}", flush=True)
                    details = ", ".join(
                        f"K={k}: requested={len(center_sets[k])}, "
                        f"distinct={len(set(center_sets[k]))}"
                        for k in window_counts
                    )
                    volume_shape = vol.shape
                    del vol, mask, center_sets, encoded_centers
                    return i, sid, xw, gather_indices, volume_shape, details
                except Exception as exc:
                    raise RuntimeError(
                        f"DICOM preprocessing failed on {device} for study {sid}: {exc}"
                    ) from exc

            def prepared_studies():
                if not PREFETCH_STUDIES or len(study_indices) < 2:
                    for index in study_indices:
                        yield prepare_study(index)
                    return
                # 队列深度固定为1：下一例CPU预处理与当前例GPU推理重叠，避免RAM失控。
                iterator = iter(study_indices)
                with ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix=f"prep-{worker_index}") as prep_executor:
                    pending = prep_executor.submit(prepare_study, next(iterator))
                    while True:
                        prepared = pending.result()
                        try:
                            next_index = next(iterator)
                        except StopIteration:
                            yield prepared
                            break
                        pending = prep_executor.submit(prepare_study, next_index)
                        yield prepared

            with device_context:
                for i, sid, xw, gather_indices, volume_shape, details in prepared_studies():
                    if i == 0:
                        print(f"[shapes:{device}] volume={volume_shape}, "
                              f"eval_windows={tuple(xw.shape)}, each_output=(1,{len(LAB)})",
                              flush=True)
                        print(f"[windows] {details}", flush=True)
                    # Every worker owns one model/GPU; a study's complete MIL bag stays together.
                    study_predictions = infer_multi_probs(
                        model, xw, gather_indices, device, dtype, WINDOW_BATCH
                    )
                    for k in window_counts:
                        predictions[k][i] = study_predictions[k]
                    del xw, gather_indices, study_predictions
                    with progress_lock:
                        completed += 1
                        if completed % 20 == 0 or completed == len(test_ids):
                            print(f"[predict] {completed}/{len(test_ids)} | "
                                  f"{time.time()-t0:.0f}s", flush=True)
        return worker_index, len(study_indices)

    # Round-robin studies keep the two GPUs balanced while preserving output row order.
    shards = [list(range(worker_index, len(test_ids), len(devices)))
              for worker_index in range(len(devices))]
    with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="gpu-infer") as executor:
        futures = [executor.submit(predict_shard, worker_index, shard)
                   for worker_index, shard in enumerate(shards)]
        for future in as_completed(futures):
            worker_index, count = future.result()
            print(f"[worker {worker_index}] finished {count} studies on "
                  f"{devices[worker_index]}", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub = make_submission(predictions[window_count], test_ids, sub_cols)
    out = os.path.join(OUTPUT_DIR, "submission.csv")
    sub.to_csv(out, index=False)
    print(f"wrote {out} | K={window_count} | "
          f"{len(sub)} rows x {len(sub.columns)} columns | {time.time()-t0:.0f}s", flush=True)
    print(sub.head().to_string(index=False), flush=True)
    del models
    gc.collect()
    for device in devices:
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
