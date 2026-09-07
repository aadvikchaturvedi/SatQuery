"""
Standalone, notebook-independent module for the unified optical/SAR/fusion change-detection
model. Download this alongside unified_changenet_best.pt — the checkpoint has no idea what
the model architecture looks like on its own; this file is what reconstructs it.

Usage elsewhere (outside Kaggle):
    from change_detection_module import load_change_detector
    detector = load_change_detector("unified_changenet_best.pt", device="cuda")  # or "cpu"
    result = detector.predict(optical_t1=..., optical_t2=..., sar_t1=None, sar_t2=None, hw=(600,600))
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

PATCH_SIZE = 256
PATCH_STRIDE = 128
OPTICAL_CLIP_MAX = 3000.0


# ---------------------------------------------------------------------------
# Preprocessing — MUST match exactly what was used during training
# ---------------------------------------------------------------------------

def normalize_optical(x):
    x = np.clip(x, 0, OPTICAL_CLIP_MAX) / OPTICAL_CLIP_MAX
    return x.astype(np.float32)


def normalize_sar(x, lo_pct=1, hi_pct=99):
    lo = np.percentile(x, lo_pct)
    hi = np.percentile(x, hi_pct)
    x = np.clip(x, lo, hi)
    x = (x - lo) / max(hi - lo, 1e-6)
    return x.astype(np.float32)


def crop_patch(arr, y, x, patch_size):
    return arr[..., y:y + patch_size, x:x + patch_size]


# ---------------------------------------------------------------------------
# Model — copy of the trained architecture (Section 7 of the notebook)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_ch, chs=(16, 32, 64, 128, 256)):
        super().__init__()
        self.stages = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.stages.append(ConvBlock(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if i != len(self.stages) - 1:
                x = self.pool(x)
        return feats

class TemporalFusionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )
    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)
        return self.conv(torch.cat([f1, f2, diff], dim=1))

class CrossModalFusionBlock(nn.Module):
    def __init__(self, ch_opt, ch_sar, ch_out):
        super().__init__()
        self.proj_opt = nn.Conv2d(ch_opt, ch_out, 1)
        self.proj_sar = nn.Conv2d(ch_sar, ch_out, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_out * 2, ch_out, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, 2, 1), nn.Softmax(dim=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(ch_out * 2, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
        )
    def forward(self, f_opt, f_sar):
        o = self.proj_opt(f_opt)
        s = self.proj_sar(f_sar)
        w = self.gate(torch.cat([o, s], dim=1))
        return self.fuse(torch.cat([o * w[:, 0:1], s * w[:, 1:2]], dim=1))

class SingleModalProj(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )
    def forward(self, f):
        return self.conv(f)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UnifiedChangeDetectionNet(nn.Module):
    def __init__(self, optical_in_ch=13, sar_in_ch=2, base_chs=(16, 32, 64, 128, 256)):
        super().__init__()
        self.base_chs = base_chs
        self.opt_encoder = UNetEncoder(optical_in_ch, base_chs)
        self.opt_temporal = nn.ModuleList([TemporalFusionBlock(c) for c in base_chs])
        self.sar_encoder = UNetEncoder(sar_in_ch, base_chs)
        self.sar_temporal = nn.ModuleList([TemporalFusionBlock(c) for c in base_chs])
        self.cross_fusion = nn.ModuleList([CrossModalFusionBlock(c, c, c) for c in base_chs])
        self.single_proj = nn.ModuleList([SingleModalProj(c) for c in base_chs])
        dec_chs = list(reversed(base_chs))
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(dec_chs[i], dec_chs[i + 1], dec_chs[i + 1]) for i in range(len(dec_chs) - 1)
        ])
        self.final_conv = nn.Conv2d(dec_chs[-1], 1, 1)

    def _encode_temporal(self, encoder, temporal_blocks, x1, x2):
        f1, f2 = encoder(x1), encoder(x2)
        return [tb(a, b) for tb, a, b in zip(temporal_blocks, f1, f2)]

    def forward(self, optical_t1=None, optical_t2=None, sar_t1=None, sar_t2=None, mode="fusion"):
        assert mode in ("optical", "sar", "fusion")
        if mode == "optical":
            feats = self._encode_temporal(self.opt_encoder, self.opt_temporal, optical_t1, optical_t2)
            feats = [proj(f) for proj, f in zip(self.single_proj, feats)]
            ref = optical_t1
        elif mode == "sar":
            feats = self._encode_temporal(self.sar_encoder, self.sar_temporal, sar_t1, sar_t2)
            feats = [proj(f) for proj, f in zip(self.single_proj, feats)]
            ref = sar_t1
        else:
            opt_feats = self._encode_temporal(self.opt_encoder, self.opt_temporal, optical_t1, optical_t2)
            sar_feats = self._encode_temporal(self.sar_encoder, self.sar_temporal, sar_t1, sar_t2)
            feats = [cf(o, s) for cf, o, s in zip(self.cross_fusion, opt_feats, sar_feats)]
            ref = optical_t1
        x = feats[-1]
        skips = feats[:-1][::-1]
        for block, skip in zip(self.decoder_blocks, skips):
            x = block(x, skip)
        logits = self.final_conv(x)
        if logits.shape[-2:] != ref.shape[-2:]:
            logits = F.interpolate(logits, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return logits


# ---------------------------------------------------------------------------
# Inference helpers + ChangeDetector wrapper
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_full_scene(model, data, mode, device, patch_size=PATCH_SIZE, stride=PATCH_STRIDE, batch_size=16):
    h, w = data["hw"]
    prob_sum = np.zeros((h, w), dtype=np.float32)
    prob_count = np.zeros((h, w), dtype=np.float32)
    ys = list(range(0, max(h - patch_size, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - patch_size, 0) + 1, stride)) or [0]
    if h > ys[-1] + patch_size: ys.append(h - patch_size)
    if w > xs[-1] + patch_size: xs.append(w - patch_size)
    coords = [(y, x) for y in ys for x in xs]

    def get_patch_tensors(y, x):
        kw = {}
        if mode in ("optical", "fusion"):
            kw["optical_t1"] = torch.from_numpy(normalize_optical(crop_patch(data["optical_t1"], y, x, patch_size)))
            kw["optical_t2"] = torch.from_numpy(normalize_optical(crop_patch(data["optical_t2"], y, x, patch_size)))
        if mode in ("sar", "fusion"):
            kw["sar_t1"] = torch.from_numpy(normalize_sar(crop_patch(data["sar_t1"], y, x, patch_size)))
            kw["sar_t2"] = torch.from_numpy(normalize_sar(crop_patch(data["sar_t2"], y, x, patch_size)))
        return kw

    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        from collections import defaultdict
        batch_kw = defaultdict(list)
        for y, x in chunk:
            for k, v in get_patch_tensors(y, x).items():
                batch_kw[k].append(v)
        batch_kw = {k: torch.stack(v, dim=0).to(device) for k, v in batch_kw.items()}
        logits = model(**batch_kw, mode=mode)
        probs = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        for (y, x), p in zip(chunk, probs):
            ph, pw = min(patch_size, h - y), min(patch_size, w - x)
            prob_sum[y:y + ph, x:x + pw] += p[:ph, :pw]
            prob_count[y:y + ph, x:x + pw] += 1.0
    return prob_sum / np.maximum(prob_count, 1e-6)


def extract_bounding_boxes(binary_mask, prob_map, pixel_size_m=10.0, min_area=25):
    labeled, n = ndi.label(binary_mask)
    boxes, region_id = [], 0
    for comp_id in range(1, n + 1):
        ys, xs = np.where(labeled == comp_id)
        area_px = len(ys)
        if area_px < min_area:
            continue
        region_id += 1
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        boxes.append({
            "region_id": region_id,
            "bbox_px": [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)],
            "area_px": int(area_px),
            "area_m2": float(area_px * (pixel_size_m ** 2)),
            "mean_confidence": float(prob_map[ys, xs].mean()),
            "centroid_px": [float(xs.mean()), float(ys.mean())],
        })
    return boxes


class ChangeDetector:
    def __init__(self, model, device, temperatures=None, pixel_size_m=10.0, threshold=0.5, min_area=25):
        self.model = model.to(device).eval()
        self.device = device
        self.temperatures = temperatures or {"optical": 1.0, "sar": 1.0, "fusion": 1.0}
        self.pixel_size_m = pixel_size_m
        self.threshold = threshold
        self.min_area = min_area

    def _select_mode(self, has_optical, has_sar):
        if has_optical and has_sar: return "fusion"
        if has_sar: return "sar"
        if has_optical: return "optical"
        raise ValueError("Need at least one modality (optical or SAR).")

    @torch.no_grad()
    def predict(self, optical_t1=None, optical_t2=None, sar_t1=None, sar_t2=None, hw=None, region_id="scene"):
        has_optical = optical_t1 is not None and optical_t2 is not None
        has_sar = sar_t1 is not None and sar_t2 is not None
        mode = self._select_mode(has_optical, has_sar)
        data = {"hw": hw}
        if has_optical: data["optical_t1"], data["optical_t2"] = optical_t1, optical_t2
        if has_sar: data["sar_t1"], data["sar_t2"] = sar_t1, sar_t2

        raw_prob_map = predict_full_scene(self.model, data, mode, self.device)
        logits_equiv = np.log(np.clip(raw_prob_map, 1e-6, 1 - 1e-6) / np.clip(1 - raw_prob_map, 1e-6, 1 - 1e-6))
        T = self.temperatures.get(mode, 1.0)
        confidence_map = 1 / (1 + np.exp(-logits_equiv / T))
        binary_mask = (confidence_map > self.threshold).astype(np.uint8)
        regions = extract_bounding_boxes(binary_mask, confidence_map, self.pixel_size_m, self.min_area)

        return {
            "region": region_id, "model_mode": mode,
            "change_mask": binary_mask, "confidence_map": confidence_map,
            "regions": regions, "change_percentage": float(100.0 * binary_mask.mean()),
        }

    def predict_to_json(self, **kwargs):
        result = self.predict(**kwargs)
        return {
            "region": result["region"], "model_mode": result["model_mode"],
            "change_percentage": result["change_percentage"],
            "num_changed_regions": len(result["regions"]), "regions": result["regions"],
        }


def load_change_detector(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    base_chs = tuple(ckpt.get("base_chs", (16, 32, 64, 128, 256)))
    model = UnifiedChangeDetectionNet(
        optical_in_ch=ckpt.get("optical_in_ch", 13),
        sar_in_ch=ckpt.get("sar_in_ch", 2),
        base_chs=base_chs,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return ChangeDetector(model, device=device, temperatures=ckpt.get("temperatures"))
