# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from tactile_ssl import algorithm
from tactile_ssl.data.vision_based_interactive import DemoForceFieldData

from .test_task import TestTaskSL


PATCH_SIZE = 16
GRID = 14
IMG_SIZE = 224


class DemoEncoderRerun(TestTaskSL):
    """Live-stream DIGIT/GelSight frames through the Sparsh SSL encoder and
    visualize its raw 768-D patch features in the Rerun viewer via:

      - PCA-to-RGB (DINOv2 style)
      - cosine similarity to a fixed query patch
      - last-block register-token -> patch attention

    The downstream force-field head is never invoked.
    """

    def __init__(
        self,
        digit_serial: Optional[str],
        gelsight_device_id: Optional[int],
        device,
        module: algorithm.Module,
        n_warmup_frames: int = 60,
        query_patch: Tuple[int, int] = (7, 7),
        refit_every: int = 0,
    ):
        super().__init__(device=device, module=module)
        self.digit_serial = digit_serial
        self.gelsight_device_id = gelsight_device_id
        self.n_warmup_frames = n_warmup_frames
        self.query_patch = tuple(query_patch)
        self.refit_every = refit_every

    def init(self):
        self.sensor_handler = DemoForceFieldData(
            config=self.config.data.dataset.config,
            digit_serial=self.digit_serial,
            gelsight_device_id=self.gelsight_device_id,
        )

        encoder = self.module.model_encoder
        self.num_register_tokens = getattr(encoder, "num_register_tokens", 1)

        self._qkv_cache = {}
        last_block_attn = encoder.blocks[-1].attn
        last_block_attn.qkv.register_forward_hook(self._make_qkv_hook())
        self._last_attn_ref = last_block_attn

        self._feat_buffer = deque(maxlen=self.n_warmup_frames)
        self._pca: Optional[PCA] = None
        self._frame_idx = 0

        rr.init("sparsh_encoder", spawn=True)
        rr.log(
            "world/tactile",
            rr.TextDocument(
                f"Sparsh encoder features | layer=last | grid={GRID}x{GRID} | "
                f"warmup={self.n_warmup_frames} frames | query={self.query_patch}"
            ),
            static=True,
        )

    def _make_qkv_hook(self):
        def hook(_module, _inputs, output):
            self._qkv_cache["qkv"] = output
        return hook

    @torch.no_grad()
    def _forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Returns post-norm patch tokens, shape [N_patches, D]."""
        patch_tokens = self.module.model_encoder(x)
        return patch_tokens[0]

    def _compute_attention_map(self) -> np.ndarray:
        qkv = self._qkv_cache["qkv"]
        B, N, _ = qkv.shape
        H = self._last_attn_ref.num_heads
        Dh = qkv.shape[-1] // (3 * H)
        qkv = qkv.reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        attn = (q @ k.transpose(-2, -1)) * (Dh ** -0.5)
        attn = attn.softmax(dim=-1).mean(dim=1)
        reg_to_patch = attn[0, 0, self.num_register_tokens:]
        heatmap = reg_to_patch.reshape(GRID, GRID).detach().cpu().numpy()
        return heatmap

    def _fit_pca(self):
        stacked = torch.stack(list(self._feat_buffer), dim=0)
        stacked = stacked.reshape(-1, stacked.shape[-1]).detach().cpu().numpy()
        self._pca = PCA(n_components=3)
        self._pca.fit(stacked)

    def _pca_rgb(self, feats: torch.Tensor) -> np.ndarray:
        feats_np = feats.detach().cpu().numpy()
        proj = self._pca.transform(feats_np)
        lo = proj.min(axis=0, keepdims=True)
        hi = proj.max(axis=0, keepdims=True)
        rng = np.where(hi - lo < 1e-8, 1.0, hi - lo)
        proj = (proj - lo) / rng
        rgb = (proj.reshape(GRID, GRID, 3) * 255).astype(np.uint8)
        return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    def _similarity_map(self, feats: torch.Tensor) -> Tuple[np.ndarray, float]:
        r, c = self.query_patch
        q_idx = r * GRID + c
        q = feats[q_idx:q_idx + 1]
        sim = F.cosine_similarity(feats, q, dim=-1)
        sim_grid = sim.reshape(GRID, GRID).detach().cpu().numpy()
        sim_vis = np.clip((sim_grid + 1.0) * 0.5, 0.0, 1.0)
        sim_vis = cv2.resize(sim_vis, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        sim_vis = cv2.applyColorMap((sim_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
        sim_vis = cv2.cvtColor(sim_vis, cv2.COLOR_BGR2RGB)
        return sim_vis, float(sim_grid.max())

    def _attention_rgb(self, attn_map: np.ndarray) -> np.ndarray:
        lo, hi = attn_map.min(), attn_map.max()
        norm = (attn_map - lo) / (hi - lo + 1e-8)
        up = cv2.resize(norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.applyColorMap((up * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    def _log_frame(self, img_color: np.ndarray, feats: torch.Tensor):
        rr.set_time_sequence("frame", self._frame_idx)

        img_color = cv2.resize(img_color, (IMG_SIZE, IMG_SIZE))
        rr.log("world/tactile/image", rr.Image(img_color))

        r, c = self.query_patch
        query_xy = np.array([[c * PATCH_SIZE + PATCH_SIZE // 2,
                              r * PATCH_SIZE + PATCH_SIZE // 2]], dtype=np.float32)
        rr.log(
            "world/tactile/image/query",
            rr.Points2D(query_xy, colors=[(255, 0, 0)], radii=[4.0]),
        )

        rr.log("plots/feat_norm", rr.Scalars(float(feats.norm(dim=-1).mean().item())))

        if self._pca is None:
            return

        rr.log("world/tactile/pca_rgb", rr.Image(self._pca_rgb(feats)))

        sim_rgb, sim_max = self._similarity_map(feats)
        rr.log("world/tactile/sim", rr.Image(sim_rgb))
        rr.log("plots/sim_max", rr.Scalars(sim_max))

        attn_map = self._compute_attention_map()
        rr.log("world/tactile/attention", rr.Image(self._attention_rgb(attn_map)))

    def run_model(self):
        while True:
            try:
                sample = self.sensor_handler.get_model_inputs()
            except Exception as e:
                print(f"[DemoEncoderRerun] sensor read failed: {e}")
                break

            x = sample["image"].unsqueeze(0).to(self.device)
            feats = self._forward_encoder(x)

            self._feat_buffer.append(feats.detach().cpu())
            if self._pca is None and len(self._feat_buffer) >= self.n_warmup_frames:
                print(f"[DemoEncoderRerun] fitting PCA on {self.n_warmup_frames} warmup frames")
                self._fit_pca()
            elif (
                self.refit_every > 0
                and self._pca is not None
                and self._frame_idx > 0
                and self._frame_idx % self.refit_every == 0
            ):
                self._fit_pca()

            self._log_frame(sample["current_image_color"], feats)
            self._frame_idx += 1
