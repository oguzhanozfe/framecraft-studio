"""Dub a locally generated presenter's moving face with the campaign's actual narration."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import WhisperFeatureExtractor, WhisperModel

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framecraft.campaign_face import (
    face_boxes,
    lip_mask,
    lower_mask,
    rest_crop,
    resting_references,
    speech_activity,
    track,
)


class Dubber:
    cache_version = "registered-jaw-v3"

    def __init__(self, weights, source, device="cuda"):
        sys.path.insert(0, str(source))
        from musetalk.models.unet import PositionalEncoding

        stream_weights = device == "cuda-stream"
        self.device = "cuda" if stream_weights else device
        weight_device = "cpu" if stream_weights else self.device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        with torch.device("meta"):
            self.net = UNet2DConditionModel.from_config(
                json.loads((weights / "musetalkV15/musetalk.json").read_text())
            )
        state = torch.load(
            weights / "musetalkV15/unet.pth",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        self.net.load_state_dict(state, assign=True)
        self.net.to(device=weight_device, dtype=self.dtype).eval()
        self.vae = (
            AutoencoderKL.from_pretrained(
                str(weights / "sd-vae"), torch_dtype=self.dtype
            )
            .to(weight_device)
            .eval()
        )
        self.pe = (
            PositionalEncoding(d_model=384)
            .to(device=self.device, dtype=self.dtype)
            .eval()
        )
        self.whisper = (
            WhisperModel.from_pretrained(
                str(weights / "whisper"), torch_dtype=self.dtype
            )
            .to(weight_device)
            .eval()
        )
        self.encoder = WhisperFeatureExtractor.from_pretrained(str(weights / "whisper"))
        if stream_weights:
            from accelerate import cpu_offload

            for module in (self.vae, self.net, self.whisper.encoder):
                cpu_offload(module, torch.device(self.device), offload_buffers=True)

    def prompts(self, audio, frames, fps):
        feature = self.encoder(
            audio, return_tensors="pt", sampling_rate=16000
        ).input_features.to(self.device, self.dtype)
        hidden = torch.stack(
            self.whisper.encoder(feature, output_hidden_states=True).hidden_states,
            dim=2,
        )
        hidden = hidden[:, : math.ceil(len(audio) / 320)]
        padding = math.ceil(50 / fps) * 2
        padded = torch.cat(
            [
                torch.zeros_like(hidden[:, :padding]),
                hidden,
                torch.zeros_like(hidden[:, : padding * 3]),
            ],
            dim=1,
        )[0]
        return torch.stack(
            [
                padded[int(i * 50 / fps) : int(i * 50 / fps) + 10].reshape(-1, 384)
                for i in range(frames)
            ]
        )

    @torch.inference_mode()
    def run(
        self,
        src,
        voice,
        out,
        start,
        at,
        length,
        batch=6,
        voice_delay_ms=120,
        speech_only=False,
    ):
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = math.ceil(length * fps - 1e-6)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        frames = []
        try:
            for _ in range(count):
                ok, im = cap.read()
                if not ok:
                    raise ValueError(
                        "Presenter source cannot cover the requested continuous interval"
                    )
                frames.append(im)
        finally:
            cap.release()
        h, w = frames[0].shape[:2]
        mesh = track(src, start, count)
        boxes = face_boxes(mesh, w, h)
        references = resting_references(mesh)
        # Use the exact audio interval that will reach the final edit, including the initial music lead-in.
        raw = subprocess.check_output(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(voice),
                "-af",
                (
                    f"adelay={voice_delay_ms},"
                    if voice_delay_ms >= 0
                    else f"atrim=start={-voice_delay_ms / 1000},asetpts=PTS-STARTPTS,"
                )
                + f"apad,atrim=start={at}:duration={length + 0.1}",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "f32le",
                "pipe:1",
            ]
        )
        signal = np.frombuffer(raw, dtype="<f4").copy()
        features = self.prompts(signal, count, fps)
        envelope = speech_activity(signal, count, fps)
        if speech_only and not np.any(envelope):
            raise ValueError("A talking take needs audible speech")
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "3",
                "-preset",
                "veryfast",
                "-crf",
                "15",
                "-pix_fmt",
                "yuv420p",
                str(out),
            ],
            stdin=subprocess.PIPE,
        )
        deltas = []
        try:
            for offset in range(0, count, batch):
                crops, conditioning, reference_surfaces, masks = [], [], [], []
                for i in range(offset, min(count, offset + batch)):
                    x, y, r, b = boxes[i]
                    original = frames[i][y:b, x:r]
                    mask = lower_mask(mesh[i], boxes[i])
                    rest = rest_crop(frames, mesh, i, references[i], boxes[i])
                    local_mask = lip_mask(mesh[i], boxes[i])
                    neutral = np.rint(
                        rest * local_mask + original * (1 - local_mask)
                    ).astype("uint8")
                    crops.append(
                        cv2.resize(
                            original, (256, 256), interpolation=cv2.INTER_LANCZOS4
                        )[:, :, ::-1]
                    )
                    conditioning.append(
                        cv2.resize(
                            neutral, (256, 256), interpolation=cv2.INTER_LANCZOS4
                        )[:, :, ::-1]
                    )
                    masks.append(mask)
                    reference_surfaces.append(neutral)

                def tensor(images):
                    return (
                        torch.from_numpy(np.stack(images))
                        .permute(0, 3, 1, 2)
                        .to(self.device, self.dtype)
                        / 127.5
                        - 1
                    )

                rgb = tensor(crops)
                reference_rgb = tensor(conditioning)
                masked = rgb.clone()
                masked[:, :, 128:, :] = -1.0
                z = (
                    self.vae.encode(
                        torch.cat([masked, reference_rgb])
                    ).latent_dist.mode()
                    * self.vae.config.scaling_factor
                )
                za, zb = z.chunk(2)
                latent = self.net(
                    torch.cat([za, zb], dim=1),
                    torch.tensor([0], device=self.device),
                    encoder_hidden_states=self.pe(
                        features[offset : offset + len(crops)]
                    ),
                ).sample
                recon = self.vae.decode(latent / self.vae.config.scaling_factor).sample
                images = (
                    ((recon / 2 + 0.5).clamp(0, 1) * 255)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .float()
                    .numpy()
                    .astype("uint8")
                )
                for k, (x, y, r, b) in enumerate(boxes[offset : offset + len(crops)]):
                    original = frames[offset + k]
                    surface = original[y:b, x:r]
                    patch = cv2.resize(
                        images[k][:, :, ::-1],
                        (r - x, b - y),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    mask = masks[k]
                    # The trained model drives the jaw while speaking. In extended
                    # silence use the same presenter's registered listening expression.
                    weight = envelope[offset + k]
                    if references[offset + k] >= 0 and weight < 1.0:
                        patch = patch * weight + reference_surfaces[k] * (1.0 - weight)
                    result = original.copy()
                    result[y:b, x:r] = np.rint(
                        patch * mask + surface * (1 - mask)
                    ).astype("uint8")
                    deltas.append(
                        round(
                            float(
                                (np.abs(patch.astype("f4") - surface) * mask).sum()
                                / max(1, 3 * mask.sum())
                            ),
                            2,
                        )
                    )
                    proc.stdin.write(result.tobytes())
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("Dub encode failed")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        return {
            "model": "MuseTalk 1.5",
            "native_fps": fps,
            "frames": count,
            "face_frames": len(mesh),
            "pipeline": self.cache_version,
            "rest_reference_frames": int((references >= 0).sum()),
            "audio_start": at,
            "source_start": start,
            "voice_delay_ms": voice_delay_ms,
            "speech_only": speech_only,
            "mouth_pixel_change_mean": round(float(np.mean(deltas)), 2),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "weights", "source"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.plan.parent
    spec = json.loads(args.plan.read_text(encoding="utf8"))
    dub = Dubber(args.weights, args.source, args.device)
    for clip in spec["clips"]:
        output = root / clip["output"]
        report = dub.run(
            root / clip["file"],
            root / spec["voice"],
            output,
            clip.get("start", 0),
            clip["timeline"],
            clip["seconds"],
            batch=spec.get("batch", 6),
            voice_delay_ms=spec.get("voice_delay_ms", 120),
            speech_only=spec.get("speech_only", False),
        )
        output.with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf8"
        )
        print(output, report, flush=True)
