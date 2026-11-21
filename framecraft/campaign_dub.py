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


class Dubber:
    def __init__(self, weights, source, device="cuda"):
        sys.path.insert(0, str(source))
        from musetalk.models.unet import PositionalEncoding

        stream_weights = device == "cuda-stream"
        self.device = "cuda" if stream_weights else device
        weight_device = "cpu" if stream_weights else self.device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.yunet = cv2.FaceDetectorYN.create(
            str(weights / "yunet.onnx"), "", (320, 320), 0.67
        )
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

    def landmarks(self, img):
        h, w = img.shape[:2]
        self.yunet.setInputSize((w, h))
        _, faces = self.yunet.detect(img)
        if faces is None or len(faces) != 1:
            return None
        x, y, fw, fh = faces[0, :4]
        eyes = faces[0, 4:8].reshape(2, 2)
        nose = faces[0, 8:10]
        mouth = faces[0, 10:14].reshape(2, 2).mean(axis=0)
        center = 0.63 * nose[1] + 0.37 * eyes[:, 1].mean()
        chin = y + fh
        return np.array(
            [
                max(0, x - 0.025 * fw),
                max(0, 2 * center - chin),
                min(w, x + fw * 1.025),
                min(h, chin + 0.05 * fh),
                *mouth,
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
        boxes = []
        try:
            for _ in range(count):
                ok, im = cap.read()
                if not ok:
                    raise ValueError(
                        "Presenter source cannot cover the requested continuous interval"
                    )
                frames.append(im)
                boxes.append(self.landmarks(im))
        finally:
            cap.release()
        good = [i for i, b in enumerate(boxes) if b is not None]
        if len(good) < count * 0.82:
            raise ValueError(f"Face visible in only {len(good)}/{count} frames")
        coords = np.stack(
            [
                np.interp(np.arange(count), good, [boxes[i][k] for i in good])
                for k in range(6)
            ],
            axis=1,
        )
        smooth = np.stack(
            [
                np.median(coords[max(0, i - 2) : min(count, i + 3)], axis=0)
                for i in range(count)
            ]
        )
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
        envelope = np.ones(count, dtype=np.float32)
        if speech_only:
            loud = np.flatnonzero(np.abs(signal) > 0.012)
            if len(loud) < 100:
                raise ValueError("A talking take needs audible speech")
            onset, end = (loud[[0, -1]] / 16000) * fps
            positions = np.arange(count)
            envelope = np.clip(
                np.minimum(positions - onset + 3, end - positions + 3) / 2, 0, 1
            )
        h, w = frames[0].shape[:2]
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
                if not np.any(envelope[offset : offset + batch]):
                    for original in frames[offset : offset + batch]:
                        proc.stdin.write(original.tobytes())
                    continue
                crops = []
                regions = []
                for i in range(offset, min(count, offset + batch)):
                    x, y, r, b, mx, my = smooth[i].round().astype(int)
                    crops.append(
                        cv2.resize(
                            frames[i][y:b, x:r],
                            (256, 256),
                            interpolation=cv2.INTER_LANCZOS4,
                        )[:, :, ::-1]
                    )
                    regions.append((x, y, r, b, mx, my))
                rgb = (
                    torch.from_numpy(np.stack(crops))
                    .permute(0, 3, 1, 2)
                    .to(self.device, self.dtype)
                    / 127.5
                    - 1
                )
                masked = rgb.clone()
                masked[:, :, 128:, :] = -1.0
                z = (
                    self.vae.encode(torch.cat([masked, rgb])).latent_dist.mode()
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
                for k, (x, y, r, b, mx, my) in enumerate(regions):
                    original = frames[offset + k]
                    surface = original[y:b, x:r]
                    patch = cv2.resize(
                        images[k][:, :, ::-1],
                        (r - x, b - y),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    hh, ww = surface.shape[:2]
                    mask = np.zeros((hh, ww), np.float32)
                    cv2.ellipse(
                        mask,
                        (mx - x, my - y + round(hh * 0.024)),
                        (max(2, round(ww * 0.30)), max(2, round(hh * 0.17))),
                        0,
                        0,
                        360,
                        1.0,
                        -1,
                    )
                    mask = cv2.GaussianBlur(mask, (0, 0), max(1, ww * 0.034))[
                        :, :, None
                    ]
                    mask *= envelope[offset + k]
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
            "face_detections": len(good),
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
