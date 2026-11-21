"""Inspect sampler previews when the renderer exposes them."""

import json
import struct
import time

import cv2
import numpy as np
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

from .audit import norm, read_text


class PreviewMonitor:
    def __init__(self, base, token, folder):
        self.socket = None
        self.folder = folder
        self.step = 0
        self.maximum = 0
        self.last_step = -1
        self.reason = ""
        self.streak = 0
        self.active = False
        try:
            self.socket = connect(
                base.replace("https://", "wss://").replace("http://", "ws://")
                + "/ws?clientId="
                + token,
                open_timeout=3,
                close_timeout=1,
            )
        except (OSError, TimeoutError, WebSocketException):
            pass

    def close(self):
        if self.socket:
            self.socket.close()

    def inspect(self, client, pid, progress):
        if not self.socket:
            time.sleep(1)
            return
        try:
            data = self.socket.recv(timeout=1)
        except TimeoutError:
            return
        except Exception:
            self.close()
            self.socket = None
            return
        if isinstance(data, str):
            msg = json.loads(data)
            event = msg.get("data", {})
            if event.get("prompt_id") != pid:
                return
            self.active = True
            if msg["type"] == "progress":
                self.step, self.maximum = event["value"], event["max"]
                progress(f"Sampling · {self.step}/{self.maximum}")
            return
        if (
            not self.active
            or not self.maximum
            or self.step / self.maximum < 0.55
            or self.step == self.last_step
            or len(data) < 9
        ):
            return
        event, header = struct.unpack(">II", data[:8])
        if event not in (1, 4):
            return
        if event == 4:
            meta = json.loads(data[8 : 8 + header])
            if meta.get("prompt_id", pid) != pid:
                return
        raw = data[8 + header :] if event == 4 else data[8:]
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        self.last_step = self.step
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lines = read_text(img)
        reason = ""
        if gray.std() < 4 and (gray.mean() < 6 or gray.mean() > 249):
            reason = "uniform exposure"
        elif any(l["confidence"] > 0.92 and len(norm(l["text"])) >= 4 for l in lines):
            reason = "unplanned text"
        self.streak = (
            self.streak + 1 if reason and reason == self.reason else int(bool(reason))
        )
        self.reason = reason
        with (self.folder / "previews.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "step": self.step,
                        "max": self.maximum,
                        "ocr": lines,
                        "suspect": reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if self.streak >= 2:
            res = client.post("/api/jobs/" + pid + "/cancel")
            if res.is_success and res.json().get("cancelled"):
                raise ValueError("Sampler preview rejected: " + reason)
