#!/usr/bin/env python3
"""
实时语音转写 WebSocket 后端服务（Whisper 版）
基于 OpenAI Whisper + Silero VAD，实现与前端兼容的 WebSocket 协议

与 SenseVoice 版相比：精度更高，但延迟略高（3-5秒）
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Windows 控制台 UTF-8
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

try:
    import whisper
except ImportError:
    print("错误: 请先安装 openai-whisper: pip3 install openai-whisper", file=sys.stderr)
    sys.exit(1)

try:
    import sherpa_onnx  # 仅用于 VAD
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    print("警告: sherpa-onnx 未安装，VAD 不可用（无 VAD 模式）", file=sys.stderr, flush=True)

try:
    try:
        from websockets.asyncio.server import serve
    except ImportError:
        from websockets.server import serve
except ImportError:
    print("错误: 请先安装 websockets: pip3 install websockets", file=sys.stderr)
    sys.exit(1)

import tempfile
import os
from collections import deque


class WhisperTranscriber:
    """Whisper 模型封装，支持 numpy 数组输入（避免文件 I/O）"""

    def __init__(self, model_name: str = "turbo", device: str = "cpu"):
        print(f"加载 Whisper 模型: {model_name} (device={device})...", file=sys.stderr,
              flush=True)
        self.model = whisper.load_model(model_name)
        self.model_name = model_name
        print(f"Whisper 模型就绪: {model_name}", file=sys.stderr, flush=True)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   language: str | None = None) -> str:
        """转写音频 numpy 数组，返回纯文本"""
        # Whisper 期望 float32 且归一化到 [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        opts: dict = {"fp16": False}
        if language and language != "auto":
            opts["language"] = language

        result = self.model.transcribe(audio, **opts)
        return result["text"].strip()


class TranscriptionServer:
    def __init__(
        self,
        model_name: str = "turbo",
        device: str = "cpu",
        model_dir: str = "",
        vad_threshold: float = 0.5,
        vad_min_silence: float = 1.0,
        vad_min_speech: float = 0.25,
        partial_interval_sec: float = 2.0,
    ):
        self.sample_rate = 16000
        self.clients: set = set()
        self._recording_start_time: dict = {}
        self.vad_threshold = vad_threshold
        self.vad_min_silence = vad_min_silence
        self.vad_min_speech = vad_min_speech
        self.partial_interval_sec = partial_interval_sec

        # 加载 Whisper 模型
        self.transcriber = WhisperTranscriber(model_name=model_name, device=device)

        # 加载 VAD 模型（复用 SenseVoice 版的 Silero VAD）
        vad_model_dir = Path(model_dir) if model_dir else Path(__file__).parent
        vad_model_path = vad_model_dir / "silero_vad.onnx"

        if not vad_model_path.exists():
            print(f"VAD 模型不存在 {vad_model_path}，无 VAD 模式",
                  file=sys.stderr, flush=True)
            self.vad_enabled = False
        elif not SHERPA_AVAILABLE:
            print("sherpa-onnx 未安装，无 VAD 模式", file=sys.stderr, flush=True)
            self.vad_enabled = False
        else:
            self.vad_enabled = True
            vad_config = sherpa_onnx.VadModelConfig()
            vad_config.silero_vad.model = str(vad_model_path)
            vad_config.silero_vad.threshold = vad_threshold
            vad_config.silero_vad.min_silence_duration = vad_min_silence
            vad_config.silero_vad.min_speech_duration = vad_min_speech
            vad_config.sample_rate = self.sample_rate
            self._vad_config = vad_config
            print(f"VAD 模型就绪: {vad_model_path}", file=sys.stderr, flush=True)

    def _create_vad(self):
        """创建 VAD 实例"""
        if not self.vad_enabled:
            return None
        return sherpa_onnx.VoiceActivityDetector(
            self._vad_config, buffer_size_in_seconds=30
        )

    @staticmethod
    def _guess_language(text: str) -> str:
        """从文本字符推断语言"""
        import re
        if not text:
            return "zh"
        han = len(re.findall(r"[\u3400-\u9fff]", text))
        latin = len(re.findall(r"[A-Za-z]", text))
        if han >= 2:
            return "zh" if latin < max(12, int(han * 2.5)) else "en"
        if han > 0:
            return "zh"
        if latin >= 3:
            return "en"
        return "zh"

    async def handle_client(self, websocket):
        self.clients.add(websocket)
        client_id = id(websocket)
        self._recording_start_time[client_id] = time.time()

        # 音频缓冲区
        audio_buffer = np.array([], dtype=np.float32)
        last_partial_text = ""
        last_partial_at = 0.0
        partial_min_samples = int(self.sample_rate * 2.0)
        partial_max_samples = int(self.sample_rate * 30.0)

        # VAD
        client_vad = self._create_vad()

        print(f"客户端连接 (Whisper): {websocket.remote_address}", flush=True)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Int16 PCM -> float32 [-1, 1]
                    samples = (
                        np.frombuffer(message, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    audio_buffer = np.concatenate((audio_buffer, samples))
                    if len(audio_buffer) > partial_max_samples:
                        audio_buffer = audio_buffer[-partial_max_samples:]

                    now = time.time()

                    # ── Partial 快照 ──
                    if (
                        len(audio_buffer) >= partial_min_samples
                        and (now - last_partial_at) >= self.partial_interval_sec
                    ):
                        try:
                            partial_raw = self.transcriber.transcribe(
                                audio_buffer, self.sample_rate
                            )
                            if partial_raw and partial_raw != last_partial_text:
                                lang = self._guess_language(partial_raw)
                                resp = {
                                    "type": "partial",
                                    "text": partial_raw,
                                    "language": lang,
                                    "timestamps": {
                                        "start": 0,
                                        "duration": round(len(audio_buffer) / self.sample_rate, 2),
                                    },
                                }
                                await websocket.send(json.dumps(resp, ensure_ascii=False))
                                last_partial_text = partial_raw
                        except Exception as e:
                            print(f"partial 转写错误: {e}", file=sys.stderr, flush=True)
                        last_partial_at = now

                    # ── VAD Final ──
                    if client_vad is not None:
                        client_vad.accept_waveform(samples)
                        while not client_vad.empty():
                            speech = client_vad.front
                            try:
                                text = self.transcriber.transcribe(
                                    speech.samples, self.sample_rate
                                )
                                if text:
                                    lang = self._guess_language(text)
                                    elapsed = speech.start / self.sample_rate
                                    duration = len(speech.samples) / self.sample_rate
                                    resp = {
                                        "type": "final",
                                        "text": text,
                                        "language": lang,
                                        "timestamps": {
                                            "start": round(elapsed, 2),
                                            "duration": round(duration, 2),
                                        },
                                    }
                                    await websocket.send(json.dumps(resp, ensure_ascii=False))
                            except Exception as e:
                                print(f"final 转写错误: {e}", file=sys.stderr, flush=True)
                            client_vad.pop()

                elif isinstance(message, str):
                    try:
                        cmd = json.loads(message)
                        if cmd.get("type") == "ping":
                            await websocket.send(json.dumps({"type": "pong"}))
                        elif cmd.get("type") == "reset":
                            client_vad = self._create_vad()
                            self._recording_start_time[client_id] = time.time()
                            audio_buffer = np.array([], dtype=np.float32)
                            last_partial_text = ""
                            last_partial_at = 0.0
                        elif cmd.get("type") == "flush_partial":
                            audio_buffer = np.array([], dtype=np.float32)
                            last_partial_text = ""
                            last_partial_at = 0.0
                    except json.JSONDecodeError:
                        pass

        except Exception:
            pass
        finally:
            self.clients.discard(websocket)
            self._recording_start_time.pop(client_id, None)
            print(f"客户端断开 (Whisper): {websocket.remote_address}", flush=True)


async def main():
    parser = argparse.ArgumentParser(description="实时语音转写 WebSocket 后端 (Whisper)")
    parser.add_argument("--model-name", default="turbo",
                        help="Whisper 模型名称 (tiny/base/small/medium/large/turbo)")
    parser.add_argument("--model-dir", default="",
                        help="VAD 模型目录（silero_vad.onnx 所在）")
    parser.add_argument("--port", type=int, default=18889,
                        help="WebSocket 端口 (默认: 18889)")
    parser.add_argument("--device", default="cpu", help="推理设备 (cpu/cuda)")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-silence", type=float, default=1.0)
    parser.add_argument("--vad-min-speech", type=float, default=0.25)
    parser.add_argument("--partial-interval", type=float, default=2.0,
                        help="partial 快照间隔(秒)")
    # 以下参数为兼容 SenseVoice 版，忽略
    parser.add_argument("--use-int8", action="store_true")
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--idle-timeout", type=int, default=0)
    parser.add_argument("--partial-profile", default="stable")
    parser.add_argument("--recognition-mode", default="zh-en")
    args = parser.parse_args()

    server = TranscriptionServer(
        model_name=args.model_name,
        device=args.device,
        model_dir=args.model_dir,
        vad_threshold=args.vad_threshold,
        vad_min_silence=args.vad_min_silence,
        vad_min_speech=args.vad_min_speech,
        partial_interval_sec=args.partial_interval,
    )

    stop_event = asyncio.Event()

    def handle_signal():
        print("\n收到终止信号，正在关闭...", flush=True)
        stop_event.set()

    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, handle_signal)
    else:
        signal.signal(signal.SIGINT, lambda *_: handle_signal())
        signal.signal(signal.SIGTERM, lambda *_: handle_signal())

    async with serve(
        server.handle_client,
        "127.0.0.1",
        args.port,
        max_size=2**20,
    ) as ws_server:
        print(f"Server started on ws://127.0.0.1:{args.port}", flush=True)
        await stop_event.wait()

    print("服务已关闭", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
