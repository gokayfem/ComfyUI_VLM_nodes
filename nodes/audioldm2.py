"""Lazy AudioLDM2 generation with legacy and standard ComfyUI AUDIO outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import folder_paths

from .runtime import (
    CachedModelNode,
    execution_device,
    require_module,
    reserve_external_vram,
    snapshot_download,
    torch_dtype,
)


class AnyType(str):
    def __ne__(self, other):
        return False


ANY = AnyType("*")


class AudioLDM2Predictor:
    def __init__(self, cpu_offload=True):
        diffusers = require_module("diffusers")
        path = snapshot_download(
            "cvssp/audioldm2",
            "audioldm2",
            ignore_patterns=["*.bin", "*.jpg", "*.png"],
        )
        self.device = execution_device()
        dtype = torch_dtype("float16", self.device)
        if self.device.type != "cpu":
            reserve_external_vram(8 * 1024**3)
        self.pipeline = diffusers.AudioLDM2Pipeline.from_pretrained(
            path, torch_dtype=dtype
        )
        # Accelerate's model CPU offload is currently reliable on the CUDA API,
        # which covers both NVIDIA CUDA and AMD ROCm PyTorch builds.
        if self.device.type == "cuda" and cpu_offload:
            require_module("accelerate")
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(self.device)

    def close(self):
        self.pipeline = None
        import gc

        gc.collect()
        try:
            import comfy.model_management as model_management

            model_management.soft_empty_cache()
        except Exception:
            pass

    def generate(self, text, negative, duration, guidance, seed, count, steps):
        # MPS generators are not supported by every PyTorch/Diffusers pairing.
        # A CPU generator remains deterministic and works with every pipeline.
        generator_device = (
            self.device if self.device.type in {"cuda", "xpu"} else "cpu"
        )
        generator = torch.Generator(device=generator_device).manual_seed(
            int(seed)
        )
        audios = self.pipeline(
            text,
            negative_prompt=negative or None,
            audio_length_in_s=float(duration),
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            num_waveforms_per_prompt=int(count),
            generator=generator,
        ).audios
        array = np.asarray(audios, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        native_rate = int(
            getattr(
                getattr(getattr(self.pipeline, "vae", None), "config", None),
                "sampling_rate",
                16000,
            )
        )
        return array, native_rate


class AudioLDM2Node(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "duration": (
                    "INT",
                    {"default": 10, "min": 1, "max": 60},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.1, "max": 20.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": 42, "min": 0}),
                "n_candidates": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10},
                ),
                "sample_rate": (
                    "INT",
                    {"default": 16000, "min": 8000, "max": 48000},
                ),
                "extension": (["wav", "flac"],),
            },
            "optional": {
                "steps": ("INT", {"default": 100, "min": 10, "max": 500}),
                "cpu_offload": ("BOOLEAN", {"default": True}),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_NAMES = ("wave_form", "sample_rate", "audio")
    RETURN_TYPES = (ANY, "INT", "AUDIO")
    OUTPUT_NODE = True
    FUNCTION = "generate_audio_final"
    CATEGORY = "VLM Nodes/Audio"

    def generate_audio_final(
        self,
        text,
        negative_prompt,
        duration,
        guidance_scale,
        sample_rate,
        seed,
        n_candidates,
        extension,
        steps=100,
        cpu_offload=True,
        unload_after=False,
    ):
        del extension
        predictor = self.get_or_create_model(
            ("audioldm2", bool(cpu_offload)),
            lambda: AudioLDM2Predictor(cpu_offload),
        )
        try:
            waveforms, native_rate = predictor.generate(
                text,
                negative_prompt,
                duration,
                guidance_scale,
                seed,
                n_candidates,
                steps,
            )
            if int(sample_rate) != native_rate:
                samples = torch.from_numpy(waveforms).unsqueeze(1)
                target_length = round(
                    samples.shape[-1] * int(sample_rate) / native_rate
                )
                waveforms = (
                    torch.nn.functional.interpolate(
                        samples,
                        size=target_length,
                        mode="linear",
                        align_corners=False,
                    )
                    .squeeze(1)
                    .numpy()
                )
            # Standard Comfy AUDIO is [batch, channels, samples].
            audio = {
                "waveform": torch.from_numpy(waveforms).unsqueeze(1),
                "sample_rate": int(sample_rate),
            }
            return (waveforms[0].tolist(), int(sample_rate), audio)
        finally:
            self.maybe_clear_model(unload_after)


class SaveAudioNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "waveforms": (ANY,),
                "sample_rate": ("INT",),
                "extension": (["wav", "flac"],),
                "filename": ("STRING", {"default": "audio"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    CATEGORY = "VLM Nodes/Audio"
    OUTPUT_NODE = True

    def save_audio(self, waveforms, sample_rate, extension, filename):
        soundfile = require_module("soundfile")
        safe_name = Path(filename).name.strip() or "audio"
        output = Path(folder_paths.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        base = output / safe_name
        path = base.with_suffix(f".{extension}")
        counter = 2
        while path.exists():
            path = output / f"{safe_name}_{counter:05d}.{extension}"
            counter += 1
        soundfile.write(path, np.asarray(waveforms), int(sample_rate))
        return ()


NODE_CLASS_MAPPINGS = {
    "AudioLDM2Node": AudioLDM2Node,
    "SaveAudioNode": SaveAudioNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioLDM2Node": "AudioLDM2",
    "SaveAudioNode": "Save Audio",
}
