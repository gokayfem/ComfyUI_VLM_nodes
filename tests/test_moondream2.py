from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

PACKAGE = __package__.split(".")[0] if __package__ else "ComfyUI_VLM_nodes"
module = importlib.import_module(f"{PACKAGE}.nodes.moondream2")


def test_native_checkpoint_loader_bypasses_transformers_from_pretrained(
    tmp_path: Path,
    monkeypatch,
):
    package = ModuleType(module._CHECKPOINT_PACKAGE)
    package.__path__ = [str(tmp_path.resolve())]
    package.__package__ = module._CHECKPOINT_PACKAGE
    checkpoint = ModuleType(f"{module._CHECKPOINT_PACKAGE}.hf_moondream")
    calls = {}

    class FakeConfig:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["config"] = (Path(model_path), kwargs)
            return cls()

    class FakeModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            calls["model_config"] = config

    checkpoint.HfConfig = FakeConfig
    checkpoint.HfMoondream = FakeModel
    monkeypatch.setitem(sys.modules, module._CHECKPOINT_PACKAGE, package)
    monkeypatch.setitem(
        sys.modules,
        f"{module._CHECKPOINT_PACKAGE}.hf_moondream",
        checkpoint,
    )

    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"test")

    def load_model(model, filename, *, strict):
        calls["weights"] = (model, Path(filename), strict)
        model.weight.data.fill_(1)
        return set(), []

    monkeypatch.setattr(
        module,
        "require_module",
        lambda name: (
            SimpleNamespace(load_model=load_model)
            if name == "safetensors.torch"
            else None
        ),
    )

    model = module._load_native_checkpoint(tmp_path)

    assert isinstance(model, FakeModel)
    assert not model.training
    assert model.weight.item() == 1
    assert calls["config"] == (tmp_path, {"local_files_only": True})
    assert calls["weights"] == (model, weights, True)


def test_photon_requirements_pin_cuda_runtime_with_required_symbol():
    requirements = (
        Path(module.__file__).resolve().parents[1] / "requirements-moondream31.txt"
    ).read_text(encoding="utf-8")
    assert "kestrel-kernels==0.4.6" in requirements
    assert "nvidia-cuda-runtime-cu12==12.9.79" in requirements
