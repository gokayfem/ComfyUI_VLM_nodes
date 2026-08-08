"""Probe and benchmark a real TensorRT vision path for Qwen3-VL.

The experiment deliberately compiles only the vision tower.  It reports
TensorRT graph coverage, numerical drift, isolated vision latency, and (when
conversion succeeds) can be extended to the unchanged language decoder.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch_tensorrt
from PIL import Image
from qwen3_vl_transformers import (
    aggregate,
    prepare_inputs,
    resize_to_longest_edge,
    run_sample,
)
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)


class StaticVisionTensorOutputs(torch.nn.Module):
    """Tensor-only vision tower with fixed-shape positional metadata.

    Transformers derives this metadata from ``grid_thw`` using Python integer
    conversions.  Hoisting it is both export-safe and valid for our explicitly
    static benchmark shape.
    """

    def __init__(self, visual: torch.nn.Module, grid_thw: torch.Tensor) -> None:
        super().__init__()
        self.visual = visual
        indices, weights = get_vision_bilinear_indices_and_weights(
            grid_thw,
            num_grid_per_side=visual.num_grid_per_side,
            spatial_merge_size=visual.config.spatial_merge_size,
            kwargs={},
        )
        position_ids = get_vision_position_ids(
            grid_thw, visual.spatial_merge_size, kwargs={}
        )
        cu_seqlens = get_vision_cu_seqlens(grid_thw, kwargs={})
        self.register_buffer("bilinear_indices", indices)
        self.register_buffer("bilinear_weights", weights)
        self.register_buffer("position_ids", position_ids)
        self.register_buffer("cu_seqlens", cu_seqlens)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden_states = self.visual.patch_embed(pixel_values)
        pos_embeds = (
            self.visual.pos_embed(self.bilinear_indices)
            * self.bilinear_weights[:, :, None]
        ).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary_pos_emb = self.visual.rotary_pos_emb(self.position_ids)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        embedding = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (embedding.cos(), embedding.sin())
        deepstack_features = []
        for layer_num, block in enumerate(self.visual.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.visual.deepstack_visual_indexes:
                merger_index = self.visual.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(
                    self.visual.deepstack_merger_list[merger_index](hidden_states)
                )
        return (
            hidden_states,
            self.visual.merger(hidden_states),
            *deepstack_features,
        )


class CompiledVisionAdapter(torch.nn.Module):
    """Restore the Transformers vision API around a compiled tensor graph."""

    def __init__(
        self,
        compiled: torch.nn.Module,
        *,
        dtype: torch.dtype,
        spatial_merge_size: int,
    ) -> None:
        super().__init__()
        self.compiled = compiled
        self._output_dtype = dtype
        self.spatial_merge_size = spatial_merge_size

    @property
    def dtype(self) -> torch.dtype:
        return self._output_dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
        return_dict: bool = True,
        **_: Any,
    ) -> BaseModelOutputWithDeepstackFeatures | tuple[torch.Tensor, ...]:
        del grid_thw
        outputs = self.compiled(pixel_values)
        if not return_dict:
            return outputs
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=outputs[0],
            pooler_output=outputs[1],
            deepstack_features=list(outputs[2:]),
        )


def timed_samples(
    module: torch.nn.Module,
    pixel_values: torch.Tensor,
    *,
    warmups: int,
    runs: int,
) -> tuple[tuple[torch.Tensor, ...], list[float]]:
    output: tuple[torch.Tensor, ...] | None = None
    samples: list[float] = []
    with torch.inference_mode():
        for index in range(warmups + runs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = module(pixel_values)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            if index >= warmups:
                samples.append(elapsed_ms)
    assert output is not None
    return output, samples


def tensor_errors(
    eager: tuple[torch.Tensor, ...], compiled: tuple[torch.Tensor, ...]
) -> list[dict[str, Any]]:
    errors = []
    for index, (reference, candidate) in enumerate(zip(eager, compiled, strict=True)):
        difference = (reference.float() - candidate.float()).abs()
        errors.append(
            {
                "output_index": index,
                "shape": list(reference.shape),
                "max_absolute_error": float(difference.max()),
                "mean_absolute_error": float(difference.mean()),
                "cosine_similarity": float(
                    torch.nn.functional.cosine_similarity(
                        reference.float().flatten(),
                        candidate.float().flatten(),
                        dim=0,
                    )
                ),
            }
        )
    return errors


def graph_coverage(module: torch.nn.Module) -> dict[str, Any]:
    graph = getattr(module, "graph", None)
    if graph is None:
        return {"available": False}
    nodes = list(graph.nodes)
    call_modules = [node for node in nodes if node.op == "call_module"]
    targets = [str(node.target) for node in call_modules]
    engine_targets = [target for target in targets if "run_on_acc" in target]
    fallback_targets = [target for target in targets if "run_on_gpu" in target]
    return {
        "available": True,
        "graph_nodes": len(nodes),
        "call_modules": targets,
        "tensorrt_engine_partitions": len(engine_targets),
        "pytorch_fallback_partitions": len(fallback_targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--prompt", default="Describe this image precisely in one sentence."
    )
    parser.add_argument("--longest-edge", type=int, default=448)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--generation-warmups", type=int, default=1)
    parser.add_argument("--generation-runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-block-size", type=int, default=5)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--require-full-compilation", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/qwen3-vl-2b-tensorrt-vision.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    image = resize_to_longest_edge(
        Image.open(args.image).convert("RGB"), args.longest_edge
    )
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    inputs = prepare_inputs(
        processor,
        image,
        args.prompt,
        min_pixels=None,
        max_pixels=None,
    )
    pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)
    grid_thw = inputs["image_grid_thw"].to("cuda")
    visual = StaticVisionTensorOutputs(model.model.visual, grid_thw).eval()

    eager_output, eager_ms = timed_samples(
        visual,
        pixel_values,
        warmups=args.warmups,
        runs=args.runs,
    )
    eager_generation = []
    for index in range(args.generation_warmups + args.generation_runs):
        sample = run_sample(
            model,
            processor,
            image,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            cache_implementation="static",
            min_pixels=None,
            max_pixels=None,
            disable_compile=False,
        )
        if index >= args.generation_warmups:
            eager_generation.append(sample)
    compile_started = time.perf_counter()
    compiled = torch_tensorrt.compile(
        visual,
        ir="dynamo",
        arg_inputs=(pixel_values,),
        enabled_precisions={torch.bfloat16},
        min_block_size=args.min_block_size,
        optimization_level=args.optimization_level,
        require_full_compilation=args.require_full_compilation,
        pass_through_build_failures=True,
        enable_experimental_decompositions=True,
        cache_built_engines=True,
        reuse_cached_engines=True,
        engine_cache_dir="benchmarks/results/tensorrt-engine-cache",
    )
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_started
    compiled_output, compiled_ms = timed_samples(
        compiled,
        pixel_values,
        warmups=args.warmups,
        runs=args.runs,
    )
    original_visual = model.model.visual
    model.model.visual = CompiledVisionAdapter(
        compiled,
        dtype=original_visual.dtype,
        spatial_merge_size=original_visual.spatial_merge_size,
    )
    tensorrt_generation = []
    for index in range(args.generation_warmups + args.generation_runs):
        sample = run_sample(
            model,
            processor,
            image,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            cache_implementation="static",
            min_pixels=None,
            max_pixels=None,
            disable_compile=False,
        )
        if index >= args.generation_warmups:
            tensorrt_generation.append(sample)

    eager_median = statistics.median(eager_ms)
    compiled_median = statistics.median(compiled_ms)
    artifact = {
        "schema": "comfyui-vlm/tensorrt-vision-probe",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "media": {
            "path": str(args.image.resolve()),
            "processed_size": list(image.size),
            "pixel_values_shape": list(pixel_values.shape),
            "image_grid_thw": grid_thw.cpu().tolist(),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "torch_tensorrt": torch_tensorrt.__version__,
            "tensorrt": __import__("tensorrt").__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(),
        },
        "configuration": {
            "precision": "bfloat16",
            "min_block_size": args.min_block_size,
            "optimization_level": args.optimization_level,
            "require_full_compilation": args.require_full_compilation,
            "warmups": args.warmups,
            "runs": args.runs,
            "generation_warmups": args.generation_warmups,
            "generation_runs": args.generation_runs,
        },
        "model_load_seconds": round(load_seconds, 3),
        "compile_seconds": round(compile_seconds, 3),
        "coverage": graph_coverage(compiled),
        "fidelity": tensor_errors(eager_output, compiled_output),
        "latency_ms": {
            "eager_samples": [round(value, 3) for value in eager_ms],
            "tensorrt_samples": [round(value, 3) for value in compiled_ms],
            "eager_median": round(eager_median, 3),
            "tensorrt_median": round(compiled_median, 3),
            "speedup": round(eager_median / compiled_median, 3),
        },
        "generation": {
            "eager": aggregate(eager_generation),
            "tensorrt": aggregate(tensorrt_generation),
            "exact_output_match": all(
                sample["output_sha256"] == eager_generation[0]["output_sha256"]
                for sample in tensorrt_generation
            ),
            "eager_output": eager_generation[0]["output"],
            "tensorrt_output": tensorrt_generation[0]["output"],
            "eager_samples": eager_generation,
            "tensorrt_samples": tensorrt_generation,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2), flush=True)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
