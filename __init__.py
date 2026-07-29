import importlib
import logging

from .nodes.runtime import register_model_folder

LOGGER = logging.getLogger("ComfyUI_VLM_nodes")
register_model_folder()

node_list = [
    "acceleration",
    "audioldm2",
    "diagnostics",
    "florence2",
    "grounding",
    "hosted_api",
    "joytag",
    "kosmos2",
    "llavaloader",
    "mcllava",
    "minicpm",
    "modern_vlm",
    "molmo",
    "moondream31",
    "moondream2",
    "moondream_script",
    "paligemma",
    "playmusic",
    "qwen2vl",
    "sam2",
    "sam3_adapter",
    "simpletext",
    "spatial_parser",
    "suggest",
    "tracking",
    "uform",
    "video_intelligence",
    "vision_utils",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
IMPORT_ERRORS = {}

for module_name in node_list:
    try:
        imported_module = importlib.import_module(f".nodes.{module_name}", __name__)
    except Exception as exc:
        # A broken optional model must never prevent unrelated nodes from loading.
        IMPORT_ERRORS[module_name] = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Could not load optional node module %s", module_name)
        continue
    NODE_CLASS_MAPPINGS.update(
        getattr(imported_module, "NODE_CLASS_MAPPINGS", {})
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(imported_module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )


WEB_DIRECTORY = "./web"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
