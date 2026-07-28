"""Backward-compatible MoonDream node powered by the current Moondream 2."""

from .moondream2 import MODEL_ID, MODEL_REVISION, Moondream2Predictor
from .runtime import CachedModelNode


class MoonDream(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe this image in detail.",
                    },
                ),
            },
            "optional": {
                "unload_after": ("BOOLEAN", {"default": False})
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "answer_questions"
    CATEGORY = "VLM Nodes/MoonDream"

    def answer_questions(self, image, question, unload_after=False):
        predictor = self.get_or_create_model(
            (MODEL_ID, MODEL_REVISION), Moondream2Predictor
        )
        try:
            return (predictor.generate(image, question),)
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"MoonDream": MoonDream}
NODE_DISPLAY_NAME_MAPPINGS = {"MoonDream": "MoonDream (Moondream 2)"}
