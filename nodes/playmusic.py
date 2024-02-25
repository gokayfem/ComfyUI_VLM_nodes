# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
any = AnyType("*")


class PlayMusic:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mode": (["always", "on empty queue"], {}),
            "volume": ("FLOAT", {"min": 0, "max": 1, "step": 0.1, "default": 0.5}),
            "wave_form": ([], {"forceInput": True}),
            "sample_rate": ("INT", {"forceInput": True}),
        }}

    FUNCTION = "nop"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    RETURN_TYPES = (any,)

    CATEGORY = "VLM Nodes/AudioLDM2"

    def IS_CHANGED(self, **kwargs):
        return float("NaN")

    def nop(self, mode, volume, wave_form, sample_rate):
        return {"ui": {"a": wave_form, "b": sample_rate}, "result": (any,)}


NODE_CLASS_MAPPINGS = {
    "PlaySound": PlayMusic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlaySound": "PlaySound Node",
}
