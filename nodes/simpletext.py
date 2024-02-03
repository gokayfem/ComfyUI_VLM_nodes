class SimpleText:
    def __init__(self):
    	pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": (
                            "STRING",
                            {
                                "multiline": True,
                                "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "simple_text"

    CATEGORY = "VLM Nodes/SimpleText"

    def simple_text(self, input_text):

        return (input_text, )
    
# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {"SimpleText": SimpleText}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"SimpleText": "SimpleText Node"}
