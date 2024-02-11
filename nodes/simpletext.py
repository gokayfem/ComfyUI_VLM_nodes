import json
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

    CATEGORY = "VLM Nodes/Text"

    def simple_text(self, input_text):

        return (input_text, )
    
class JsonToText:
    def __init__(self):
    	pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "json_to_text"
    OUTPUT_NODE = True

    CATEGORY = "VLM Nodes/Text"

    def json_to_text(self, text):
        # Parse the combined JSON string
        loaded_json = json.loads(text)

        # Process each element in the combined data
        merged_ideas = []
        excluded_keywords = ['create', 'generate']  # Words to exclude in the prompt

        for key, value in loaded_json.items():
            if key == "prompt":  # Special handling for the "prompt" key
                value = ' '.join(word for word in value.split() if word.lower() not in excluded_keywords)
                merged_ideas.append(value)  # Append the processed prompt without the key
            elif key.startswith("sugg"):  # Handle keys that start with "suggestion"
            # Directly append the value, skipping the key
                merged_ideas.append(value)
            elif isinstance(value, list):  # Unpack the list if the value is a list and include the key
                merged_ideas.append(f"{key}: {', '.join(value)}")
            else:  # For non-list values other than "prompt", include the key
                merged_ideas.append(f"{key}: {value}")

        formatted_output_str = "\n\n".join(merged_ideas)
        return {"ui": {"text": formatted_output_str}, "result": (formatted_output_str,)}

class ViewText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "view_text"
    OUTPUT_NODE = True

    CATEGORY = "VLM Nodes/Text"

    def view_text(self, text):
        # Parse the combined JSON string
        return {"ui": {"text": text}, "result": (text,)}

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {"SimpleText": SimpleText,
                       "JsonToText": JsonToText,
                       "ViewText": ViewText}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"SimpleText": "SimpleText",
                              "JsonToText": "JsonToText",
                              "ViewText": "ViewText"}
