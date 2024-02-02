from openai import OpenAI

# Define the system message
system_msg = """
You are an helpful asistant. Answer optional questions or help the user for their optional queries.
"""

class SimpleChat:
    def __init__(self):
    	pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["ChatGPT-3.5", "ChatGPT-4", "DeepSeek"],
                    {
                        "default" : "ChatGPT-3.5"
                    }
                )
                ,        
                "api_key": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                "query_from_other_nodes": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    }
                ),
                "optional_question": (
                            "STRING",
                            {
                                "multiline": True,
                                "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "chat"

    CATEGORY = "VLM Nodes/Simple Chat"

    def chat(self, model_name, api_key, query_from_other_nodes, optional_question):

        # Define the user message
        if model_name == "DeepSeek":
            model = "deepseek-chat"
            base_url = "https://api.deepseek.com/v1"
        elif model_name == "ChatGPT-3.5":
            model = "gpt-3.5-turbo"
            base_url = None
        elif model_name == "ChatGPT-4":
            model = "gpt-4"
            base_url = None
        user_msg = f"""
        Optional User Query: {query_from_other_nodes}
	    Optional User Question: {optional_question}

        Output: 
        """
        
        client = OpenAI(api_key = api_key, base_url=base_url)

        completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        )

        prompt = completion.choices[0].message.content       
        return (prompt,)
    
# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {"SimpleChat": SimpleChat}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"SimpleChat": "SimpleChat Node"}
