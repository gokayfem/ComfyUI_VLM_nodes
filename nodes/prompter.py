from openai import OpenAI
from .prompts import system_msg_prompts
from .prompts import system_msg_simple

class PromptGenerateNode:
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
                "chat_type": 
                    ("BOOLEAN", 
                    {
                        "default": True, "label_on": "PromptGenerator", "label_off": "SimpleChat"
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
                "description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    }
                ),
                "question": (
                            "STRING",
                            {
                                "multiline": True,
                                "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "generate_prompt"

    CATEGORY = "VLM Nodes/Prompt Generator"

    def generate_prompt(self, model_name, chat_type, api_key, description, question):

        if chat_type == True:
            system_msg = system_msg_prompts
        elif chat_type == False:
            system_msg = system_msg_simple


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
        Description: {description}
	Optional Question: {question}

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
NODE_CLASS_MAPPINGS = {"PromptGenerate": PromptGenerateNode}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"PromptGenerate": "PromptGenerate Node"}
