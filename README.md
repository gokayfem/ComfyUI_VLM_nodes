# 👁️ VLM_nodes - Work In Progress

## LLavaLoader, LavaSampler, LLavaClipLoader Nodes
Utilizes ```llama-cpp-python``` for integration of LLaVa models. You can load and use LLaVa models in GGUF format with this nodes.  
You need to download the clip handler from this repositories. ```python=>3.9``` is necessary. Put all of the files inside ```models/LLavacheckpoints```
- [Llava 1.5 7B](https://huggingface.co/mys/ggml_llava-v1.5-7b/)
- [Llava 1.5 13B](https://huggingface.co/mys/ggml_llava-v1.5-13b)
- [BakLLava](https://huggingface.co/mys/ggml_bakllava-1)

## moondream Node
This node is designed to work with the Moondream model, a powerful small vision language model built by @vikhyatk using SigLIP, Phi-1.5, and the LLaVa training dataset. 
The model boasts 1.6 billion parameters and is made available for research purposes only; commercial use is not allowed.

## Automatic Prompt Generator or SimpleChat Node
This node is designed to transform textual descriptions into automatically generated image generation prompts. 
It simplifies the process of creating vivid and detailed prompts for image generation. Optionally you can chat with llms using SimpleChat Node.

You can use:
- ChatGPT-4
- ChatGPT-3.5
- DeepSeek  
https://platform.deepseek.com/ gives 10m free tokens.

## JoyTag Node
@fpgamine's JoyTag is a state of the art AI vision model for tagging images, with a focus on sex positivity and inclusivity.  
It uses the Danbooru tagging schema, but works across a wide range of images, from hand drawn to photographic.

## Usage

```
cd custom_nodes
git clone https://github.com/gokayfem/ComfyUI_VLM_nodes.git
```
## Example LLaVa Nodes
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/35501b3c-571d-4503-a14a-36851f8b5968)

## Example moondream
![image](https://github.com/gokayfem/VLM_nodes/assets/88277926/2e82fe70-550d-437c-8738-6fb638e42d1d)

## Example Joytag
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/df9da377-59e8-4b39-a31a-0e3b5071a8cc)

## Example Prompt Generation 
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/1c557f10-52ee-4e1f-ab8a-20932a07dd3b)

## Example SimpleChat
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/057cfc2e-e772-43c0-972f-2916e6aeb03d)
