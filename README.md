# 👁️ VLM_nodes 
Examples below!

## LLavaLoader, LavaSampler, LLavaClipLoader Nodes
Utilizes ```llama-cpp-python``` for integration of LLaVa models. You can load and use any llm with LLaVa models in GGUF format with this nodes.  
You need to download the clip projector ```mmproj-model-f16.gguf```  from this repositories. ```python=>3.9``` is necessary. Put all of the files inside ```models/LLavacheckpoints```
- [LlaVa 1.6 Mistral 7B](https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/)
- [Nous Hermes 2 Vision](billborkowski/llava-NousResearch_Nous-Hermes-2-Vision-GGUF)
- [LlaVa 1.5 7B](https://huggingface.co/mys/ggml_llava-v1.5-7b/)
- [LlaVa 1.5 13B](https://huggingface.co/mys/ggml_llava-v1.5-13b)
- [BakLLaVa](https://huggingface.co/mys/ggml_bakllava-1) 
etc..

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
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/c30b9599-fa14-4f1a-b023-65a3697892f2)

## Example moondream
![image](https://github.com/gokayfem/VLM_nodes/assets/88277926/2e82fe70-550d-437c-8738-6fb638e42d1d)

## Example Joytag
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/df9da377-59e8-4b39-a31a-0e3b5071a8cc)

## Example Prompt Generation 
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/1c557f10-52ee-4e1f-ab8a-20932a07dd3b)

## Example SimpleChat
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/057cfc2e-e772-43c0-972f-2916e6aeb03d)

## Example LLava Sampler Advanced
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/32210c37-fe7d-479f-b0a6-2eb13ea0aac1)

