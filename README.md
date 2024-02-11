# 👁️ VLM_nodes 
Examples below!

## Usage
```
cd custom_nodes
git clone https://github.com/gokayfem/ComfyUI_VLM_nodes.git
```
## VLM Nodes
Utilizes ```llama-cpp-python``` for integration of LLaVa models. You can load and use any VLM with LLaVa models in GGUF format with this nodes.   
You need to download the model ```ggml-model-q4_k.gguf``` and it's clip projector ```mmproj-model-f16.gguf``` from this repositories (in the files and versions).  
```python=>3.9``` is necessary.  
Put all of the files inside ```models/LLavacheckpoints```  
Note that every **model's clip projector** is different!  
- [LlaVa 1.6 Mistral 7B](https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/)
- [Nous Hermes 2 Vision](https://huggingface.co/billborkowski/llava-NousResearch_Nous-Hermes-2-Vision-GGUF)
- [LlaVa 1.5 7B](https://huggingface.co/mys/ggml_llava-v1.5-7b/)
- [LlaVa 1.5 13B](https://huggingface.co/mys/ggml_llava-v1.5-13b)
- [BakLLaVa](https://huggingface.co/mys/ggml_bakllava-1)  
etc..

## Automatic Prompt Generation and Suggestion Nodes
**Get Keyword** node: It can take LLava outputs and extract keywords from them.    
**LLava PromptGenerator** node: It can create prompts given descriptions or keywords using  (input prompt could be Get Keyword or LLava output directly).  
**Suggester** node: It can generate 5 different prompts based on the original prompt using consistent in the options or random prompts using random in the options.  
Works best with LLava 1.5 and 1.6.  

Outputs are JSON files, you can see them as a text using JsonToText Node.  
You can see any string output with ViewText Node  
You can set any string input using SimpleText Node  

## LLM Prompt Generation nodes

**LLM PromptGenerator** node: 
[Qwen 1.8B Stable Diffusion Prompt](https://huggingface.co/hahahafofo/Qwen-1_8B-Stable-Diffusion-Prompt-GGUF)  
[IF prompt MKR](https://huggingface.co/impactframes/IFpromptMKR-7b-L2-gguf-q4_k_m)  
This LLM's works best for now for prompt generation.  
**LLMSampler** node: You can chat with any LLM in gguf format, you can use LLava models as an LLM also.  

**API PromptGenerator** node: You can use ChatGPT and DeepSeek API's to create prompts. https://platform.deepseek.com/ gives 10m free tokens.
- ChatGPT-4
- ChatGPT-3.5
- DeepSeek
You can use them for simple chat also there is an option in the node.

## moondream Node
This node is designed to work with the Moondream model, a powerful small vision language model built by @vikhyatk using SigLIP, Phi-1.5, and the LLaVa training dataset. 
The model boasts 1.6 billion parameters and is made available for research purposes only; commercial use is not allowed.

## JoyTag Node
@fpgamine's JoyTag is a state of the art AI vision model for tagging images, with a focus on sex positivity and inclusivity.  
It uses the Danbooru tagging schema, but works across a wide range of images, from hand drawn to photographic.

## Example LLaVa Nodes
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/c30b9599-fa14-4f1a-b023-65a3697892f2)

## Example Using Automatic Prompt Generation
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/bff68f6f-5f77-4cd6-ade3-6810a32500bf)

## LLM Nodes
![VLM + LLM](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/4897d11a-e818-4d7e-bf04-0cd7dd4102dc)

## Example moondream
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/79ea61e9-60c6-406d-9e83-0d16128e30a6)

## Example Joytag
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/df9da377-59e8-4b39-a31a-0e3b5071a8cc)

## Example Prompt Generation 
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/1c557f10-52ee-4e1f-ab8a-20932a07dd3b)

## Example SimpleChat
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/057cfc2e-e772-43c0-972f-2916e6aeb03d)

## Example LLava Sampler Advanced
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/32210c37-fe7d-479f-b0a6-2eb13ea0aac1)

