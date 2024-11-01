<div align="center">
  <h1> 👁️ VLM Nodes</h1>
  <p align="center">
    <b> 🔽Examples below</b>  •  
    📙 <a href="https://github.com/gokayfem/Awesome-VLM-Architectures">Visit my other repo to learn more about Vision Language Models</a> 
  </p>
</div>
<br/>

## Usage
- For **Windows** and **Linux**
```
cd custom_nodes
git clone https://github.com/gokayfem/ComfyUI_VLM_nodes.git
```
## Acknowledgements

 - [JAGS](https://github.com/jags111) 
 - [EnragedAntelope](https://github.com/EnragedAntelope)

**If you get errors related to llama-cpp-python or if it is not using GPU.**  
**I recommend installing it with the right arguments provided in this link [llama-cpp-python](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation)**  

## VLM Nodes
Utilizes ```llama-cpp-python``` for integration of LLaVa models. You can load and use any VLM with LLaVa models in GGUF format with this nodes.   
You need to download the model similar to ```ggml-model-q4_k.gguf``` and it's clip projector similar to ```mmproj-model-f16.gguf``` from this repositories (in the files and versions).  
```python=>3.9``` is necessary.  
Put all of the files inside ```models/LLavacheckpoints```  
Note that every **model's clip projector** is different!  
- [LlaVa 1.6 Mistral 7B](https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/)
- [Nous Hermes 2 Vision](https://huggingface.co/billborkowski/llava-NousResearch_Nous-Hermes-2-Vision-GGUF)
- [LlaVa 1.5 7B](https://huggingface.co/mys/ggml_llava-v1.5-7b/)
- [LlaVa 1.5 13B](https://huggingface.co/mys/ggml_llava-v1.5-13b)
- [BakLLaVa](https://huggingface.co/mys/ggml_bakllava-1)  
etc..

## Structured Output
Getting structured outputs can be quite challenging through prompt engineering alone.  
I've added the Structured Output node to VLM Nodes.  
Now, you can obtain your answers reliably.  
You can extract entities, numbers, classify prompts with given classes, and generate one specific prompt. These are just a few examples.  
You can add additional descriptions to fields and choose the attributes you want it to return.  
![structured](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/43b86ad4-0b91-499f-b2fd-d9771ee4acdd)

## Image to Music
Utilizes VLMs, LLMs and [AudioLDM-2](https://arxiv.org/abs/2308.05734) to make music from images.  
Use SaveAudioNode to save the music inside ```output``` folder.  
It will automatically download the necessary files into ```models/LLavacheckpoints/files_for_audioldm2```  

https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/2c5bdcde-d637-49ad-b317-14ac0a12f7df

## LLM to Music
Utilizes Chat Musician, an open-source LLM that integrates intrinsic musical abilities.    
[ChatMusician Demo Page](https://ezmonyi.github.io/ChatMusician/)  
You can try prompts from this demo page.

**Download the GGUF file**  
[ChatMusician GGUF Files](https://huggingface.co/MaziyarPanahi/ChatMusician-GGUF/tree/main)  
**ChatMusician.Q5_K_M.gguf** or **ChatMusician.Q5_K_S.gguf** recommended  
### BIG BIG BIG Warning: It **does NOT work perfectly**, if you got errors accept the error **queue prompt** again with the same settings!!

https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/7f22d4f2-b998-402e-88c8-c382a730d624

## InternLM-XComposer2-VL Node
Utilizes ```AutoGPTQ``` for integration of InternLM-XComposer2-VL Model. It will automatically download the necessary files into ```models/LLavacheckpoints/files_for_internlm```.
This is one of the best models for visual perception.   
**Important Note : This model is heavy.**
- [InternLM-XComposer2](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b-4bit)

## Automatic Prompt Generation and Suggestion Nodes
**Get Keyword** node: It can take LLava outputs and extract keywords from them.    
**LLava PromptGenerator** node: It can create prompts given descriptions or keywords using  (input prompt could be Get Keyword or LLava output directly).  
**Suggester** node: It can generate 5 different prompts based on the original prompt using consistent in the options or random prompts using random in the options.  
- Works best with **LLava 1.5** and **1.6**.  

**Play with the ```temperature``` for creative or consistent results. Higher the temperature more creative are the results.**  
If you want to dive deep into [LLM Settings](https://www.promptingguide.ai/introduction/settings)  

Outputs are JSON looking texts, you can see them as a text using JsonToText Node.  
You can see any string output with ViewText Node  
You can set any string input using SimpleText Node  
Utilizes ```llama-cpp-agents``` for getting structured outputs.  
## LLM Prompt Generation from text nodes

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

## UForm-Gen2 Qwen Node
UForm-Gen2 is an extremely fast small generative vision-language model primarily designed for Image Captioning and Visual Question Answering.  
[UForm-Gen2 Qwen](https://huggingface.co/unum-cloud/uform-gen2-qwen-500m)  
It will automatically download the necessary files into ```models/LLavacheckpoints/files_for_uform_gen2_qwen```

## Kosmos-2 Node
Kosmos-2: Grounding Multimodal Large Language Models to the World.
[Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)
It will automatically download the necessary files into ```models/LLavacheckpoints/files_for_kosmos2```

## moondream1 and moondream2 Node
This node is designed to work with the Moondream model, a powerful small vision language model built by @vikhyatk using SigLIP, Phi-1.5, and the LLaVa training dataset. 
The model boasts 1.6 billion parameters and is made available for research purposes only; commercial use is not allowed.  

moondream2 is a small vision language model designed to run efficiently on edge devices.  

It will automatically download the necessary files into ```models/LLavacheckpoints/files_for__moondream``` and ```models/LLavacheckpoints/files_for_moondream2```

## JoyTag Node
@fpgamine's JoyTag is a state of the art AI vision model for tagging images, with a focus on sex positivity and inclusivity.  
It uses the Danbooru tagging schema, but works across a wide range of images, from hand drawn to photographic.
It will automatically download the necessary files into ```models/LLavacheckpoints/files_for_joytagger```

## Qwen2-VL Node
Utilizes the latest Qwen2-VL series of models, which are state-of-the-art vision language models supporting various resolutions, ratios, and languages. The models excel at:
- Understanding images of various resolutions & ratios
- Complex visual reasoning and decision making
- Multilingual support (English, Chinese, European languages, Japanese, Korean, Arabic, Vietnamese, etc.)

Available models include 2B, 7B, and 72B parameter versions, with standard, AWQ, and GPTQ quantized variants. It will automatically download the necessary files into `models/LLavacheckpoints/files_for_qwen2vl`.

**Important Note**: Larger models (7B, 72B) require significant VRAM. Choose quantized versions (AWQ, GPTQ) for reduced memory usage.

[Link to Qwen2-VL Models](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

## Example LLaVa Nodes
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/c30b9599-fa14-4f1a-b023-65a3697892f2)

## Example Image to Music
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/e216c299-c9ea-4227-aa85-9533cb6af260)

## Example InternLM-XComposer Node
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/ff051e6c-5ad8-41fe-9d77-fdeea6eb2c5c)

## Example Using Automatic Prompt Generation
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/bff68f6f-5f77-4cd6-ade3-6810a32500bf)

## LLM Nodes
![VLM + LLM](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/4897d11a-e818-4d7e-bf04-0cd7dd4102dc)

## Example UForm-Gen2 Qwen Node
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/4531f8f2-94af-498f-b364-f9e07c826eb5)

# Example Kosmos-2 Node
![image](https://github.com/gokayfem/ComfyUI_VLM_nodes/assets/88277926/a28035dc-a0c4-4c4f-9c87-e8b284c3997d)

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

