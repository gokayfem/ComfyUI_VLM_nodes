from huggingface_hub import snapshot_download
from pathlib import Path
import torch
import os
import soundfile as sf
from folder_paths import output_directory
import folder_paths
import datetime
from pathlib import Path

# Define the directory for saving files related to the audio model
files_for_audio_model = Path(folder_paths.folder_names_and_paths["LLavacheckpoints"][0][0]) / "files_for_audioldm2"
files_for_audio_model.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
base_path = os.path.dirname(os.path.realpath(__file__))

# Our any instance wants to be a wildcard string
any = AnyType("*")
class AudioLDM2ModelPredictor:
    
    def __init__(self):
        from diffusers import AudioLDM2Pipeline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Use snapshot_download to manage the model download/cache
        self.model_path = snapshot_download("cvssp/audioldm2",
                                            local_dir=files_for_audio_model,
                                            force_download=False,  # Set to True to always download
                                            local_files_only=False,  # Download if not available locally
                                            use_auth_token=False,  # Set to True if using a private model
                                            local_dir_use_symlinks="auto",  # Auto-manage symlinks
                                            ignore_patterns=["*.bin", "*.jpg", "*.png"])  # Ignore unrelated files

        self.pipeline = AudioLDM2Pipeline.from_pretrained(self.model_path, 
                                                          torch_dtype=torch_dtype).to(self.device)
        self.generator = torch.Generator(self.device)

    def generate_audio(self, text, negative_prompt, duration, guidance_scale, random_seed, sample_rate, n_candidates=1, extension="wav"):
        if text is None:
            raise ValueError("Please provide a text input.")
        
        # Manual seed for reproducibility
        self.generator.manual_seed(int(random_seed))

        # Generate audio
        waveforms = self.pipeline(
            text,
            audio_length_in_s=duration,
            guidance_scale=guidance_scale,
            num_inference_steps=200,
            negative_prompt=negative_prompt,
            num_waveforms_per_prompt=n_candidates,
            generator=self.generator,
        )["audios"]

        final_waveforms = waveforms[0].tolist()
        return (final_waveforms, sample_rate)  # Return the path of the generated audio file


class AudioLDM2Node:
    def __init__(self):
        self.predictor = AudioLDM2ModelPredictor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING",{"default": "", "forceInput": True}),
                "negative_prompt": ("STRING",{"default": "", "forceInput": True}),
                "duration": ("INT",{"default": 10, "min": 1, "max": 60, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.1, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "step": 1}),
                "n_candidates": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "sample_rate": ("INT", {"default": 16000, "min": 8000, "max": 48000, "step": 1}),
                "extension": (["wav", "mp3", "flac"], {"default": "wav"}),
            }
        }

    RETURN_NAMES = ("wave_form", "sample_rate", )
    RETURN_TYPES = (any, "INT", )
    OUTPUT_NODE = True
    FUNCTION = "generate_audio_final"

    CATEGORY = "VLM Nodes/Audio"

    def generate_audio_final(self, text, negative_prompt, duration, guidance_scale, sample_rate, seed, n_candidates, extension):
        wave_form, sample_rate_final = self.predictor.generate_audio(text, negative_prompt, duration, guidance_scale, seed, sample_rate, n_candidates, extension)
        return (wave_form, sample_rate_final, )

class SaveAudioNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "waveforms": (any, {}),  # Assuming 'any' is a placeholder for the actual data type
                "sample_rate": ("INT", {"forceInput": True}),
                "extension": (["wav", "mp3", "flac"], {"default": "wav"})  # mp3, wav, flac
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    CATEGORY = "VLM Nodes/Audio"
    OUTPUT_NODE = True

    def save_audio(self, waveforms, sample_rate, extension):
        # Define the date format
        date_formats = {
            'yyyyMMdd_HHmmss': lambda d: '{}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(d.year, d.month, d.day, d.hour, d.minute, d.second),
        }
        
        # Generate the date-based prefix
        current_datetime = datetime.datetime.now()
        print(current_datetime.hour, current_datetime.minute, current_datetime.second)
        for format_key, format_lambda in date_formats.items():
            preset_prefix = f"{format_lambda(current_datetime)}"
        
        # Build the filename and save the audio
        audio_path = Path(output_directory) / f"{preset_prefix}_audio.{extension}"
        sf.write(audio_path.as_posix(), waveforms, sample_rate)

        return ()

NODE_CLASS_MAPPINGS = {"AudioLDM2Node": AudioLDM2Node,
                       "SaveAudioNode": SaveAudioNode}
NODE_DISPLAY_NAME_MAPPINGS = {"AudioLDM2Node": "AudioLDM-2 Node",
                             "SaveAudioNode": "Save Audio Node"}