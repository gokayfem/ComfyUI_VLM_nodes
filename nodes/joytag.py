from .joytagger import Models
from PIL import Image
import torch.amp.autocast_mode
from pathlib import Path
import torch
import torchvision.transforms.functional as TVF
from huggingface_hub import snapshot_download
from torchvision import transforms
import folder_paths

THRESHOLD = 0.4

# Define your local directory where you want to save the files
files_for_joytagger = Path(folder_paths.folder_names_and_paths["LLavacheckpoints"][0][0]) / "files_for_joytagger"

# Check if the directory exists, create if it doesn't (optional)
files_for_joytagger.mkdir(parents=True, exist_ok=True)

def download_joytag():
    # Ensure the correct behavior based on the existence of the local directory
    print(f"Target directory for download: {files_for_joytagger}")
    
    # Call snapshot_download with specified parameters
    path = snapshot_download(
        "fancyfeast/joytag",  # Example repo_id
        local_dir=files_for_joytagger,
        force_download=False,  # Set to True if you always want to download, regardless of local copy
        local_files_only=False,  # Set to False to allow downloading if not available locally
        local_dir_use_symlinks="auto"  # or set to True/False based on your symlink preference
    )
    print(f"Model path: {path}")
    return path


def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
	# Pad image to square
	image_shape = image.size
	max_dim = max(image_shape)
	pad_left = (max_dim - image_shape[0]) // 2
	pad_top = (max_dim - image_shape[1]) // 2

	padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
	padded_image.paste(image, (pad_left, pad_top))

	# Resize image
	if max_dim != target_size:
		padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
	
	# Convert to tensor
	image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

	# Normalize
	image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

	return image_tensor




# Extract and process the tags
def process_tag(tag):
	tag = tag.replace("(medium)", "")  # Remove (medium)
	tag = tag.replace("\\", "")  # Remove \
	tag = tag.replace("m/", "")  # Remove m/
	tag = tag.replace("-", "")  # Remove -
	tag = tag.replace("_", " ")  # Replace underscores with spaces
	tag = tag.strip()  # Remove leading and trailing spaces
	return tag

class Joytag:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
				"tag_number": ("INT", {
                    "default": 1, 
                    "min": 1, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),                
			},
		}

	RETURN_TYPES = ("STRING",)

	FUNCTION = "tags"

	CATEGORY = "VLM Nodes/JoyTag"

	def tags(self, image, tag_number):
		path = download_joytag()
		print(f"Model path: {path}")
		model = Models.VisionModel.load_model(Path(path), device='cuda')
		model.eval()
		with open(Path(path) / 'top_tags.txt', 'r') as f:
			top_tags = [line.strip() for line in f.readlines() if line.strip()]

		@torch.no_grad()
		def predict(image: Image.Image):
			image_tensor = prepare_image(image, model.image_size)
			batch = {
				'image': image_tensor.unsqueeze(0).to('cuda'),
			}

			with torch.amp.autocast_mode.autocast('cuda', enabled=True):
				preds = model(batch)
				tag_preds = preds['tags'].sigmoid().cpu()
			
			scores = {top_tags[i]: tag_preds[0][i] for i in range(len(top_tags))}
			predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
			tag_string = ', '.join(predicted_tags)

			return tag_string, scores
			
		image = transforms.ToPILImage()(image[0].permute(2, 0, 1))
		_, scores = predict(image)

		# Get the top 50 tag and score pairs
		top_tags_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:tag_number]

		# Extract the tags from the pairs
		top_tags_processed = [process_tag(tag) for tag, _ in top_tags_scores]
		
		top_tags_full = [tag for tag in top_tags_processed if tag]

		# Concatenate the tags with a comma separator
		top_50_tags_string = ', '.join(top_tags_full)
		
		return (top_50_tags_string, )


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {"Joytag": Joytag}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"Joytag": "Joytag Node"}
