import torch
import os
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from diffusers import DiffusionPipeline
from sklearn.preprocessing import MultiLabelBinarizer
from diffusers.models.attention_processor import AttnProcessor2_0

def load_attention(attention_path, image_names):
    """
    Loads attention weights from .pt files for a given list of image names.

    Args:
        attention_path (str): The path to the directory containing the .pt files.
        image_names (list): A list of image filenames (e.g., ['0.png', '1.png']).

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              attention weights for a corresponding image. Returns None for missing files.
    """
    all_attention_weights = []
    for image_name in image_names:
        base_name = os.path.splitext(image_name)[0]
        pt_path = os.path.join(attention_path, f'{base_name}.pt')
        
        if os.path.exists(pt_path):
            try:
                attention_weights = torch.load(pt_path, map_location=torch.device('cpu'))
                all_attention_weights.append(attention_weights)
            except Exception as e:
                print(f"Warning: Could not load attention file {pt_path}. Error: {e}")
                all_attention_weights.append(None)
        else:
            print(f"Warning: Attention file not found for {image_name} at {pt_path}")
            all_attention_weights.append(None)

    return all_attention_weights
    

def load_labels(label_path):
    # Loads and processes CVAT annotations
    tree = ET.parse(label_path)
    root = tree.getroot()

    # Extract all possible labels from the meta section to define the classes
    all_labels = [label.find('name').text for label in root.findall('.//labels/label')]

    # Parse image tags and their associated labels
    labels_dict = {}
    for image in root.findall('image'):
        image_name = image.get('name')
        image_labels = [tag.get('label') for tag in image.findall('tag')]
        labels_dict[image_name] = image_labels

    # Sort image names to ensure a consistent order
    sorted_image_names = sorted(labels_dict.keys())

    # Create a list of labels in the same order as the sorted image names
    labels_list = [labels_dict[name] for name in sorted_image_names]

    # Use MultiLabelBinarizer to one-hot encode the labels
    binarizer = MultiLabelBinarizer(classes=all_labels)
    y = binarizer.fit_transform(labels_list)

    return sorted_image_names, y, binarizer


class AttnProcessorWithWeights(AttnProcessor2_0):
    def __init__(self, name, storage_dict):
        super().__init__()
        self.name = name
        self.storage = storage_dict

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        # Run normal attention
        output = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)

        # Manually compute and store attention weights
        if encoder_hidden_states is not None:  # Cross-attention
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)

            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale
            attention_probs = F.softmax(attention_scores, dim=-1)

            # Store on CPU to save GPU memory
            self.storage[self.name] = attention_probs.detach().cpu()

        return output


def generate_data(num_images=50, data_path='.', batch_size=4):
    """Batched inference with per-image attention extraction."""

    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("device: ", device)

    print("=== Running Inference ===")

    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

    # Define prompts to test different scenarios
    prompt = "a red sphere and a blue cube"
    num_inference_steps = 1

    # Calculate number of batches
    num_batches = (num_images + batch_size - 1) // batch_size

    print(f"Generating {num_images} images in {num_batches} batches of size {batch_size}")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        current_batch_size = end_idx - start_idx

        # Storage for this batch
        attn_weights = {}

        # Register processors for this batch
        for name, module in pipe.unet.named_modules():
            if name.endswith('0.attn2'):
                module.processor = AttnProcessorWithWeights(name, attn_weights)

        # Generate batch
        output = pipe(
            prompt=[prompt] * current_batch_size,  # Replicate prompt for batch
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0
        )

        # Split batched attention weights per image
        for img_idx in range(current_batch_size):
            global_idx = start_idx + img_idx

            # Extract per-image attention weights
            per_image_weights = {}
            for layer_name, weights in attn_weights.items():
                # weights shape: (batch*heads, spatial, text)
                batch_heads = weights.shape[0]
                heads_per_image = batch_heads // current_batch_size

                # Extract this image's heads
                start_head = img_idx * heads_per_image
                end_head = start_head + heads_per_image
                per_image_weights[layer_name] = weights[start_head:end_head]

            # Save
            torch.save(per_image_weights, os.path.join(data_path, f'{global_idx}.pt'))
            output.images[img_idx].save(os.path.join(data_path, f'{global_idx}.png'))

        print(f"Batch {batch_idx + 1}/{num_batches} complete ({end_idx}/{num_images} images)")

# if __name__ == '__main__':
#
#     DATA_DIR = r'/Users/arielkeslassy/Documents/reichman/courses/SNA/experiments/data'
#     ANNOTATIONS_FILE = 'annotations.xml'
#
#     try:
#         # 1. Load labels from the annotation file
#         print("--- Loading Labels ---")
#         image_names, y, binarizer = load_labels(ANNOTATIONS_FILE)
#
#         print(f"Successfully loaded and processed labels for {len(image_names)} images.")
#         print(f"Shape of the label matrix: {y.shape}")
#         print(f"Classes: {list(binarizer.classes_)}")
#
#         # 2. Load the corresponding attention maps
#         print("\n--- Loading Attention Maps ---")
#         attention_maps = load_attention(DATA_DIR, image_names)
#
#         # Filter out any maps that failed to load
#         loaded_attention_maps = [m for m in attention_maps if m is not None]
#         print(f"Successfully loaded {len(loaded_attention_maps)} out of {len(image_names)} attention maps.")
#
#         # 3. Example inspection
#         if loaded_attention_maps:
#             # Find the index of the first successfully loaded map
#             first_valid_index = next(i for i, m in enumerate(attention_maps) if m is not None)
#
#             print(f"\n--- Example Inspection ---")
#             print(f"Inspecting data for image: '{image_names[first_valid_index]}'")
#
#             # Show labels for this image
#             raw_labels = binarizer.inverse_transform(y[first_valid_index:first_valid_index+1])
#             print(f"  - Raw labels: {raw_labels[0] if raw_labels else '()'}")
#             print(f"  - Encoded vector: {y[first_valid_index]}")
#
#             # Show info about the loaded attention map
#             first_map_keys = loaded_attention_maps[0].keys()
#             print(f"  - Attention map layers captured: {len(first_map_keys)}")
#             if first_map_keys:
#                 example_layer = list(first_map_keys)[0]
#                 example_tensor = loaded_attention_maps[0][example_layer]
#                 print(f"  - Example layer ('{example_layer}') tensor shape: {example_tensor.shape}")
#
#     except FileNotFoundError as e:
#         print(f"Error: A required file was not found. Please ensure '{ANNOTATIONS_FILE}' and the .pt files are in the correct directory.")
#         print(f"Details: {e}")
#     except ET.ParseError:
#         print(f"Error: Failed to parse '{ANNOTATIONS_FILE}'. The file may be corrupt or not a valid XML.")
#     except ImportError:
#         print("Error: A required library is not installed. Please run 'pip install torch scikit-learn diffusers'.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
