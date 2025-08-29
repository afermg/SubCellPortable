import numpy as np
import torch
import torch.nn.functional as F
from skimage.io import imsave
from torchvision.utils import make_grid
import concurrent.futures
import threading
import h5py
import os

CLASS2NAME = {
    0: "Actin filaments",
    1: "Aggresome",
    2: "Cell Junctions",
    3: "Centriolar satellite",
    4: "Centrosome",
    5: "Cytokinetic bridge",
    6: "Cytoplasmic bodies",
    7: "Cytosol",
    8: "Endoplasmic reticulum",
    9: "Endosomes",
    10: "Focal adhesion sites",
    11: "Golgi apparatus",
    12: "Intermediate filaments",
    13: "Lipid droplets",
    14: "Lysosomes",
    15: "Microtubules",
    16: "Midbody",
    17: "Mitochondria",
    18: "Mitotic chromosome",
    19: "Mitotic spindle",
    20: "Nuclear bodies",
    21: "Nuclear membrane",
    22: "Nuclear speckles",
    23: "Nucleoli",
    24: "Nucleoli fibrillar center",
    25: "Nucleoli rim",
    26: "Nucleoplasm",
    27: "Peroxisomes",
    28: "Plasma membrane",
    29: "Vesicles",
    30: "Negative",
}

CLASS2COLOR = {
    0: "#ffeb3b",
    1: "#76ff03",
    2: "#ff6d00",
    3: "#eb30c1",
    4: "#faadd4",
    5: "#795548",
    6: "#64ffda",
    7: "#00e676",
    8: "#03a9f4",
    9: "#4caf50",
    10: "#ffc107",
    11: "#00bcd4",
    12: "#cddc39",
    13: "#212121",
    14: "#8bc34a",
    15: "#ff9800",
    16: "#ae8c08",
    17: "#ffff00",
    18: "#31b61f",
    19: "#9e9e9e",
    20: "#2196f3",
    21: "#e91e63",
    22: "#3f51b5",
    23: "#9c27b0",
    24: "#673ab7",
    25: "#d3a50b",
    26: "#f44336",
    27: "#009688",
    28: "#ff9e80",
    29: "#242e4b",
    30: "#000000",
}


def min_max_standardize(batch_data):
    min_val = torch.amin(batch_data, dim=(1, 2, 3), keepdims=True)
    max_val = torch.amax(batch_data, dim=(1, 2, 3), keepdims=True)

    batch_data = (batch_data - min_val) / (max_val - min_val + 1e-8)
    return batch_data


def save_attention_map(attn, input_shape, output_path):
    attn = F.interpolate(attn, size=input_shape, mode="bilinear", align_corners=False)
    attn = make_grid(
        attn.permute(1, 0, 2, 3),
        normalize=True,
        nrow=attn.shape[1],
        padding=0,
        scale_each=True,
    )
    attn = (attn.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    imsave(output_path + "_attention_map.png", attn)


@torch.no_grad()
def run_model(model, batch_data, device, output_paths, save_attention_maps=True, embeddings_only=False, 
              output_format="individual", async_saving=False):
    """
    Run model inference on a batch of images

    Args:
        model: The ViT model
        batch_data: Tensor of shape (batch_size, channels, height, width)
        device: Device to run inference on
        output_paths: List of output paths for each image in batch
        save_attention_maps: Whether to save attention map images (default: True)
        embeddings_only: If True, skip probability computation and saves (default: False)
        output_format: "individual" for separate files, "h5ad" for combined (default: "individual")
        async_saving: If True, save files asynchronously (default: False)

    Returns:
        If async_saving=False: List of (embedding, probabilities) tuples for each image
        If async_saving=True: (results_list, save_future) where save_future can be awaited
        Note: probabilities will be None if embeddings_only=True
    """
    batch_data = batch_data.to(device)
    batch_data = min_max_standardize(batch_data)
    
    # Run model on entire batch
    output = model(batch_data)
    
    # Convert to numpy once for all items in batch
    embeddings_batch = output.pool_op.cpu().numpy()
    
    # Only compute probabilities if not in embeddings_only mode
    probabilities_batch = None
    if not embeddings_only:
        probabilities_batch = output.probabilities.cpu().numpy()
    
    # Free GPU memory immediately - critical for async mode
    del output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Prepare results
    results = []
    for i, output_path in enumerate(output_paths):
        embedding = embeddings_batch[i]
        probabilities = probabilities_batch[i] if probabilities_batch is not None else None
        results.append((embedding, probabilities))
    
    # Handle file saving based on mode
    if output_format == "h5ad":
        # For H5AD, return results without saving - will be collected and saved at end
        return results
        
    elif async_saving:
        # Return immediately, save asynchronously
        save_future = _async_save_batch(
            results, output_paths, embeddings_only, save_attention_maps, 
            None, batch_data, output_format  # Pass None for output since we freed it
        )
        return results, save_future
    else:
        # Synchronous saving (original behavior)
        _sync_save_batch(
            results, output_paths, embeddings_only, save_attention_maps,
            None, batch_data, output_format  # Pass None for output since we freed it
        )
        return results


def _sync_save_batch(results, output_paths, embeddings_only, save_attention_maps, output, batch_data, output_format):
    """Synchronous batch saving (original behavior)"""
    
    # Individual files
    for i, output_path in enumerate(output_paths):
        embedding, probabilities = results[i]
        
        # Save embedding
        np.save(output_path + "_embedding.npy", embedding)
        
        # Save probability (only if not embeddings_only)
        if not embeddings_only and probabilities is not None:
            np.save(output_path + "_probabilities.npy", probabilities)
        
        # Save attention map for this image (configurable)
        # Note: attention maps disabled in async mode to prevent GPU memory issues
        if save_attention_maps and output is not None and output.pool_attn is not None:
            attn = output.pool_attn[i:i+1]  # Keep batch dimension for F.interpolate
            save_attention_map(attn, (batch_data.shape[2], batch_data.shape[3]), output_path)


def _async_save_batch(results, output_paths, embeddings_only, save_attention_maps, output, batch_data, output_format):
    """Asynchronous batch saving using ThreadPoolExecutor"""
    
    # Individual files - can parallelize
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    def save_single_item(i):
        output_path = output_paths[i]
        embedding, probabilities = results[i]
        
        # Save embedding
        np.save(output_path + "_embedding.npy", embedding)
        
        # Save probability (only if not embeddings_only)
        if not embeddings_only and probabilities is not None:
            np.save(output_path + "_probabilities.npy", probabilities)
        
        # Save attention map (configurable) 
        # Note: attention maps disabled in async mode to prevent GPU memory issues
        if save_attention_maps and output is not None and output.pool_attn is not None:
            attn = output.pool_attn[i:i+1]  # Keep batch dimension for F.interpolate
            save_attention_map(attn, (batch_data.shape[2], batch_data.shape[3]), output_path)
    
    # Submit all save tasks
    futures = [executor.submit(save_single_item, i) for i in range(len(output_paths))]
    
    # Return a future that completes when all saves are done
    def wait_all():
        concurrent.futures.wait(futures)
        executor.shutdown(wait=True)
    
    return concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(wait_all)



