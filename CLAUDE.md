# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SubCellPortable is a wrapper for running the Lundberg lab SubCell model (created by Ankit Gupta) in inference mode. It processes microscopy cell images to generate:
- **Embeddings**: 1536-dimensional feature vectors representing protein localization
- **Probability predictions**: Weighted probabilities across 31 subcellular location classes
- **Attention maps**: Visualizations showing where the model focused

The codebase has been optimized for high-performance batch processing using PyTorch DataLoader, reducing inference time from ~1 week to ~15.5 hours for 2.56M images.

## Common Commands

### Environment Setup
```bash
# Using uv (recommended - faster)
uv sync

# Using pip
pip install -r requirements.txt
```

### Running Inference

**Basic usage (new format):**
```bash
# Specify output directory (required for new CSV format)
python process.py --output_dir ./results

# Or set in config.yaml
python process.py
```

**With command-line arguments:**
```bash
# High-performance settings
python process.py -o ./results -c rybg -t mae_contrast_supcon_model -b 512 -w 8 -g 0

# Embeddings only (no classification)
python process.py -o ./results --embeddings_only -csv

# CPU-only mode
python process.py -o ./results -g -1

# Custom config and input files
python process.py --config custom_config.yaml --path-list experiment1.csv -o ./results

# Update model files from remote
python process.py -u
```

**Configuration options:**
- `--config`: Path to configuration YAML file (default: `config.yaml`)
- `--path-list`: Path to input CSV file with image paths (default: `path_list.csv`)
- `-o, --output_dir`: Output directory for all results (required for new CSV format)
- `-c, --model_channels`: Channel configuration (`rybg`, `rbg`, `ybg`, `bg`)
- `-t, --model_type`: Model type (`mae_contrast_supcon_model`, `vit_supcon_model`)
- `-b, --batch_size`: Batch size for GPU processing (default: 256)
- `-w, --num_workers`: Number of parallel data loading workers (default: 4)
- `-p, --prefetch_factor`: Prefetch factor for data loading (default: 2)
- `-g, --gpu`: GPU ID to use, -1 for CPU (default: -1)
- `-csv, --create_csv`: Generate combined CSV with results
- `-q, --quiet`: Suppress verbose logging
- `-u, --update_model`: Download/update model files

## Architecture

### Core Processing Pipeline

1. **Input CSV (`path_list.csv`)**: Defines image paths and output configuration
2. **Dataset Loading (`dataset.py`)**: PyTorch Dataset implementation for efficient batch loading
3. **Model Loading (`process.py`)**: Loads encoder and classifier(s) from model files
4. **Batch Inference (`inference.py`)**: Runs ViT model on image batches
5. **Output Generation**: Saves embeddings, probabilities, and attention maps

### Key Files and Their Roles

**`process.py`** - Main entry point
- Orchestrates the entire inference pipeline
- Handles configuration from CLI args, config.yaml, and constants
- Manages DataLoader creation and batch processing loop
- Supports both synchronous and asynchronous file saving
- Supports individual file output or H5AD format for large datasets

**`inference.py`** - Model inference logic
- `run_model()`: Core batch inference function
- `CLASS2NAME`: Dictionary mapping class indices (0-30) to subcellular location names
- `CLASS2COLOR`: HEX color codes for visualization
- Handles attention map generation and file I/O
- Supports async saving for performance optimization

**`dataset.py`** - PyTorch Dataset
- `SubCellDataset`: Custom Dataset for loading cell images
- `collate_fn()`: Custom collate function for batching
- Handles channel mapping (r=microtubules, y=ER, b=nuclei, g=protein)
- Supports graceful handling of missing images

**`vit_model.py`** - Model architecture
- `ViTPoolClassifier`: Main model class with encoder and classifier heads
- `GatedAttentionPooler`: Custom attention pooling mechanism
- `ViTInferenceModel`: Custom ViT implementation based on HuggingFace transformers
- `load_model_dict()`: Loads encoder and classifier(s) from checkpoint files

**`image_utils.py`** - Image handling
- Robust grayscale image reading from various sources (files, URLs, S3)
- Handles different image formats and bit depths
- Applies normalization and preprocessing

**`utils.py`** - Utility functions
- `get_nearest_neighbours()`: Find similar proteins from reference database

### Channel Configuration

The model supports different channel combinations:
- **`rybg`**: 4 channels - microtubules (R), ER (Y), nuclei (B), protein (G)
- **`rbg`**: 3 channels - microtubules (R), nuclei (B), protein (G)
- **`ybg`**: 3 channels - ER (Y), nuclei (B), protein (G)
- **`bg`**: 2 channels - nuclei (B), protein (G)

Channel mapping in CSV:
- `r_image`: Microtubules targeting marker
- `y_image`: ER targeting marker
- `b_image`: Nuclei targeting marker
- `g_image`: Protein targeting marker (the protein of interest)

### Model Directory Structure

Models are organized as:
```
models/
├── rybg/
│   ├── mae_contrast_supcon_model/
│   │   ├── encoder.pth
│   │   ├── classifier_s0.pth
│   │   └── model_config.yaml
│   └── vit_supcon_model/
├── rbg/
├── ybg/
└── bg/
```

Model files can be downloaded automatically using `update_model: True` or `-u` flag.

### Output Directory Configuration

**New Format (Recommended):**
All outputs go to a single `output_dir`:
```yaml
# config.yaml
output_dir: "./results"
```
Or via CLI: `python process.py -o ./results`

**Output Structure:**
```
results/
├── log.txt
├── result.csv (if create_csv: True)
├── embeddings.h5ad (if output_format: "h5ad")
├── cell1_embedding.npy
├── cell1_probabilities.npy
├── cell1_attention_map.png
├── experiment_A/
│   ├── cell2_embedding.npy
│   └── cell2_probabilities.npy
└── experiment_B/
    └── replicate_1/
        ├── cell3_embedding.npy
        └── cell3_probabilities.npy
```

**Subdirectory Support:**
Use subdirectories in `output_prefix` for organization:
- `cell1_` → `{output_dir}/cell1_*.npy`
- `experiment_A/cell2_` → `{output_dir}/experiment_A/cell2_*.npy`

### Output Formats

**Individual files (default):**
- `{output_dir}/{prefix}_embedding.npy`: 1536-dimensional embedding
- `{output_dir}/{prefix}_probabilities.npy`: 31-class probability array
- `{output_dir}/{prefix}_attention_map.png`: 64x64 attention visualization

**H5AD format (`output_format: "h5ad"`):**
- `{output_dir}/embeddings.h5ad`: Single file with all embeddings, probabilities, and metadata
- Compatible with AnnData/Scanpy for downstream analysis

**CSV output (`create_csv: True`):**
- `{output_dir}/result.csv`: Combined results with top predictions and full feature vectors
- Columns: id, top predictions, prob00-prob30, feat0000-feat1535

**Log file:**
- `{output_dir}/log.txt`: Processing log with timing and predictions

### Performance Optimization Features

**Batch Processing:**
- Uses PyTorch DataLoader for efficient GPU utilization
- Configurable batch size (typically 256-600 depending on GPU memory)
- Automatic memory cleanup after each batch

**Parallel Data Loading:**
- Multi-worker data loading (configurable via `num_workers`)
- Prefetching for reduced I/O bottleneck
- Persistent workers to avoid recreation overhead

**Async Saving:**
- Set `async_saving: True` to save files while processing next batch
- Reduces time spent waiting for disk I/O
- Automatically waits for all saves to complete at end

**Embeddings-Only Mode:**
- Set `embeddings_only: True` to skip probability computation
- Faster processing when only embeddings are needed
- No classifier loading required

**Memory Management:**
- Explicit CUDA cache clearing after each batch
- Pin memory for faster CPU-GPU transfers
- Configurable prefetch factor for memory vs. speed tradeoff

## Configuration Priority

Configuration sources are applied in this order (later overrides earlier):
1. Default values in `SubCellConfig` class
2. Configuration file (`config.yaml` by default, or custom file via `--config`)
3. Command-line arguments (highest priority)

**Note:** You can specify custom configuration and input files:
- `--config <path>`: Use a custom config file instead of `config.yaml`
- `--path-list <path>`: Use a custom input CSV instead of `path_list.csv`

## Subcellular Location Classes

The model predicts 31 classes (see `inference.py:11-43`):
- 0: Actin filaments
- 1: Aggresome
- 2: Cell Junctions
- 3: Centriolar satellite
- 4: Centrosome
- 5: Cytokinetic bridge
- 6: Cytoplasmic bodies
- 7: Cytosol
- 8: Endoplasmic reticulum
- 9: Endosomes
- 10: Focal adhesion sites
- 11: Golgi apparatus
- 12: Intermediate filaments
- 13: Lipid droplets
- 14: Lysosomes
- 15: Microtubules
- 16: Midbody
- 17: Mitochondria
- 18: Mitotic chromosome
- 19: Mitotic spindle
- 20: Nuclear bodies
- 21: Nuclear membrane
- 22: Nuclear speckles
- 23: Nucleoli
- 24: Nucleoli fibrillar center
- 25: Nucleoli rim
- 26: Nucleoplasm
- 27: Peroxisomes
- 28: Plasma membrane
- 29: Vesicles
- 30: Negative

## Important Implementation Details

### path_list.csv Format

**New Format (Recommended):**
```csv
r_image,y_image,b_image,g_image,output_prefix
images/cell_1_mt.png,,images/cell_1_nuc.png,images/cell_1_prot.png,cell1_
images/cell_2_mt.png,,images/cell_2_nuc.png,images/cell_2_prot.png,experiment_A/cell2_
```
- **No `output_folder` column** - use `--output_dir` or `output_dir` in config instead
- `output_prefix` can include subdirectories for organization (e.g., `experiment_A/cell2_`)
- Final output path: `{output_dir}/{output_prefix}{suffix}`
- Supports relative paths, absolute paths, and URLs for images
- Empty fields (like `y_image` above) are valid for unused channels
- Lines starting with `#` are skipped

**Old Format (Deprecated):**
```csv
r_image,y_image,b_image,g_image,output_folder,output_prefix
images/cell_1_mt.png,,images/cell_1_nuc.png,images/cell_1_prot.png,output,cell1_
```
- Still supported for backward compatibility
- Will show deprecation warning
- Each row specifies its own `output_folder`

### Model Configuration Files

Each model variant has a `model_config.yaml` containing:
- `encoder_path`: Path to encoder weights
- `classifier_paths`: List of classifier weight paths
- `model_config`: ViT architecture parameters

### GPU Memory Considerations

For processing large datasets:
- Monitor GPU memory usage and adjust batch size accordingly
- Use `save_attention_maps: False` to save memory
- Consider `embeddings_only: True` if classification not needed
- Set `async_saving: True` with explicit CUDA cache clearing

### Expected Processing Times

With optimized settings (batch_size=600, num_workers=15, GPU):
- ~15.5 hours for 2.56M images
- ~6.1 μs per image
- Compared to ~1 week with original sequential processing

## Troubleshooting

**CUDA out of memory:**
- Reduce `batch_size`
- Set `save_attention_maps: False`
- Enable `embeddings_only: True`

**Slow data loading:**
- Increase `num_workers` (up to CPU core count)
- Increase `prefetch_factor` if memory allows
- Check I/O bottleneck (slow disk or network storage)

**Model files not found:**
- Run with `-u` flag to download models
- Check `models_urls.yaml` for correct URLs
- Verify model directory structure matches channel/type configuration
