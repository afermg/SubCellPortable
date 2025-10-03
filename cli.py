"""Command-line interface for SubCellPortable."""

import argparse
import sys
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="SubCellPortable: Run SubCell model inference on microscopy images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument(
        "-c",
        "--model_channels",
        help="Channel images to be used",
        choices=["rybg", "rbg", "ybg", "bg"],
        default="rybg",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model_type",
        help="Model type to be used",
        choices=["mae_contrast_supcon_model", "vit_supcon_model"],
        default="mae_contrast_supcon_model",
        type=str,
    )
    parser.add_argument(
        "-u",
        "--update_model",
        action="store_true",
        help="Download/update the selected model files (default: False)",
    )

    # Output configuration
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory for all results (required for new CSV format without output_folder column)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-csv",
        "--create_csv",
        action="store_true",
        help="Generate a combined CSV of probabilities and embeddings (default: False)",
    )
    parser.add_argument(
        "--embeddings_only",
        action="store_true",
        help="Only generate embeddings, skip classification (faster)",
    )
    parser.add_argument(
        "--output_format",
        choices=["individual", "combined"],
        default="combined",
        help="Output format: individual (.npy files) or combined (.h5ad file)",
    )
    parser.add_argument(
        "--save_attention_maps",
        action="store_true",
        help="Save attention map images (default: False)",
    )

    # Performance configuration
    parser.add_argument(
        "-g",
        "--gpu",
        help="The GPU id to use [0, 1, 2, 3...]. -1 for CPU usage",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Batch size for processing",
        default=128,
        type=int,
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        help="Number of workers for data loading",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-p",
        "--prefetch_factor",
        help="Prefetch factor for data loading",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--async_saving",
        action="store_true",
        help="Save individual files asynchronously (only for individual format, not combined; default: False)",
    )

    # Logging
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose logging (quiet mode)",
    )

    return parser


def parse_args(args: Optional[list] = None) -> dict:
    """Parse command-line arguments.

    Args:
        args: List of arguments to parse. If None, uses sys.argv

    Returns:
        Dictionary of parsed arguments
    """
    parser = create_parser()

    # Only parse if there are actual arguments beyond the script name
    if args is None:
        args = sys.argv[1:]

    if len(args) == 0:
        # No arguments provided, return empty dict to use defaults
        return {}

    parsed = parser.parse_args(args)
    return vars(parsed)
