"""Model downloading and loading functionality."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import boto3
import requests
import yaml
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from config import MODEL_CONFIG_FILE, MODELS_URLS_FILE

logger = logging.getLogger(__name__)


def download_file_from_url(url: str, output_path: str) -> bool:
    """Download a file from HTTP URL.

    Args:
        url: URL to download from
        output_path: Local path to save file

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            logger.info(f"  - {output_path} updated.")
            return True
        else:
            logger.warning(
                f"  - {output_path} url {url} returned status {response.status_code}."
            )
            return False
    except Exception as e:
        logger.warning(f"  - {output_path} download failed: {e}")
        return False


def download_file_from_s3(s3_url: str, output_path: str) -> bool:
    """Download a file from S3.

    Args:
        s3_url: S3 URL (s3://bucket/key)
        output_path: Local path to save file

    Returns:
        True if successful, False otherwise
    """
    try:
        s3 = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))
        url_components = urlparse(s3_url)
        bucket = url_components.netloc
        key = url_components.path[1:]  # Remove leading /
        s3.download_file(bucket, key, output_path)
        logger.info(f"  - {output_path} updated.")
        return True
    except ClientError as e:
        logger.warning(f"  - {output_path} s3 url {s3_url} not working: {e}")
        return False


def download_model_file(url: str, output_path: str) -> bool:
    """Download a model file from URL or S3.

    Args:
        url: URL or S3 path
        output_path: Local path to save file

    Returns:
        True if successful, False otherwise
    """
    if url.startswith("s3://"):
        return download_file_from_s3(url, output_path)
    else:
        return download_file_from_url(url, output_path)


def download_models(
    model_channels: str,
    model_type: str,
    classifier_paths: Optional[List[str]] = None,
    encoder_path: Optional[str] = None,
) -> None:
    """Download model files from configured URLs.

    Args:
        model_channels: Channel configuration (rybg, rbg, etc.)
        model_type: Model type (mae_contrast_supcon_model, etc.)
        classifier_paths: List of classifier file paths (None for embeddings_only)
        encoder_path: Encoder file path
    """
    logger.info("- Downloading models...")

    # Load URL configuration
    with open(MODELS_URLS_FILE, "r") as f:
        url_info = yaml.safe_load(f)

    model_urls = url_info[model_channels][model_type]

    # Download classifiers if needed (not in embeddings_only mode)
    if classifier_paths:
        classifier_urls = model_urls["classifiers"]
        for idx, (url, output_path) in enumerate(
            zip(classifier_urls, classifier_paths)
        ):
            download_model_file(url, output_path)

    # Download encoder
    encoder_url = model_urls["encoder"]
    if encoder_path:
        download_model_file(encoder_url, encoder_path)


def check_models_exist(
    classifier_paths: Optional[List[str]] = None,
    encoder_path: Optional[str] = None,
) -> bool:
    """Check if model files exist locally.

    Args:
        classifier_paths: List of classifier file paths
        encoder_path: Encoder file path

    Returns:
        True if all files exist, False otherwise
    """
    if classifier_paths:
        for path in classifier_paths:
            if not os.path.isfile(path):
                return False

    if encoder_path and not os.path.isfile(encoder_path):
        return False

    return True


def load_model_config(
    model_channels: str,
    model_type: str,
    embeddings_only: bool = False,
) -> Tuple[Optional[List[str]], str, dict]:
    """Load model configuration from YAML file.

    Args:
        model_channels: Channel configuration
        model_type: Model type
        embeddings_only: If True, don't load classifier paths

    Returns:
        Tuple of (classifier_paths, encoder_path, model_config)
        classifier_paths will be None if embeddings_only=True

    Raises:
        FileNotFoundError: If model config file doesn't exist
    """
    config_path = os.path.join(
        "models",
        model_channels,
        model_type,
        MODEL_CONFIG_FILE,
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model configuration not found: {config_path}. "
            f"Please ensure the model files are downloaded or run with -u flag."
        )

    with open(config_path, "r") as f:
        model_config_file = yaml.safe_load(f)

    # Get classifier paths (None if embeddings_only)
    classifier_paths = None
    if not embeddings_only and "classifier_paths" in model_config_file:
        classifier_paths = model_config_file["classifier_paths"]

    encoder_path = Path("~").expanduser() / ".cache" / model_config_file["encoder_path"]
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    model_config = model_config_file.get("model_config")

    return classifier_paths, encoder_path, model_config


def ensure_models_available(
    model_channels: str,
    model_type: str,
    embeddings_only: bool,
    update_model: bool,
) -> Tuple[Optional[List[str]], str, dict]:
    """Ensure model files are available, downloading if necessary.

    Args:
        model_channels: Channel configuration
        model_type: Model type
        embeddings_only: If True, skip classifier download
        update_model: If True, force download even if files exist

    Returns:
        Tuple of (classifier_paths, encoder_path, model_config)

    Raises:
        FileNotFoundError: If model config doesn't exist
    """
    # Load config to get paths
    classifier_paths, encoder_path, model_config = load_model_config(
        model_channels, model_type, embeddings_only
    )

    # Check if download is needed
    needs_update = update_model or not check_models_exist(
        classifier_paths, encoder_path
    )

    if needs_update:
        download_models(
            model_channels,
            model_type,
            classifier_paths,
            encoder_path,
        )

    return classifier_paths, encoder_path, model_config
