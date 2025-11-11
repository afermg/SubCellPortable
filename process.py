"""Main inference orchestration for SubCellPortable."""

import datetime
import logging
import os
import sys
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import inference
from vit_model import ViTPoolClassifier
from dataset import SubCellDataset, collate_fn
from config import SubCellConfig, PATH_LIST_CSV, LOG_FILE
from cli import parse_args
from model_loader import ensure_models_available
from output_handlers import CSVOutputHandler, H5ADOutputHandler, compute_top_predictions

# Set CUDA device ordering
os.environ["DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def setup_logging(log_file: str = LOG_FILE) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_file: Path to log file

    Returns:
        Configured logger
    """
    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        format="%(levelname)s: %(message)s",
        filemode="w",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    logger = logging.getLogger("SubCell inference")
    return logger


def load_config() -> tuple[SubCellConfig, str, str]:
    """Load configuration from multiple sources.

    Configuration priority (highest to lowest):
    1. Command-line arguments
    2. config.yaml file (or custom config file)
    3. Default values

    Returns:
        Tuple of (config object, config_file_path, path_list_path)
    """
    # Parse CLI args (only explicitly provided args)
    cli_args = parse_args()

    # Get config file path (default or custom)
    config_file = cli_args.pop("config", "config.yaml")

    # Get path_list file path (default or custom)
    path_list = cli_args.pop("path_list", PATH_LIST_CSV)

    # Start with defaults
    config_dict = {}

    # Load from config file if exists
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config_dict.update(yaml_config)

    # Override with command-line arguments
    # Note: cli_args only contains explicitly provided arguments (not defaults)
    # This ensures config file values aren't overwritten by CLI defaults
    config_dict.update(cli_args)

    # Create config object (with validation)
    return SubCellConfig.from_dict(config_dict), config_file, path_list


def setup_device(gpu_id: int, logger: logging.Logger) -> torch.device:
    """Set up computing device (CPU or GPU).

    Args:
        gpu_id: GPU ID to use (-1 for CPU)
        logger: Logger instance

    Returns:
        Configured torch device
    """
    if torch.cuda.is_available() and gpu_id != -1:
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Using GPU: cuda:{gpu_id}")
    else:
        if gpu_id != -1:
            logger.warning(f"GPU {gpu_id} requested but CUDA not available. Using CPU.")
        else:
            logger.info("Using CPU")
        device = torch.device("cpu")
    return device


def create_dataloader(
    csv_path: str,
    config: SubCellConfig,
    logger: logging.Logger
) -> DataLoader:
    """Create DataLoader for batch processing.

    Args:
        csv_path: Path to CSV file with image paths
        config: Configuration object
        logger: Logger instance

    Returns:
        Configured DataLoader

    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Input file not found: {csv_path}. "
            f"Please create this file with your image paths."
        )

    dataset = SubCellDataset(
        csv_path,
        config.model_channels
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    args = argparser.parse_args()
    for key in args.__dict__:
        if args.__dict__[key]: config[key] = args.__dict__[key]

# If you want to use a configuration file with your script, add it here
with open("config.yaml", "r") as file:
    config_contents = yaml.safe_load(file)
    if config_contents:
        for key, value in config_contents.items():
            config[key] = value

# Log the start time and the final configuration so you can keep track of what you did
config["log"].info("Start: " + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
config["log"].info("Parameters used:")
config["log"].info(config)
config["log"].info("----------")


try:
    # We load the selected model information
    with open(
        os.path.join(
            "models",
            config["model_channels"],
            config["model_type"],
            "model_config.yaml",
        ),
        "r",
    ) as config_buffer:
        model_config_file = yaml.safe_load(config_buffer)

    classifier_paths = None
    if "classifier_paths" in model_config_file:
        classifier_paths = model_config_file["classifier_paths"]
    encoder_path = model_config_file["encoder_path"]

    needs_update = config["update_model"]
    for curr_classifier in classifier_paths:
        needs_update = needs_update or not os.path.isfile(curr_classifier)
    needs_update = needs_update or not os.path.isfile(encoder_path)

    # Checking for model update
    if needs_update:
        config["log"].info("- Downloading models...")
        with open("models_urls.yaml", "r") as urls_file:
            url_info = yaml.safe_load(urls_file)

            for index, curr_url_info in enumerate(url_info[config["model_channels"]][config["model_type"]]["classifiers"]):
                if curr_url_info.startswith("s3://"):
                    try:
                        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                        urlcomponents = urlparse(curr_url_info)
                        s3.download_file(urlcomponents.netloc, urlcomponents.path[1:], classifier_paths[index])
                        config["log"].info("  - " + classifier_paths[index] + " updated.")
                    except ClientError as e:
                        config["log"].warning("  - " + classifier_paths[index] + " s3 url " + curr_url_info + " not working.")
                else:
                    response = requests.get(curr_url_info)
                    if response.status_code == 200:
                        with open(classifier_paths[index], "wb") as downloaded_file:
                            downloaded_file.write(response.content)
                        config["log"].info("  - " + classifier_paths[index] + " updated.")
                    else:
                        config["log"].warning("  - " + classifier_paths[index] + " url " + curr_url_info + " not found.")

            curr_url_info = url_info[config["model_channels"]][config["model_type"]]["encoder"]
            if curr_url_info.startswith("s3://"):
                try:
                    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                    urlcomponents = urlparse(curr_url_info)
                    s3.download_file(urlcomponents.netloc, urlcomponents.path[1:], encoder_path)
                    config["log"].info("  - " + encoder_path + " updated.")
                except ClientError as e:
                    config["log"].warning("  - " + encoder_path + " s3 url " + curr_url_info + " not working.")
            else:
                response = requests.get(curr_url_info)
                if response.status_code == 200:
                    with open(encoder_path, "wb") as downloaded_file:
                        downloaded_file.write(response.content)
                    config["log"].info("  - " + encoder_path + " updated.")
                else:
                    config["log"].warning("  - " + encoder_path + " url " + curr_url_info + " not found.")

    model_config = model_config_file.get("model_config")
    model = ViTPoolClassifier(model_config)
    model.load_model_dict(encoder_path, classifier_paths)
    model.eval()

    if torch.cuda.is_available() and config["gpu"] != -1:
        device = torch.device("cuda:" + str(config["gpu"]))
    else:
        config["log"].warning("CUDA not available. Using CPU.")
        device = torch.device("cpu")
    model.to(device)

    # if we want to generate a csv result
    if config["create_csv"]:
        final_columns = [
            "id"
        ]
        if classifier_paths:
            final_columns.extend([
                "top_class_name",
                "top_class",
                "top_3_classes_names",
                "top_3_classes",
            ])
            prob_columns = []
            for i in range(31):
                prob_columns.append("prob" + "%02d" % (i,))
            final_columns.extend(prob_columns)
            feat_columns = []
        for i in range(1536):
            feat_columns.append("feat" + "%04d" % (i,))
        final_columns.extend(feat_columns)
        df = pd.DataFrame(columns=final_columns)

    # We iterate over each set of images to process
    if os.path.exists("./path_list.csv"):
        path_list = open("./path_list.csv", "r")
        for curr_set in path_list:

            if curr_set.strip() != "" and not curr_set.startswith("#"):
                curr_set_arr = curr_set.split(",")
                # We create the output folder
                os.makedirs(curr_set_arr[4].strip(), exist_ok=True)
                # We load the images as numpy arrays
                cell_data = []
                if "r" in config["model_channels"]:
                    cell_data.append([image_utils.read_grayscale_image(curr_set_arr[0].strip())])
                if "y" in config["model_channels"]:
                    cell_data.append([image_utils.read_grayscale_image(curr_set_arr[1].strip())])
                if "b" in config["model_channels"]:
                    cell_data.append([image_utils.read_grayscale_image(curr_set_arr[2].strip())])
                if "g" in config["model_channels"]:
                    cell_data.append([image_utils.read_grayscale_image(curr_set_arr[3].strip())])

                # We run the model in inference
                embedding, probabilities = inference.run_model(
                    model,
                    cell_data,
                    device,
                    os.path.join(curr_set_arr[4], curr_set_arr[5].strip()),
                )

        # Initialize output handlers
        csv_handler = None
        h5ad_handler = None

        if config.create_csv:
            csv_handler = CSVOutputHandler(has_classifier=classifier_paths is not None)

        if config.output_format == "combined":
            if uses_old_format:
                # Old format: use first row's output_folder
                first_item = dataloader.dataset.data_list[0]
                h5ad_handler = H5ADOutputHandler(first_item["output_folder"])
            else:
                # New format: use output_dir
                h5ad_handler = H5ADOutputHandler(config.output_dir)

        # Process batches
        pending_executors = []
        pending_futures = []

        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            images = batch["images"]
            output_prefixes = batch["output_prefixes"]

            # Prepare output paths based on format
            if uses_old_format:
                # Old format: use per-row output_folder from CSV
                output_folders = batch["output_folders"]
                # Create output directories
                for output_folder in set(output_folders):
                    os.makedirs(output_folder, exist_ok=True)
                # Prepare output paths
                output_paths = [
                    os.path.join(output_folders[i], output_prefixes[i])
                    for i in range(len(output_folders))
                ]
            else:
                # New format: use config.output_dir + output_prefix
                # output_prefix can include subdirectories (e.g., "experiment_A/cell1_")
                output_paths = []
                for prefix in output_prefixes:
                    full_path = os.path.join(config.output_dir, prefix)
                    # Create subdirectories if prefix contains them
                    output_dir_for_file = os.path.dirname(full_path)
                    if output_dir_for_file:
                        os.makedirs(output_dir_for_file, exist_ok=True)
                    output_paths.append(full_path)

            # Run inference
            try:
                inference_result = inference.run_model(
                    model, images, device, output_paths,
                    save_attention_maps=config.save_attention_maps,
                    embeddings_only=config.embeddings_only,
                    output_format=config.output_format,
                    async_saving=config.async_saving
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("=" * 60)
                    logger.error("❌ CUDA OUT OF MEMORY ERROR")
                    logger.error("=" * 60)
                    logger.error(f"Current configuration:")
                    logger.error(f"  batch_size: {config.batch_size}")
                    logger.error(f"  num_workers: {config.num_workers}")
                    logger.error(f"  GPU: {config.gpu}")
                    logger.error("")
                    logger.error("💡 Suggestions to fix:")
                    logger.error(f"  1. Reduce batch_size: -b {max(1, config.batch_size // 2)}")
                    logger.error(f"  2. Use fewer workers: -w {max(1, config.num_workers // 2)}")
                    logger.error("  3. Switch to CPU: -g -1 (slower but uses RAM instead)")
                    logger.error("  4. Close other GPU applications")
                    logger.error("=" * 60)
                raise

            # Handle different return formats
            if config.output_format == "combined":
                batch_results = inference_result
            elif config.async_saving:
                batch_results, (executor, futures) = inference_result
                pending_executors.append(executor)
                pending_futures.extend(futures)
            else:
                batch_results = inference_result

            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Extract embeddings and probabilities
            embeddings = [result[0] for result in batch_results]
            probabilities_list = [result[1] for result in batch_results]

            # Add to output handlers
            if csv_handler:
                csv_handler.add_batch(output_prefixes, embeddings, probabilities_list)

            if h5ad_handler:
                h5ad_handler.add_batch(output_prefixes, embeddings, probabilities_list)

            # Log progress
            process_batch_results(
                batch_results,
                output_prefixes,
                classifier_paths,
                config
            )

        # Wait for async saves to complete
        if config.async_saving and pending_futures:
            logger.info(f"Waiting for {len(pending_futures)} async save operations...")
            import concurrent.futures
            concurrent.futures.wait(pending_futures)
            for executor in pending_executors:
                executor.shutdown(wait=True)
            logger.info("All async saves completed")

        # Save accumulated outputs
        if csv_handler:
            if uses_old_format or not config.output_dir:
                # Old format or no output_dir: save to CWD
                csv_handler.save("result.csv")
            else:
                # New format: save to output_dir
                csv_path = os.path.join(config.output_dir, "result.csv")
                csv_handler.save(csv_path)

        if h5ad_handler:
            h5ad_handler.save(embeddings_only=config.embeddings_only)

        # Calculate timing statistics
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time
        total_images = len(dataloader.dataset)
        images_per_sec = total_images / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0

        # Log success summary
        logger.info("-" * 60)
        logger.info("✓ Processing completed successfully")
        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Average speed: {images_per_sec:.2f} images/sec")
        logger.info(f"Average time per image: {elapsed.total_seconds()/total_images:.4f} sec")

        # Log output location
        if config.output_dir:
            logger.info(f"Output saved to: {os.path.abspath(config.output_dir)}")
            if config.create_csv:
                logger.info(f"  - result.csv")
            if config.output_format == "combined":
                logger.info(f"  - embeddings.h5ad")
            logger.info(f"  - log.txt")
            if config.output_format == "individual":
                logger.info(f"  - {total_images} embedding files (*_embedding.npy)")
                if not config.embeddings_only:
                    logger.info(f"  - {total_images} probability files (*_probabilities.npy)")
                if config.save_attention_maps:
                    logger.info(f"  - {total_images} attention maps (*_attention_map.png)")

        # Clean up
        del dataloader

    except FileNotFoundError as e:
        logger.error("-" * 60)
        logger.error(f"✗ File not found: {e}")
        logger.error("Please check that all input files exist.")
        raise
    except ValueError as e:
        logger.error("-" * 60)
        logger.error(f"✗ Configuration error: {e}")
        logger.error("Please check your configuration and CSV format.")
        raise
    except RuntimeError as e:
        logger.error("-" * 60)
        logger.error(f"✗ Runtime error: {e}")
        if "out of memory" in str(e).lower():
            logger.error("Suggestion: Try reducing batch_size in config.yaml")
        raise
    except Exception as e:
        logger.error("-" * 60)
        logger.error(f"✗ Unexpected error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

    finally:
        end_time = datetime.datetime.now()
        logger.info("=" * 60)
        logger.info(f"End: {end_time.strftime('%Y/%m/%d %H:%M:%S')}")
        logger.info("=" * 60)


if __name__ == "__main__":
    run_inference()
