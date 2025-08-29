import argparse
import datetime
import logging
import os
import sys
import pandas as pd
import requests
import torch
import yaml
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from urllib.parse import urlparse
from torch.utils.data import DataLoader
from tqdm import tqdm

import inference
import image_utils
from vit_model import ViTPoolClassifier
from dataset import SubCellDataset, collate_fn


os.environ["DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# This is the log configuration. It will log everything to a file AND the console
logging.basicConfig(
    filename="log.txt",
    encoding="utf-8",
    format="%(levelname)s: %(message)s",
    filemode="w",
    level=logging.INFO,
)
console = logging.StreamHandler()
logging.getLogger().addHandler(console)
logger = logging.getLogger("SubCell inference")

# This is the general configuration variable. We are going to use the special key "log" in the dictionary to use the log in our code
config = {"log": logger}

# If you want to use constants with your script, add them here
config["model_channels"] = "rybg"
config["model_type"] = "mae_contrast_supcon_model"
config["update_model"] = False
config["create_csv"] = False
config["gpu"] = -1

# If you want to use command line parameters with your script, add them here
if len(sys.argv) > 1:
    argparser = argparse.ArgumentParser(
        description="Please input the following parameters"
    )
    argparser.add_argument(
        "-c",
        "--model_channels",
        help="channel images to be used [rybg, rbg, ybg, bg]",
        default="rybg",
        type=str,
    )
    argparser.add_argument(
        "-t",
        "--model_type",
        help="model type to be used [mae_contrast_supcon_model, vit_supcon_model]",
        default="mae_contrast_supcon_model",
        type=str,
    )
    argparser.add_argument(
        '-u', '--update_model',
        action='store_true',
        help='download/update the selected model files'
    )
    argparser.add_argument(
        '--no-update_model',
        action='store_false',
        dest='update_model',
        help='do not download/update the selected model files'
    )
    argparser.add_argument(
        '-csv', '--create_csv',
        action='store_true',
        help='generate a combined CSV of probabilities and embeddings'
    )
    argparser.add_argument(
        '--no-create_csv',
        action='store_false',
        dest='create_csv',
        help='do not generate a combined CSV'
    )
    argparser.add_argument(
        "-g",
        "--gpu",
        help="the GPU id to use [0, 1, 2, 3]. -1 for CPU usage",
        default=-1,
        type=int,
    )
    argparser.add_argument(
        "-b",
        "--batch_size",
        help="batch size for processing (default: 512)",
        default=256,
        type=int,
    )
    argparser.add_argument(
        "-w",
        "--num_workers",
        help="number of workers for data loading (default: 4)",
        default=4,
        type=int,
    )
    argparser.add_argument(
        "-p",
        "--prefetch_factor",
        help="prefetch factor for data loading (default: 2)",
        default=2,
        type=int,
    )
    argparser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress verbose logging (quiet mode)",
    )

    args = argparser.parse_args()
    config = config | args.__dict__

# If you want to use a configuration file with your script, add it here
with open("config.yaml", "r") as file:
    config_contents = yaml.safe_load(file)
    if config_contents:
        config = config | config_contents

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

    # Skip classifier loading if embeddings_only mode
    classifier_paths = None
    if not config.get("embeddings_only", False) and "classifier_paths" in model_config_file:
        classifier_paths = model_config_file["classifier_paths"]
    encoder_path = model_config_file["encoder_path"]

    needs_update = config["update_model"]
    if classifier_paths:
        for curr_classifier in classifier_paths:
            needs_update = needs_update or not os.path.isfile(curr_classifier)
    needs_update = needs_update or not os.path.isfile(encoder_path)

    # Checking for model update
    if needs_update:
        config["log"].info("- Downloading models...")
        with open("models_urls.yaml", "r") as urls_file:
            url_info = yaml.safe_load(urls_file)

            # Only download classifiers if not in embeddings_only mode
            if classifier_paths:
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
    
    # Pass empty list for classifier_paths if None (embeddings_only mode)
    classifier_paths_for_loading = classifier_paths if classifier_paths is not None else []
    model.load_model_dict(encoder_path, classifier_paths_for_loading)
    model.eval()
    
    # Log what mode we're running in
    if config.get("embeddings_only", False):
        config["log"].info("🔍 Running in EMBEDDINGS ONLY mode - no classification")
    else:
        config["log"].info("🎯 Running in FULL mode - embeddings + classification")

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

    # We use DataLoader for efficient batch processing
    if os.path.exists("./path_list.csv"):
        # Create dataset and dataloader
        dataset = SubCellDataset("./path_list.csv", config["model_channels"], minmax_norm=config.get("minmax_norm", False))
        dataloader = DataLoader(
            dataset,
            batch_size=config.get("batch_size", 256),
            num_workers=config.get("num_workers", 4),
            prefetch_factor=config.get("prefetch_factor", 2),
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.get("num_workers", 4) > 0,
        )

        config["log"].info(
            f"Processing {len(dataset)} images in batches of {config.get('batch_size', 8)}"
        )
        config["log"].info(
            f"Using {config.get('num_workers', 4)} workers for data loading"
        )

        # Initialize list to track async save operations
        pending_saves = []
        
        # Initialize H5AD data collection (if using H5AD format)
        all_embeddings = []
        all_probabilities = []
        all_image_names = []
        all_output_folders = []
        
        # Wrap DataLoader with tqdm for progress bar
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Processing batches", unit="batch")
        ):
            images = batch["images"]
            output_folders = batch["output_folders"]
            output_prefixes = batch["output_prefixes"]
            original_items = batch["original_items"]

            # Create output directories
            for output_folder in output_folders:
                os.makedirs(output_folder, exist_ok=True)

            # Prepare output paths
            output_paths = [
                os.path.join(output_folders[i], output_prefixes[i])
                for i in range(len(output_folders))
            ]

            # Run batch inference
            inference_result = inference.run_model(
                model, images, device, output_paths, 
                save_attention_maps=config.get("save_attention_maps", True),
                embeddings_only=config.get("embeddings_only", False),
                output_format=config.get("output_format", "individual"),
                async_saving=config.get("async_saving", False)
            )
            
            # Handle different output formats and modes
            if config.get("output_format", "individual") == "h5ad":
                # H5AD: collect data without saving
                batch_results = inference_result
                
                # Collect embeddings and metadata for final H5AD save
                for i, (embedding, probabilities) in enumerate(batch_results):
                    all_embeddings.append(embedding)
                    if probabilities is not None:
                        all_probabilities.append(probabilities)
                    all_image_names.append(output_prefixes[i])
                    all_output_folders.append(output_folders[i])
                    
            elif config.get("async_saving", False):
                # Async individual files
                batch_results, save_future = inference_result
                pending_saves.append(save_future)
            else:
                # Sync individual files
                batch_results = inference_result
            
            # Free GPU memory after each batch (critical for async mode)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Process results for each image in batch (if needed)
            batch_rows = []
            
            for i, (embedding, probabilities) in enumerate(batch_results):
                output_prefix = output_prefixes[i]

                # Classification computation (only if not embeddings_only mode)
                max_location_name = None
                max_3_location_names = None
                if classifier_paths and not config.get("embeddings_only", False):
                    curr_probs_l = probabilities.tolist()
                    max_location_class = curr_probs_l.index(max(curr_probs_l))
                    max_location_name = inference.CLASS2NAME[max_location_class]
                    max_3_location_classes = sorted(
                        range(len(curr_probs_l)), key=lambda sub: curr_probs_l[sub]
                    )[-3:]
                    max_3_location_classes.reverse()
                    max_3_location_names = (
                        inference.CLASS2NAME[max_3_location_classes[0]]
                        + ","
                        + inference.CLASS2NAME[max_3_location_classes[1]]
                        + ","
                        + inference.CLASS2NAME[max_3_location_classes[2]]
                    )

                # Prepare row for batch CSV operations
                if config["create_csv"]:
                    new_row = []
                    new_row.append(output_prefix)
                    if classifier_paths:
                        new_row.append(max_location_name)
                        new_row.append(max_location_class)
                        new_row.append(max_3_location_names)
                        new_row.append(",".join(map(str, max_3_location_classes)))
                        new_row.extend(probabilities)
                    new_row.extend(embedding)
                    batch_rows.append(new_row)

                # Logging operations
                if not config.get("quiet", False):
                    log_message = f"- Saved results for {output_prefix}"
                    if classifier_paths and max_3_location_names:
                        log_message = (
                            log_message
                            + ", locations predicted ["
                            + max_3_location_names
                            + "]"
                        )
                    config["log"].info(log_message)

            # Batch CSV operations
            if config["create_csv"] and batch_rows:
                batch_df = pd.DataFrame(batch_rows, columns=df.columns)
                df = pd.concat([df, batch_df], ignore_index=True)

        # Handle final saving based on output format
        if config.get("output_format", "individual") == "h5ad":
            # Save single H5AD file with all data in the proper output directory
            config["log"].info(f"Saving H5AD file with {len(all_embeddings)} embeddings...")
            
            import numpy as np
            import h5py
            
            # Use the output folder from CSV directly - it's the same for all images
            output_folder = all_output_folders[0] if all_output_folders else "."
            h5ad_path = os.path.join(output_folder, "embeddings.h5ad")
            
            # Create proper H5AD file
            embeddings_array = np.stack(all_embeddings)
            
            with h5py.File(h5ad_path, 'w') as f:
                # Save embeddings as the main data matrix (following AnnData convention)
                f.create_dataset('X', data=embeddings_array)
                
                # Save observation names (image names)
                obs_names = np.array(all_image_names, dtype='S')
                f.create_dataset('obs/index', data=obs_names)
                
                # Save probabilities if available
                if all_probabilities:
                    probabilities_array = np.stack(all_probabilities)
                    f.create_dataset('obsm/probabilities', data=probabilities_array)
                
                # Add metadata
                f.attrs['n_obs'] = len(all_embeddings)
                f.attrs['n_vars'] = embeddings_array.shape[1]
                f.attrs['created_by'] = 'SubCellPortable'
                f.attrs['embeddings_only'] = config.get("embeddings_only", False)
            
            config["log"].info(f"H5AD file saved: {h5ad_path}")
            config["log"].info(f"Shape: {embeddings_array.shape}")
            config["log"].info(f"Contains: embeddings, image_names" + (", probabilities" if all_probabilities else ""))
                
        elif config.get("async_saving", False) and pending_saves:
            # Wait for all async saves to complete
            config["log"].info(f"Waiting for {len(pending_saves)} async save operations to complete...")
            for save_future in pending_saves:
                save_future.result()  # Wait for completion
            config["log"].info("All async saves completed")

        if config["create_csv"]:
            df.to_csv("result.csv", index=False)

        print(f"Processed {len(dataset)} images successfully")
        
        # Clean up DataLoader workers to prevent hanging
        del dataloader

except Exception as e:
    config["log"].error("- " + str(e))

config["log"].info("----------")
config["log"].info("End: " + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
