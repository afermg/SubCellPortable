import argparse
import datetime
import logging
import os
import sys
import pandas as pd
import requests
import torch
import yaml
from skimage.io import imread
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from urllib.parse import urlparse

import inference
from vit_model import ViTPoolClassifier


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
                    cell_data.append([imread(curr_set_arr[0].strip(), as_gray=True)])
                if "y" in config["model_channels"]:
                    cell_data.append([imread(curr_set_arr[1].strip(), as_gray=True)])
                if "b" in config["model_channels"]:
                    cell_data.append([imread(curr_set_arr[2].strip(), as_gray=True)])
                if "g" in config["model_channels"]:
                    cell_data.append([imread(curr_set_arr[3].strip(), as_gray=True)])

                # We run the model in inference
                embedding, probabilities = inference.run_model(
                    model,
                    cell_data,
                    device,
                    os.path.join(curr_set_arr[4], curr_set_arr[5].strip()),
                )

                if classifier_paths:
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

                # Save results in csv format
                if config["create_csv"]:
                    new_row = []
                    new_row.append(curr_set_arr[5].strip())
                    if classifier_paths:
                        new_row.append(max_location_name)
                        new_row.append(max_location_class)
                        new_row.append(max_3_location_names)
                        new_row.append(",".join(map(str, max_3_location_classes)))
                        new_row.extend(probabilities)
                    new_row.extend(embedding)
                    df.loc[len(df.index)] = new_row

                log_message = "- Saved results for " + curr_set_arr[5].strip()
                if classifier_paths:
                    log_message = log_message + ", locations predicted [" + max_3_location_names + "]"
                config["log"].info(log_message)

        if config["create_csv"]:
            df.to_csv("result.csv", index=False)
except Exception as e:
    config["log"].error("- " + str(e))

config["log"].info("----------")
config["log"].info("End: " + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
