import argparse

from model_loader import ensure_models_available


def main():
    parser = argparse.ArgumentParser(description="Download models.")
    parser.add_argument(
        "--model-channels",
        type=str,
        required=True,
        help="List of model channels to use for downloading (e.g., rybg).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="The type of model to download (e.g., mae_contrast_supcon_model, vit_supcon_model).",
    )
    args = parser.parse_args()

    ensure_models_available(args.model_channels, args.model_type, True, False)


if __name__ == "__main__":
    main()
