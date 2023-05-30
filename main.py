"""Run neural style transfer on every content/style image pair.

Based on "A Neural Algorithm of Artistic Style" by Gatys et al. (2015).
Final project for CSE 455 (Computer Vision), University of Washington.
"""

import argparse
from pathlib import Path

import tensorflow as tf

from models import MODELS, build_feature_extractor
from style_transfer import perform_style_transfer
from utils import load_image, save_image


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU found. Using CPU.")
        return
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=["VGG16", "VGG19"]
    )
    parser.add_argument("--alpha", type=float, default=5, help="content loss weight")
    parser.add_argument("--beta", type=float, default=1, help="style loss weight")
    parser.add_argument(
        "--variation-weight", type=float, default=30, help="total variation loss weight"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=5.0)
    parser.add_argument("--contents", type=Path, default=Path("contents"))
    parser.add_argument("--styles", type=Path, default=Path("styles"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_gpu()
    args.output.mkdir(parents=True, exist_ok=True)

    contents = {p.stem: load_image(str(p)) for p in sorted(args.contents.glob("*.jpg"))}
    styles = {p.stem: load_image(str(p)) for p in sorted(args.styles.glob("*.jpg"))}

    for model_name in args.models:
        print(f"\n{'=' * 60}\nProcessing with {model_name}\n{'=' * 60}")
        extract_features = build_feature_extractor(model_name)

        for content_name, content_image in contents.items():
            for style_name, style_image in styles.items():
                print(f"\nContent: {content_name}, Style: {style_name}")
                output_image = perform_style_transfer(
                    content_image,
                    style_image,
                    extract_features,
                    args.alpha,
                    args.beta,
                    args.variation_weight,
                    args.iterations,
                    args.learning_rate,
                )
                output_path = args.output / f"output_{content_name}_{style_name}_{model_name}.jpg"
                save_image(str(output_path), output_image)
                print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
