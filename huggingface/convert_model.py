#!/usr/bin/env python3
"""
Convert TensorFlow.js model to Keras H5 format.
Run this locally before uploading to Hugging Face.

Usage:
    pip install tensorflowjs tensorflow
    python convert_model.py ../public/models/model18cls ./model18cls
"""

import sys
import os

def convert_tfjs_to_keras(input_path, output_path):
    """Convert tfjs model to Keras H5 format"""
    # Import here to avoid issues if not installed
    import tensorflowjs as tfjs
    import tensorflow as tf

    print(f"Converting {input_path} to Keras format...")

    # Load tfjs model
    model_json = os.path.join(input_path, "model.json")
    model = tfjs.converters.load_keras_model(model_json)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save as H5
    h5_path = os.path.join(output_path, "model.h5")
    model.save(h5_path)
    print(f"Saved to {h5_path}")

    # Also save as SavedModel for better compatibility
    savedmodel_path = os.path.join(output_path, "saved_model")
    model.save(savedmodel_path, save_format='tf')
    print(f"Saved to {savedmodel_path}")

    print("Conversion complete!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_model.py <input_tfjs_path> <output_path>")
        print("Example: python convert_model.py ../public/models/model18cls ./model18cls")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    convert_tfjs_to_keras(input_path, output_path)
