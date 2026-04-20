import tensorflow as tf
import os


def main():
    # Verify if the model exist
    if not os.path.exists("model.h5"):
        raise FileNotFoundError("model.h5 not found. Run train_model.py first.")

    # Load the model
    model = tf.keras.models.load_model("model.h5")

    # Create the conversor
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply Dynamic Range Quantization (Optimization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    print("model.tflite generated successfully")


if __name__ == "__main__":
    main()
