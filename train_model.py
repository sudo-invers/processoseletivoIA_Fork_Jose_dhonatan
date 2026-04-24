import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

tf.random.set_seed(42)  # Define a fixed seed for consistency of results


def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalizes pixels from 0-255 to 0-1
    # stabilizes training and speeds up convergence
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255

    # (batch, height, width, channels)
    # MNIST is grayscale so, 1 channel
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    # 4 Epcochs is enought, in tests, i have get a accuracy of ~99%
    model.fit(train_images, train_labels, epochs=4, batch_size=32, verbose=2)

    # Evaluate model performance on unseen test data
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")

    model.save("model.h5")


if __name__ == "__main__":
    main()
