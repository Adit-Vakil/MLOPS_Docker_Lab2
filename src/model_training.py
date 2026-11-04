import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():
    tf.keras.utils.set_random_seed(42)

    features, labels = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        features.astype("float32"),
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    normalization = tf.keras.layers.Normalization()
    normalization.adapt(x_train)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        normalization,
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=50, verbose=0)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save("my_model.keras")
    print(f"Holdout loss: {loss:.4f}, accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
