from typing import Any, Optional

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class SoftmaxClassifier(tf.Module):
    def __init__(self, input_dim, num_classes, seed=42) -> None:
        super().__init__()
        initializer = tf.keras.initializers.TruncatedNormal(seed=seed)
        self.W = tf.Variable(initializer(shape=(input_dim, num_classes), dtype=tf.float32), name="weights")
        self.b = tf.Variable(initializer(shape=(num_classes,), dtype=tf.float32), name="bias")

    def __call__(self, x):
        logits = tf.matmul(x, self.W) + self.b
        return tf.nn.softmax(logits)


def train_model(model_name: Optional[str] = None) -> Any:
    # Load and preprocess Iris dataset
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y).astype(np.float32)

    X_train, _, y_train, _ = train_test_split(X, y_onehot, test_size=0.2, random_state=42, shuffle=False)
    X_train = tf.constant(X_train)
    y_train = tf.constant(y_train)

    # Initialize model and optimizer
    model = SoftmaxClassifier(input_dim=4, num_classes=3)
    optimizer = tf.optimizers.SGD(learning_rate=0.1)

    # Training loop
    for _ in range(2):
        with tf.GradientTape() as tape:
            y_pred = model(X_train)
            loss = tf.reduce_mean(tf.square(y_pred - y_train))
        grads = tape.gradient(loss, [model.W, model.b])
        optimizer.apply_gradients(zip(grads, [model.W, model.b]))

    return model


def predict_result(model) -> Any:
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y).astype(np.float32)

    _, X_test, _, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    X_test = tf.constant(X_test)
    y_pred = model(X_test)
    return y_pred


if __name__ == "__main__":
    model = train_model()
    __return__ = model
