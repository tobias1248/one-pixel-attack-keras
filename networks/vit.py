import os

import numpy as np


class ViT:
    def __init__(self, batch_size=128, load_weights=True, model_path_env="VIT_MODEL_PATH"):
        self.name = "vit"
        self.batch_size = batch_size
        self.model_path_env = model_path_env
        self.model_filename = None
        self._model = None

        if load_weights:
            self._load_model()

    def _load_model(self):
        model_path = os.getenv(self.model_path_env)
        if not model_path:
            raise ValueError(
                "ViT model path not set. Please export "
                f"{self.model_path_env}=/path/to/cifar10_concolic_transformer.h5"
            )

        try:
            from tensorflow import keras as tfk
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required to load the ViT model. "
                "Install TensorFlow (with MultiHeadAttention support) and retry."
            ) from exc

        self._model = tfk.models.load_model(model_path, compile=False)
        self.model_filename = model_path

    def count_params(self):
        return self._model.count_params()

    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype("float32")
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        return imgs

    def predict(self, img):
        processed = self.color_process(img)
        return self._model.predict(processed, batch_size=self.batch_size)

    def predict_one(self, img):
        return self.predict(img)[0]
