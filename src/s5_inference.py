import os
import json
import pickle
import pandas as pd
from Logger import logger

# Mapping of numeric model output labels to human-readable forest cover types
Cover_Type = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}


class Inference:
    def __init__(self):
        """
        Initializes Inference class.
        """
        self.model_path = "artifacts/Model/model.pkl"

        logger.info(f"Loading model from: {self.model_path}")
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load the model.", exc_info=True)
            raise RuntimeError("Model loading failed.") from e

    def predict(self, input_data):
        """
        Predicts the output for a given input sample using the loaded model.
        """
        logger.debug("Starting inference...")

        try:
            # Predict and get the first result (assuming batch prediction)
            prediction = self.model.predict(input_data).item()
            prediction_label = Cover_Type[prediction + 1]
            logger.info(f"Prediction successful: {prediction_label}")
            return prediction_label
        except Exception as e:
            logger.error("Prediction failed.", exc_info=True)
            raise RuntimeError("Prediction failed.") from e
