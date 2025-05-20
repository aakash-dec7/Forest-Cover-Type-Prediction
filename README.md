# Forest Cover Type Prediction

This repository focuses on predicting forest cover types based on cartographic variables from the **Roosevelt National Forest** in **Northern Colorado**. Using Machine Learning algorithms, the model classifies 30m x 30m land patches into one of seven forest cover types such as Spruce/Fir, Aspen, or Douglas-fir.

## Model Architecture

This project leverages three powerful ensemble learning classifiers - `RandomForestClassifier`, `XGBClassifier`, and `AdaBoostClassifier` - to achieve robust performance on the given classification task. Each model has been configured with a well-curated hyperparameter grid for optimal tuning using techniques like `GridSearchCV`.

- ### RandomForestClassifier

    Random Forest is an ensemble of decision trees, where each tree is trained on a bootstrapped sample of the dataset and uses a random subset of features for splitting. It improves accuracy by reducing overfitting and variance.

    **Hyperparameters tuned:**

  - `n_estimators`: Number of trees in the forest.

  - `max_depth`: Maximum depth of each tree.

  - `min_samples_split`: Minimum samples required to split a node.

  - `min_samples_leaf`: Minimum samples required at a leaf node.

  - `max_features`: Number of features to consider when looking for the best split (`sqrt` or `log2`).

- ### XGBClassifier (XGBoost)

    XGBoost (Extreme Gradient Boosting) is a highly efficient and scalable implementation of gradient boosting. It builds models in a sequential manner where each new tree corrects errors made by the previous ones.

    **Hyperparameters tuned:**

  - `n_estimators`: Number of boosting rounds.

  - `max_depth`: Maximum tree depth for base learners.

  - `learning_rate`: Step size shrinkage to prevent overfitting.

  - `gamma`: Minimum loss reduction required to make a split.

  - `reg_lambda`: L2 regularization term to reduce model complexity.

- ### AdaBoostClassifier

    AdaBoost (Adaptive Boosting) combines multiple weak learners (typically decision stumps) in a way that focuses more on the misclassified instances by adjusting their weights iteratively.

    **Hyperparameters tuned:**

  - `n_estimators`: Number of weak learners to train.

  - `learning_rate`: Shrinks the contribution of each learner to control overfitting.

## Dataset

This dataset is a structured tabular dataset used for forest cover type prediction based on various topographic and soil features. Each row represents a unique location in the Roosevelt National Forest of Northern Colorado, USA. The goal is to predict the `Cover_Type` (categorical label from 1 to 7) indicating the type of forest cover for that location.

**Key Features:**

The dataset includes both numerical and one-hot encoded categorical features:

- **Topographic Features:**

  - `Elevation`: Elevation in meters.

  - `Aspect`: Compass direction that a slope faces (0 to 360 degrees).

  - `Slope`: Gradient of the slope.

  - `Horizontal_Distance_To_Hydrology`: Horizontal distance to the nearest surface water features (lakes, streams, etc.).

  - `Vertical_Distance_To_Hydrology`: Vertical distance to the nearest surface water features.

  - `Horizontal_Distance_To_Roadways`: Distance to the nearest roadway.

  - `Horizontal_Distance_To_Fire_Points`: Distance to the nearest wildfire ignition point.

  - `Hillshade_9am`, `Hillshade_Noon`, `Hillshade_3pm`: Measure of sunlight at different times of day, based on slope and aspect.

- **Categorical Features:**

  - `Wilderness_Area1` to `Wilderness_Area4`: One-hot encoded indicators of the wilderness area.

  - `Soil_Type1` to `Soil_Type40`: One-hot encoded indicators of soil types.

- **Target Feature:**

  - `Cover_Type`: Integer class label from 1 to 7 representing the forest cover type:

    - 1: Spruce/Fir

    - 2: Lodgepole Pine

    - 3: Ponderosa Pine

    - 4: Cottonwood/Willow

    - 5: Aspen

    - 6: Douglas-fir

    - 7: Krummholz

## Model Training Metrics

The following results summarize the best cross-validation scores and corresponding hyperparameters for each classifier:

```json
[
    {
        "model": "RandomForestClassifier",
        "best_params": {
            "classifier__max_depth": 20,
            "classifier__max_features": "sqrt",
            "classifier__min_samples_leaf": 1,
            "classifier__min_samples_split": 2,
            "classifier__n_estimators": 100
        },
        "best_score": 0.8433490520969414
    },
    {
        "model": "XGBClassifier",
        "best_params": {
            "classifier__gamma": 0,
            "classifier__learning_rate": 0.3,
            "classifier__max_depth": 9,
            "classifier__n_estimators": 100,
            "classifier__reg_lambda": 1.5
        },
        "best_score": 0.8544976618301927
    },
    {
        "model": "AdaBoostClassifier",
        "best_params": {
            "classifier__learning_rate": 0.5,
            "classifier__n_estimators": 100
        },
        "best_score": 0.5416670610495824
    }
]
```

## Model Evaluation Metrics

The following results summarize the model evaluation performance for the **best model** from `GridSearchCV`:

```json
{
    "accuracy": 0.5218253968253969,
    "precision": 0.5078715666259249,
    "recall": 0.5218253968253969,
    "f1_score": 0.4964743629569057,
}
```

## Installation

Clone the repository:

```sh
git clone https://github.com/aakash-dec7/Forest-Cover-Type-Prediction.git
cd Forest-Cover-Type-Prediction
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Initiate DVC

```sh
dvc init
```

### Run the pipeline

```sh
dvc repro
```

The pipeline automatically launches the Flask application at:

```text
http://localhost:3000/
```

## Conclusion

This project demonstrates the effective use of ensemble learning methods - Random Forest, XGBoost, and AdaBoost - for multiclass classification of forest cover types using rich cartographic data. Among the models tested, XGBoost outperformed others in cross-validation, showing its strength in handling structured datasets with complex feature interactions. While AdaBoost showed relatively lower performance, its inclusion highlights the comparative strengths of boosting techniques. With a well-defined pipeline managed by DVC, reproducibility and scalability are integral parts of the project, making it easy to retrain or improve the model as needed.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
