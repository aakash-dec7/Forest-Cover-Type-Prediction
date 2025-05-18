import pandas as pd
from Logger import logger
from src.s5_inference import Inference
from flask import Flask, render_template, request

app = Flask(__name__)

# Load Inference Model
inference = Inference()


@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html", cover_type_prediction="Forest Cover Type")


@app.route("/predict", methods=["POST"])
def analyze():
    try:
        form = request.form

        selected_wilderness = form.get("Wilderness_Area")
        selected_soil = form.get("Soil_Type")

        wilderness_dict = {f"Wilderness_Area{i}": 0 for i in range(1, 5)}
        soil_dict = {f"Soil_Type{i}": 0 for i in range(1, 41)}

        if selected_wilderness and selected_wilderness.isdigit():
            wilderness_dict[f"Wilderness_Area{int(selected_wilderness)}"] = 1

        if selected_soil and selected_soil.isdigit():
            soil_dict[f"Soil_Type{int(selected_soil)}"] = 1

        input_dict = {
            "Elevation": float(form.get("Elevation", 0)),
            "Aspect": float(form.get("Aspect", 0)),
            "Slope": float(form.get("Slope", 0)),
            "Horizontal_Distance_To_Hydrology": float(
                form.get("Horizontal_Distance_To_Hydrology", 0)
            ),
            "Vertical_Distance_To_Hydrology": float(
                form.get("Vertical_Distance_To_Hydrology", 0)
            ),
            "Horizontal_Distance_To_Roadways": float(
                form.get("Horizontal_Distance_To_Roadways", 0)
            ),
            "Hillshade_9am": float(form.get("Hillshade_9am", 0)),
            "Hillshade_Noon": float(form.get("Hillshade_Noon", 0)),
            "Hillshade_3pm": float(form.get("Hillshade_3pm", 0)),
            "Horizontal_Distance_To_Fire_Points": float(
                form.get("Horizontal_Distance_To_Fire_Points", 0)
            ),
        }

        input_dict.update(wilderness_dict)
        input_dict.update(soil_dict)

        input_data = pd.DataFrame([input_dict])

        # Prediction using the Inference model
        prediction = inference.predict(input_data=input_data.values)

        return render_template("index.html", cover_type_prediction=str(prediction))

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return render_template(
            "index.html", cover_type_prediction="An error occurred. Please try again later."
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
