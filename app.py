from flask import Flask, render_template, request, send_file
from markupsafe import Markup
import pandas as pd
from utils.fertilizer import fertilizer_dict
import numpy as np
import os
from keras.preprocessing import image
from keras.models import load_model
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# Load the crop recommendation model
crop_recommendation_model_path = 'Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

app = Flask(__name__)
@ app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():

    crop_name = str(request.form['cropname'])
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    df = pd.read_csv('Data/Crop_NPK.csv')

    N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = N_desired- N_filled
    p = P_desired - P_filled
    k = K_desired - K_filled

    if n < 0:
        key1 = "NHigh"
    elif n > 0:
        key1 = "Nlow"
    else:
        key1 = "NNo"

    if p < 0:
        key2 = "PHigh"
    elif p > 0:
        key2 = "Plow"
    else:
        key2 = "PNo"

    if k < 0:
        key3 = "KHigh"
    elif k > 0:
        key3 = "Klow"
    else:
        key3 = "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    return render_template('Fertilizer-Result.html', recommendation1=response1,
                           recommendation2=response2, recommendation3=response3,
                           diff_n = abs_n, diff_p = abs_p, diff_k = abs_k)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template("index.html")

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")
cnn_model = load_model("pesticide_cnn.h5")
@app.route("/pesticide")
@app.route("/pesticide_use.html")
def pesticide_upload():
    """
    Render template for pest identification using CNN model.
    """
    return render_template("pesticide_upload.html")

@app.route("/predict_pest", methods=["POST"])
def predict_pest():
    """
    Predict pest using uploaded image and trained CNN model.

    Returns:
        Rendered template with pest identification results.
    """
    if "file" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    
    # Save uploaded file
    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)
    
    # Preprocess image for CNN
    img = image.load_img(filepath, target_size=(128, 128))   # adjust size to your model
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict pest
    prediction = cnn_model.predict(img_array)
    pest_class = np.argmax(prediction, axis=1)[0]
    
    # Map pest_class → pest_name + advice
    pest_labels = {
    0: "aphids",
    1: "armyworm",
    2: "beetle",
    3: "bollworm",
    4: "eathworm",
    5: "grasshoper",
    6: "mites",
    7: "mosquito",
    8: "sawfly",
    9: "stem_borer"}  # example
    advice_dict = {
    "aphids": {
        "description": "Aphids are small sap-sucking insects that weaken plants by feeding on their juices.",
        "steps": [
            "Spray neem oil or insecticidal soap on leaves.",
            "Introduce ladybugs as natural predators.",
            "Avoid excess nitrogen fertilizer, as it attracts aphids."
        ]
    },
    "armyworm": {
        "description": "Armyworms are caterpillars that feed on leaves, stems, and grains, causing rapid damage.",
        "steps": [
            "Apply Bacillus thuringiensis (Bt) sprays on infested crops.",
            "Use pheromone traps to monitor population.",
            "Consult experts before using chemical pesticides."
        ]
    },
    "beetle": {
        "description": "Beetles chew on leaves and stems, reducing crop productivity.",
        "steps": [
            "Handpick beetles early in the morning.",
            "Use predatory nematodes in the soil.",
            "Apply neem-based or organic sprays."
        ]
    },
    "bollworm": {
        "description": "Bollworms bore into cotton and other crops, damaging bolls and flowers.",
        "steps": [
            "Install pheromone traps to detect infestation.",
            "Spray Bt formulations for early control.",
            "Use resistant crop varieties and rotate crops."
        ]
    },
    "eathworm": {
        "description": "Earthworms are beneficial organisms that improve soil fertility.",
        "steps": [
            "No pesticide is needed.",
            "Encourage earthworm presence for soil health.",
            "Avoid unnecessary chemical applications."
        ]
    },
    "grasshoper": {
        "description": "Grasshoppers chew on leaves and stems, reducing foliage.",
        "steps": [
            "Encourage natural predators like birds.",
            "Spray neem-based or garlic extract sprays.",
            "Avoid dense weeds around fields."
        ]
    },
    "mites": {
        "description": "Mites are tiny pests that cause yellow spots and leaf curling.",
        "steps": [
            "Spray miticides or neem oil.",
            "Keep plants well-irrigated to reduce stress.",
            "Avoid excessive pesticide use that kills predators."
        ]
    },
    "mosquito": {
        "description": "Mosquitoes breed in stagnant water near farms, acting as disease carriers.",
        "steps": [
            "Drain stagnant water around fields.",
            "Introduce larvivorous fish in ponds.",
            "Use larvicides if infestation is high."
        ]
    },
    "sawfly": {
        "description": "Sawfly larvae feed on leaves, skeletonizing crops.",
        "steps": [
            "Prune and destroy affected leaves.",
            "Spray neem or pyrethrin-based insecticides.",
            "Encourage parasitic wasps that attack sawfly larvae."
        ]
    },
    "stem_borer": {
        "description": "Stem borers tunnel into stems, weakening the plant structure.",
        "steps": [
            "Use pheromone traps to reduce adult moths.",
            "Remove and burn affected stems.",
            "Apply systemic insecticides at early stages."
        ]
    }
    }

    pest_name = pest_labels.get(pest_class, "Unknown Pest")
    advice_text = advice_dict.get(pest_name, "Consult agricultural expert.")

# Render result page
    return render_template("pesticide_use.html",
                           insect=pest_name,
                           advice=advice_text,
                           img=file.filename)

@app.route("/download_pdf/<pest>/<img>")
def download_pdf(pest, img):
    # File paths
    filename = f"{pest}_report.pdf"
    filepath = os.path.join("static", "pdfs", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Create PDF
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 80, f"Pest Report: {pest.capitalize()}")
    advice_dict = {
    "aphids": {
        "description": "Aphids are small sap-sucking insects that weaken plants by feeding on their juices.",
        "steps": [
            "Spray neem oil or insecticidal soap on leaves.",
            "Introduce ladybugs as natural predators.",
            "Avoid excess nitrogen fertilizer, as it attracts aphids."
        ]
    },
    "armyworm": {
        "description": "Armyworms are caterpillars that feed on leaves, stems, and grains, causing rapid damage.",
        "steps": [
            "Apply Bacillus thuringiensis (Bt) sprays on infested crops.",
            "Use pheromone traps to monitor population.",
            "Consult experts before using chemical pesticides."
        ]
    },
    "beetle": {
        "description": "Beetles chew on leaves and stems, reducing crop productivity.",
        "steps": [
            "Handpick beetles early in the morning.",
            "Use predatory nematodes in the soil.",
            "Apply neem-based or organic sprays."
        ]
    },
    "bollworm": {
        "description": "Bollworms bore into cotton and other crops, damaging bolls and flowers.",
        "steps": [
            "Install pheromone traps to detect infestation.",
            "Spray Bt formulations for early control.",
            "Use resistant crop varieties and rotate crops."
        ]
    },
    "eathworm": {
        "description": "Earthworms are beneficial organisms that improve soil fertility.",
        "steps": [
            "No pesticide is needed.",
            "Encourage earthworm presence for soil health.",
            "Avoid unnecessary chemical applications."
        ]
    },
    "grasshoper": {
        "description": "Grasshoppers chew on leaves and stems, reducing foliage.",
        "steps": [
            "Encourage natural predators like birds.",
            "Spray neem-based or garlic extract sprays.",
            "Avoid dense weeds around fields."
        ]
    },
    "mites": {
        "description": "Mites are tiny pests that cause yellow spots and leaf curling.",
        "steps": [
            "Spray miticides or neem oil.",
            "Keep plants well-irrigated to reduce stress.",
            "Avoid excessive pesticide use that kills predators."
        ]
    },
    "mosquito": {
        "description": "Mosquitoes breed in stagnant water near farms, acting as disease carriers.",
        "steps": [
            "Drain stagnant water around fields.",
            "Introduce larvivorous fish in ponds.",
            "Use larvicides if infestation is high."
        ]
    },
    "sawfly": {
        "description": "Sawfly larvae feed on leaves, skeletonizing crops.",
        "steps": [
            "Prune and destroy affected leaves.",
            "Spray neem or pyrethrin-based insecticides.",
            "Encourage parasitic wasps that attack sawfly larvae."
        ]
    },
    "stem_borer": {
        "description": "Stem borers tunnel into stems, weakening the plant structure.",
        "steps": [
            "Use pheromone traps to reduce adult moths.",
            "Remove and burn affected stems.",
            "Apply systemic insecticides at early stages."
        ]
    }
    }

    # Pest Description
    pest_info = advice_dict.get(pest, {"description": "No data", "steps": []})
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 120, "Description:")
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(120, height - 140, pest_info["description"])

    # Add Uploaded Image
    img_path = os.path.join("static", "uploads", img)
    if os.path.exists(img_path):
        c.drawImage(img_path, 100, height - 380, width=3*inch, height=3*inch, preserveAspectRatio=True)

    # Steps for control
    y = height - 420
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y, "Steps for Control:")
    y -= 20
    c.setFont("Helvetica", 11)
    for step in pest_info["steps"]:
        c.drawString(120, y, f"• {step}")
        y -= 20

    # Save PDF
    c.save()

    return send_file(filepath, as_attachment=True)

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")


@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/' + final_prediction + '.jpg')


if __name__ == '__main__':
    app.run(debug=True)
