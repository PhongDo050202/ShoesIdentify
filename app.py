from flask import Flask, jsonify, request, render_template, session, url_for
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
app.secret_key = 'anything' 

# link to model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))
model_path = os.path.abspath(os.path.join(model_dir, "Mobilenet.h5"))

# link to input/output
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "output"))
output_image_path = os.path.abspath(os.path.join(output_dir, "bar.png"))
input_image_path = os.path.abspath(os.path.join(output_dir, "origin_image.png"))

image_uploaded = False

def plot_decision_percentage(score):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots() 
    y = ['Adidas', 'Balenciaga', 'Nike', 'Puma']
    y_axis = [0, 0.5, 1, 1.5]
    score = np.squeeze(score)
    # getting values against each value of y
    plt.barh(y_axis, score, 0.5)
    plt.yticks(y_axis, y)
    # setting label of y-axis
    plt.ylabel("brand")
    # setting label of x-axis
    plt.xlabel("percentage") 
    for bar, percen in zip(ax.patches, score.astype(np.float32) * 100):
        if percen / 200 > 0.2:
            ax.text(percen / 200, bar.get_y() + bar.get_height() / 2, str(percen) + "%", color='white', ha='center', va='center') 
    plt.savefig(output_image_path)

@app.route("/")
def index():
    session.pop("image_uploaded", None)
    return render_template("index.html", image_uploaded=image_uploaded)


@app.route("/check_image_uploaded", methods=["GET"])
def check_image_uploaded():
    return jsonify({"image_uploaded": session.get("image_uploaded", False)})



@app.route("/predict", methods=["POST"])
def predict():
    global image_uploaded
    class_name = ["Adidas", "Balenciaga", "Nike", "Puma"]
    model = keras.models.load_model(model_path)
    img = request.files["data"]
    img = Image.open(img)
    img.save(input_image_path)
    img = img.resize((224, 224))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predict = model.predict(img_array)
    score = predict
    print("score: ", score)
    plot_decision_percentage(score=score)
    brand = class_name[np.argmax(score)]
    
    image_uploaded = True

    session["image_uploaded"] = True

    return render_template("predict.html", brand=brand, image_uploaded=image_uploaded)

if __name__ == "__main__":
    app.run(debug=True)
