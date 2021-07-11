from flask import Flask, render_template, request
import os
import pickle
import PIL
from PIL import Image
import fastbook
from fastbook import *
from fastai.vision.widgets import *
from fastai.vision import *
import torchvision.transforms as T




app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def hello_world():
	if request.method=="POST":
		file = request.files["file"]
		file.save(os.path.join("uploads", file.filename))
		learn_inf = load_learner('export.pkl')
		print(os.path.join("uploads", file.filename))
		img = PILImage.create(os.path.join("uploads", file.filename))
		#print(type(img))
		pred,pred_idx,probs = learn_inf.predict(img)
		os.remove(os.path.join("uploads", file.filename))
		return render_template("index.html", message=f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
	return render_template("index.html")

if __name__ == '__main__':
    # app.run()
    app.run(debug=True)
