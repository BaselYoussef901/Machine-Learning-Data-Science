from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np

classes = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}
model = models.load_model("cifar10.keras")

def predict_image(model, image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize((32,32))
    img_data = np.asarray(img)
    img_data = img_data/255

    probs = model.predict(np.array([img_data])[:1])
    top_probs = probs.max()
    top_pred = classes[np.argmax(probs)]
    return top_probs, top_pred

def on_change(state, var_name, var_val):
    if var_name == "content":
        top_prob, top_pred = predict_image(model, var_val)
        state.prob = round(top_prob * 100)
        state.pred = "this is a " + top_pred
        state.image_path = var_val

content = ""
image_path = "placeholder_image.png"
prob = 0
pred = ""
index = """
<|text-center|
<|{"logo.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.png|>
select an image from your file system

<|{pred}|>

<|{image_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""
app = Gui(page=index)

if __name__=="__main__":
    app.run(use_reloader=True)