from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model, custom_object_scope
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import pickle

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the custom BahdanauAttention layer
class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.Wa = tf.keras.layers.Dense(units)
        self.Ua = tf.keras.layers.Dense(units)
        self.Va = tf.keras.layers.Dense(1)

    def call(self, query, values):
        score = self.Va(tf.nn.tanh(self.Wa(query)[:, tf.newaxis, :] + self.Ua(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Load the VGG16 model for feature extraction
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load the trained image captioning model and tokenizer
with custom_object_scope({'BahdanauAttention': BahdanauAttention}):
    model = load_model('models/model/best_model.h5')
with open('models/transformers/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 35

def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature

def generate_caption(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()[1:-1]
    return ' '.join(final_caption)

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    file_path = os.path.join('static/uploads', file.filename)
    with open(file_path, 'wb') as f:
        f.write(await file.read())
    feature = extract_features(file_path)
    caption = generate_caption(feature)
    return {"caption": caption}

@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Automated Image Captioning</title>
        <link rel="stylesheet" type="text/css" href="/static/css/style.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" integrity="sha512-ki+8CGFf4C2w5zGuD72yMEqJ48Kr0uH/qNz8+wOtbO/hRJFw5hzJg+vfuWroS57wiFTWxPtj0KPscvB0iuu1zg==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    </head>
    <body class="light-theme">
        <header>
            <h1>Automated Image Captioning</h1>
            <div class="theme-switch">
                <input type="checkbox" id="theme-toggle">
                <label for="theme-toggle">Dark Mode</label>
            </div>
            <a href="https://github.com/gauravkumarchaurasiya" target="_blank" class="github-icon">
                <i class="fab fa-github"></i> Go to Repo
            </a>
        </header>
        <div class="container">
            <h2>Upload an Image to Generate a Caption by Model</h2>
            <p>Allowed file types: JPEG, PNG. Maximum file size: 5MB.</p>
            <form id="upload-form" action="/upload/" enctype="multipart/form-data" method="post">
                <div class="file-drop-area">
                    <span class="choose-file-button" onclick="document.querySelector('.file-input').click()">Choose Image</span>
                    <span class="file-message">or drag and drop here</span>
                    <input class="file-input" type="file" name="file" accept=".jpg, .jpeg, .png" required>
                </div>
                <div id="file-info"></div>
                <input class="submit-btn" type="submit" value="Generate">
            </form>
            <div id="loading" class="loading">Generating caption...</div>
            <div id="result"></div>
        </div>
        <script>
            const form = document.getElementById('upload-form');
            const fileInput = document.querySelector('.file-input');
            const fileDropArea = document.querySelector('.file-drop-area');
            const fileInfo = document.getElementById('file-info');
            const loadingIndicator = document.getElementById('loading');
            const resultDiv = document.getElementById('result');
            const maxFileSize = 5 * 1024 * 1024; // 5MB

            fileInput.addEventListener('change', function() {
                if (fileInput.files.length > 0) {
                    const file = fileInput.files[0];
                    const fileName = file.name;
                    const fileSize = (file.size / 1024 / 1024).toFixed(2); // Size in MB

                    if (file.size > maxFileSize) {
                        alert('File size exceeds the maximum limit of 5MB.');
                        fileInput.value = ''; // Clear the file input
                        fileInfo.textContent = '';
                        return;
                    }

                    fileDropArea.querySelector('.file-message').textContent = fileName;
                    fileInfo.textContent = `File size: ${fileSize} MB`;
                }
            });

            fileDropArea.addEventListener('dragover', (event) => {
                event.preventDefault();
                fileDropArea.classList.add('dragover');
            });

            fileDropArea.addEventListener('dragleave', () => {
                fileDropArea.classList.remove('dragover');
            });

            fileDropArea.addEventListener('drop', (event) => {
                event.preventDefault();
                fileDropArea.classList.remove('dragover');
                const files = event.dataTransfer.files;
                fileInput.files = files;
                if (files.length > 0) {
                    const file = files[0];
                    const fileName = file.name;
                    const fileSize = (file.size / 1024 / 1024).toFixed(2); // Size in MB

                    if (file.size > maxFileSize) {
                        alert('File size exceeds the maximum limit of 5MB.');
                        fileInput.value = ''; // Clear the file input
                        fileInfo.textContent = '';
                        return;
                    }

                    fileDropArea.querySelector('.file-message').textContent = fileName;
                    fileInfo.textContent = `File size: ${fileSize} MB`;
                }
            });

            form.addEventListener('submit', async (event) => {
                event.preventDefault();
                if (fileInput.files.length === 0) {
                    alert("Please select an image file before submitting.");
                    return;
                }
                loadingIndicator.style.display = 'block';
                resultDiv.innerHTML = '';
                const formData = new FormData(form);
                const response = await fetch('/upload/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                loadingIndicator.style.display = 'none';
                resultDiv.innerHTML = `
                    <div class="caption">Generated Caption: ${result.caption}</div>
                    <img src="${URL.createObjectURL(formData.get('file'))}" alt="Uploaded Image">
                `;
            });

            const themeToggle = document.getElementById('theme-toggle');
            themeToggle.addEventListener('change', () => {
                if (themeToggle.checked) {
                    document.body.classList.add('dark-theme');
                    document.body.classList.remove('light-theme');
                } else {
                    document.body.classList.add('light-theme');
                    document.body.classList.remove('dark-theme');
                }
            });
        </script>
    </body>
    </html>
    """
    return content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
