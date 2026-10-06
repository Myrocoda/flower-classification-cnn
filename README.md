# Flower Classification with a CNN

Classifies flower photos into five classes (daisy, dandelion, roses, sunflowers, tulips) with a convolutional neural network, and serves predictions through a Streamlit app.

## Dataset
TensorFlow flower_photos dataset (~3,670 images, 5 classes):
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

Extract it and set `DATA_DIR` in the notebook to its location. The dataset is not included in this repo.

## Pipeline
- Images resized to 128x128; 70/30 train/validation split (`seed=123`)
- Data augmentation: random horizontal flip, rotation (0.2), zoom (0.2)
- CNN: 5 x (Conv2D + MaxPooling2D) blocks (32 to 512 filters), then Flatten, Dense(128), Dense(5, softmax)
- ~1.83M parameters; Adam optimizer, sparse categorical cross-entropy, 60 epochs
- The model is trained on raw 0-255 pixels (no Rescaling layer), so inference must use raw pixels too

## Results
~80% training accuracy and ~69% validation accuracy at the final epoch. The gap points to overfitting. Per-class ROC curves and overall accuracy are in the evaluation section of the notebook.

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload a JPG or PNG of a flower to get the predicted class and per-class probabilities.

## Files
- `flower_CNN.ipynb`: data exploration, training, evaluation
- `app.py`: Streamlit app
- `flower_model1.h5`: trained model

## Next steps
Add a Rescaling layer inside the model, try transfer learning (e.g. MobileNetV2), add early stopping, and hold out a separate test set.
