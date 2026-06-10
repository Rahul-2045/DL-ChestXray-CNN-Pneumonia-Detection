# PneumoScan CNN: Chest X-Ray Pneumonia Detection

This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images into two classes:

| Class | Meaning |
|---|---|
| NORMAL | X-ray does not show pneumonia signs |
| PNEUMONIA | X-ray shows pneumonia signs |

> **Important:** This project is for learning and portfolio demonstration only. It should not be used for real medical diagnosis.

---

## Project Folder Structure

Keep your project folder like this:

```text
PneumoScan-CNN-DeepLearning/
│
├── app.py
├── README.md
├── pneumonia_cnn_model.h5
├── Pneumonia_Detection_CNN_Deep_Learning_Project.ipynb
│
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## Dataset

The dataset should contain chest X-ray images in three folders:

```text
train
val
test
```

Each folder should have two class folders:

```text
NORMAL
PNEUMONIA
```

Example path:

```python
DATASET_PATH = r"E:\Rahul Verma\document\D drive\PROJECTS\Vscode\DeepLearning\chest_xray"
```

---

## Model Details

The CNN model contains:

- Conv2D layers
- BatchNormalization
- MaxPooling2D
- Flatten layer
- Dense layer
- Dropout
- Sigmoid output layer

Since this is a binary classification project, the model uses:

```text
Loss Function: binary_crossentropy
Optimizer: Adam
Output Activation: sigmoid
```

---

## How to Train the Model

Open and run the notebook:

```text
Pneumonia_Detection_CNN_Deep_Learning_Project.ipynb
```

After training, save the model as:

```text
pneumonia_cnn_model.h5
```

The app expects the model file in the same folder as `app.py`.

---

## Install Required Libraries

Run this command in your terminal:

```bash
pip install tensorflow streamlit pillow numpy
```

Optional:

```bash
pip install matplotlib scikit-learn
```

---

## Run the Interactive App

From your project folder, run:

```bash
streamlit run app.py
```

The app will open in your browser.

You can upload any chest X-ray image and the model will predict:

```text
NORMAL
```

or

```text
PNEUMONIA
```

---

## App Features

The Streamlit app provides:

- Image upload option
- X-ray image preview
- Pneumonia / Normal prediction
- Confidence score
- Probability bar
- Model path configuration
- Simple explanation for users

---

## Sample Prediction Logic

```python
if prediction > 0.5:
    result = "PNEUMONIA"
else:
    result = "NORMAL"
```

---

## How to Test Images

Use images from the `test` folder:

```text
chest_xray/test/NORMAL
chest_xray/test/PNEUMONIA
```

Example:

```text
chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
```

---

## GitHub Repository Name Suggestion

```text
PneumoScan-CNN-DeepLearning
```

Professional project title:

```text
PneumoScan: Chest X-Ray Pneumonia Detection using CNN
```

---

## Interview Explanation

You can explain this project like this:

> I built a deep learning project for pneumonia detection from chest X-ray images.  
> I used a CNN model to classify X-ray images into NORMAL and PNEUMONIA.  
> I performed image preprocessing, normalization, and data augmentation.  
> I evaluated the model using accuracy, confusion matrix, precision, recall, and F1-score.  
> Finally, I saved the trained model and created an interactive Streamlit app where users can upload an X-ray image and get the prediction.

---

## Disclaimer

This project is only for education and demonstration. It is not approved for clinical or medical diagnosis.
