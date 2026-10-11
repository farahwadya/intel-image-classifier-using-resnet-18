# Intel Image Classifier | ResNet18

A computer vision application that classifies natural scene images into six categories using a fine-tuned **ResNet18** model built with **PyTorch**.

The project demonstrates an end-to-end workflow, from a trained deep learning model to a deployed, interactive application, covering image preprocessing, inference, model configuration, and Docker containerization.

## Live Demo

**[Try the Live Application](https://intel-image-classifier-using-resnet-18-a7jj9s7uegzwmbzkqysin9.streamlit.app/)**

Upload an image and see the model's predicted scene category.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://intel-image-classifier-using-resnet-18-a7jj9s7uegzwmbzkqysin9.streamlit.app/)

## Project Overview

The application uses a fine-tuned ResNet18 model to classify natural scene images into six categories. It provides an interactive interface for image upload and model inference.

### Potential Applications

* **Image organization:** Grouping images by scene category.
* **Urban and geospatial workflows:** Supporting initial categorization of street and building imagery.
* **Environmental image collections:** Organizing forest, glacier, mountain, and sea images for further analysis.

These are potential use cases; the project itself focuses on image classification rather than full-scale urban or environmental analysis.

## What the Model Classifies

The model predicts one of six scene categories:

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

The project is based on the Intel Image Classification dataset.

## Features

* Upload images through a Streamlit interface.
* Preprocess images before inference.
* Run predictions using a fine-tuned ResNet18 model.
* Select an available CPU or GPU device where supported.
* Load class labels and model settings from JSON configuration files.
* Run the application locally or inside a Docker container.

## Tech Stack

| Category             | Technology     |
| -------------------- | -------------- |
| Programming Language | Python         |
| Deep Learning        | PyTorch        |
| Computer Vision      | Torchvision    |
| Image Processing     | Pillow         |
| Web Application      | Streamlit      |
| Containerization     | Docker         |
| Configuration        | JSON           |
| Version Control      | Git and GitHub |

## Model Performance

* **Architecture:** ResNet18
* **Framework:** PyTorch
* **Task:** Six-class image classification
* **Reported accuracy:** 90%

*Accuracy should be interpreted in the context of the evaluation dataset and test methodology.*

## Project Structure

```text
intel-image-classifier-using-resnet-18/
│
├── app.py
├── model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
│
├── classes.json
├── model_config.json
├── transform_config.json
│
├── resnet18_intel_classifier.pth
│
├── buildings.jpg
├── forestpic.jpg
├── sea.jpg
├── street.jpg
└── street.webp
```

### Main Files

* **`app.py`** — Streamlit interface, image upload, preprocessing, model loading, and prediction workflow.
* **`model.py`** — ResNet18 model definition.
* **`classes.json`** — Mapping between output indices and scene categories.
* **`model_config.json`** — Configuration used to reconstruct the model.
* **`transform_config.json`** — Image preprocessing settings.
* **`resnet18_intel_classifier.pth`** — Trained model weights.
* **`Dockerfile`** — Instructions for building the Docker image.
* **`requirements.txt`** — Python dependencies.

## Deployment

### Streamlit Community Cloud

The application is deployed on Streamlit Community Cloud and can be accessed through the live demo link above.

### Docker

The application was also containerized and tested separately to verify that the model and Streamlit interface run inside a Docker container.

**Docker Hub Image:** `farahwadya/intel-image-classifier:v1`

[View the Docker image on Docker Hub](https://hub.docker.com/r/farahwadya/intel-image-classifier)

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/farahwadya/intel-image-classifier-using-resnet-18.git
cd intel-image-classifier-using-resnet-18
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

Ensure the trained weights file, `resnet18_intel_classifier.pth`, is available in the project root, then run:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Run with Docker

### Pull the published image

```bash
docker pull farahwadya/intel-image-classifier:v1
```

### Run the container

```bash
docker run --name intel-test -p 8501:8501 farahwadya/intel-image-classifier:v1
```

Open `http://localhost:8501` in your browser.

### Build the image from source

```bash
docker build -t intel-image-classifier:v1 .
```

## Inference Pipeline

```text
Input Image
    ↓
Image Loading
    ↓
Preprocessing
    ↓
ResNet18 Inference
    ↓
Class Scores
    ↓
Predicted Scene Category
```

The application loads the trained model for inference and uses the configured image transformations and class mapping.

## Engineering Highlights

* Integrated a trained deep learning model into an interactive application.
* Separated model architecture, configuration, preprocessing, and interface logic.
* Containerized the application with Docker.
* Worked through deployment issues involving Python dependencies and model file availability.
* Published the source code and Docker image for reproducibility and review.

## Limitations

* The application performs scene classification; it does not provide object detection or detailed scene understanding.
* Predictions depend on the training data and the quality of the input image.
* The model is intended for inference and demonstration, not for model training within the deployed interface.

## Future Improvements

* Display confidence scores and top-k predictions.
* Add a model evaluation report with per-class precision, recall, and F1-score.
* Add automated tests for preprocessing and inference.
* Expose predictions through a FastAPI REST API.
* Add CI/CD with GitHub Actions.
* Track experiments and model versions with MLflow.

## Author

**Farah Alwadya**
AI Engineering | Computer Vision | Deep Learning

[GitHub Profile](https://github.com/farahwadya)

## License

This project is intended for educational, portfolio, and demonstration purposes.
