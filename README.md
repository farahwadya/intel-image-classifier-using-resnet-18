# Intel Image Classifier --- ResNet18

A containerized computer vision application that classifies natural
scene images into six categories using a fine-tuned **ResNet18** model
built with **PyTorch** and served through **Streamlit**.

The project demonstrates an end-to-end workflow from a trained deep
learning model to a reproducible Dockerized inference application,
including model configuration, preprocessing, prediction, and container
deployment.

## Demo

**Application:** Streamlit web application\
**Model:** ResNet18 · PyTorch\
**Task:** 6-class image classification\
**Deployment:** Docker

> **Note:** The trained model weights (`.pth`) are intentionally
> excluded from this repository because of GitHub file-size
> considerations. The application code expects the trained weights to be
> available locally.

------------------------------------------------------------------------

## What the Model Classifies

The classifier predicts one of six scene categories:

-   Buildings
-   Forest
-   Glacier
-   Mountain
-   Sea
-   Street

The project is based on the Intel Image Classification dataset.

------------------------------------------------------------------------

## Features

-   Upload an image through a simple Streamlit interface
-   Automatic image preprocessing
-   ResNet18 inference with PyTorch
-   CPU/GPU device selection when available
-   Class mapping loaded from JSON configuration
-   Model configuration stored separately from application logic
-   Dockerized application for reproducible deployment
-   Ready for local development and container-based execution

------------------------------------------------------------------------

## Tech Stack

  Category               Technology
  ---------------------- --------------
  Programming Language   Python
  Deep Learning          PyTorch
  Computer Vision        Torchvision
  Image Processing       Pillow
  Web Application        Streamlit
  Containerization       Docker
  Data / Configuration   JSON
  Version Control        Git & GitHub

------------------------------------------------------------------------

## Project Structure

``` text
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
├── buildings.jpg
├── forestpic.jpg
├── sea.jpg
├── street.jpg
└── street.webp
```

### Main Files

**`app.py`**\
Streamlit application responsible for the user interface, image upload,
preprocessing, model loading, and prediction.

**`model.py`**\
Contains the ResNet18 model definition used by the application.

**`classes.json`**\
Stores the mapping between model output indices and the six scene
classes.

**`model_config.json`**\
Stores model-related configuration required to reconstruct the trained
model.

**`transform_config.json`**\
Stores image preprocessing configuration used during inference.

**`Dockerfile`**\
Defines the environment and commands required to build and run the
application as a Docker container.

**`requirements.txt`**\
Contains the Python dependencies required by the application.

------------------------------------------------------------------------

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/farahwadya/intel-image-classifier-using-resnet-18.git
cd intel-image-classifier-using-resnet-18
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it on Windows:

``` powershell
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Add the trained model weights

Place the trained weights file in the project root using the expected
filename:

``` text
resnet18_intel_classifier.pth
```

The weights are not included in this repository because of GitHub
file-size considerations.

### 5. Start the application

``` bash
streamlit run app.py
```

The application will be available at:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

## Run with Docker

### Build the image

``` bash
docker build -t intel-image-classifier:v1 .
```

### Run the container

``` bash
docker run --name intel-test -p 8501:8501 intel-image-classifier:v1
```

Then open:

``` text
http://localhost:8501
```

### Docker Image

The project was also packaged and tested as a Docker image:

``` text
farahwadya/intel-image-classifier:v1
```

If the image is available through Docker Hub, it can be run with:

``` bash
docker run --name intel-test -p 8501:8501 farahwadya/intel-image-classifier:v1
```

------------------------------------------------------------------------

## Docker Workflow

The deployment workflow is:

``` text
Source Code
    ↓
Dockerfile
    ↓
Docker Image
    ↓
Docker Container
    ↓
Streamlit Application
    ↓
Image Prediction
```

The Dockerfile uses a Python 3.11 slim base image, installs the
application dependencies, copies the project files into the container,
exposes the Streamlit port, and starts the application on container
startup.

------------------------------------------------------------------------

## Inference Pipeline

When an image is uploaded, the application follows this workflow:

``` text
Input Image
    ↓
Image Loading
    ↓
Preprocessing / Transformations
    ↓
ResNet18
    ↓
Class Scores
    ↓
Predicted Class
```

The model is loaded for inference and uses the available compute device
when supported.

------------------------------------------------------------------------

## Model

The project uses **ResNet18**, a convolutional neural network
architecture commonly used for image classification.

The trained model is configured for six output classes corresponding to
the scene categories used by the project.

The application separates:

-   Model architecture
-   Model configuration
-   Class labels
-   Image transformation configuration
-   User interface / inference logic

This separation makes the project easier to maintain and adapt.

------------------------------------------------------------------------

## Reproducibility

The project includes the files required to reproduce the application
environment:

-   `requirements.txt` for Python dependencies
-   `Dockerfile` for containerized execution
-   JSON configuration files for model and preprocessing settings
-   Git version control for source-code tracking

The trained model weights are kept outside the Git repository because of
their file size.

------------------------------------------------------------------------

## Limitations

-   The trained `.pth` weights are not stored in this GitHub repository.
-   The application is currently designed as an inference/demo
    application rather than a full training pipeline.
-   The current interface is built with Streamlit and is intended for
    interactive image classification.

------------------------------------------------------------------------

## Future Improvements

Potential extensions include:

-   Add confidence scores and top-k predictions
-   Add model evaluation metrics to the application
-   Add automated tests for preprocessing and inference
-   Expose the model through a FastAPI REST API
-   Add CI/CD with GitHub Actions
-   Publish the container image through a container registry
-   Add model versioning and experiment tracking with MLflow
-   Deploy the application to a cloud platform

------------------------------------------------------------------------

## Author

**Farah Alwadya**\
AI Engineering Student \| AI & Machine Learning

GitHub:\
https://github.com/farahwadya

------------------------------------------------------------------------

## License

This project is intended for educational, portfolio, and demonstration
purposes.
