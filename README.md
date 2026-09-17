# Forest Cover Classification — Streamlit Application

## Overview

This project is a Streamlit application for exploring forest cover data and
performing forest cover type classification using machine learning.

The application provides interactive data exploration and allows users to
compare several classification models, including Random Forest, K-Nearest
Neighbors (KNN), and Logistic Regression.

The project was developed as a reproducible data science application with
automated tests, Docker containerization, and continuous integration.

## Project Structure

```text
forest-cover-project/
├── forest_cover_app.py          # Main Streamlit application
├── forest_cover_models.ipynb    # Data science / modelling notebook
├── data_utils.py                # Data loading and filtering utilities
├── tests/
│   └── test_data_utils.py       # Unit tests
├── train (1).csv                # Training dataset
├── test-full (1).csv            # Test dataset
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── .dockerignore
├── .gitignore
└── README.md
```

## Installation

### Requirements

- Python 3.12
- pip

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Streamlit Application

Run:

```bash
streamlit run forest_cover_app.py
```

The application is then available locally at:

```text
http://localhost:8501
```

## Tests

Unit tests are implemented with `pytest`.

They cover the data loading and filtering utilities.

Run the tests with:

```bash
python -m pytest -v
```

## Docker

The application can also be executed inside a Docker container, providing an
isolated and reproducible environment.

Build the Docker image:

```bash
docker build -t forest-cover-app .
```

Run the container:

```bash
docker run --rm -p 8501:8501 forest-cover-app
```

If port 8501 is already being used locally:

```bash
docker run --rm -p 8502:8501 forest-cover-app
```

Then open:

```text
http://localhost:8502
```

## Continuous Integration

Continuous Integration is implemented using GitHub Actions.

On every push or pull request, the CI pipeline automatically:

1. checks out the repository;
2. installs Python and the project dependencies;
3. runs the automated tests with pytest.

This ensures that changes to the repository can be automatically validated.

## Reproducibility

Several practices are used to make the project reproducible:

- dependencies are declared in `requirements.txt`;
- data required by the application is included with the project;
- data loading and filtering functions are covered by automated tests;
- the application can run in an isolated Docker container;
- the source code and project configuration are version-controlled with Git;
- Continuous Integration automatically executes the test suite.

## Machine Learning Models

The application includes three classification approaches:

- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression

## Author

Lancelot Brisset