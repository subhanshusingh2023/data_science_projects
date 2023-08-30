# Sentiment Analysis Project

![Sentiment Analysis](sentiment.png)

This repository contains the code and resources for a sentiment analysis project. The project involves building and deploying a machine learning model to predict the sentiment (positive or negative) of movie reviews.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Gradio Interface](#Gradio-Interface)
- [Deployment](#deployment)
- [Acknowledgments](#acknowledgments)

## Introduction

Sentiment analysis, also known as opinion mining, is the process of determining the sentiment expressed in a piece of text. In this project, we focus on sentiment analysis of movie reviews using machine learning techniques. We explore various models, including Naive Bayes, SVM, Word2Vec,and [Huggingface](https://huggingface.co/SamLowe/roberta-base-go_emotions), to predict whether a given review has a positive or negative sentiment.

## Project Overview

The project involves the following steps:

1. **Data Preparation**: Loading and preprocessing the movie review dataset.
2. **Text Preprocessing**: Cleaning and transforming the text data for analysis.
3. **Model Training**: Building and training machine learning models (Naive Bayes, SVM) on the preprocessed text data.
4. **Word2Vec Embeddings**: Using Word2Vec to create vector representations of words.
5. **Model Evaluation**: Evaluating the trained models' performance on test data.
6. **Deployment**: Creating a web interface using Gradio to allow users to input text and get sentiment predictions.

## Getting Started

To run and explore the project, follow these steps:

1. Clone the repository:
git clone https://github.com/subhanshusingh2023/data_science_projects/movie_sentiment_analysis.git
cd sentiment-analysis-project


2. Install the required dependencies:
pip install -r requirements.txt


3. Run the Jupyter notebooks or Python scripts to explore the project.

## Project Structure

The project is organized as follows:

- `IMDB_Dataset.csv`: is the dataset.
- `best_model.joblib`: is the model with best prediction.
- `Sentiment Analysis of Movie Reviews`: Jupyter notebook for data  preprocessing, model training, and analysis.
- `app.py`: Gradio application script for sentiment prediction.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project overview and instructions.
- `word2vec.model` is the word embeddings.

## Usage

- Use the Jupyter notebook to explore data preprocessing, model training, and evaluation.
- Run the Gradio application using the following command:
gradio run app.py

This will start the web interface for sentiment prediction.

# Gradio Interface
The Gradio interface provides an interactive way to predict sentiment using different models:

Best Model: Predicts sentiment using the best-performing model trained in this project.
Hugging Face Model: Predicts sentiment using the Hugging Face transformer model.

## Deployment

The Gradio application can be deployed using platforms like [Heroku](https://www.heroku.com/), [Vercel](https://vercel.com/), or [Google Cloud Platform](https://cloud.google.com/). Refer to the deployment documentation of the chosen platform for detailed instructions.

## Notes
Make sure you have the necessary dataset, models, and libraries installed before running the code.
For more information about the Hugging Face sentiment analysis model, visit Hugging Face Transformers.
Feel free to modify the code and interface to suit your preferences and needs.
Happy coding!

## Acknowledgments

This project is inspired by various sentiment analysis tutorials and resources. Special thanks to the authors of the tutorials and the open-source libraries used in this project.

Feel free to contribute, raise issues, or suggest improvements to this repository.

---
