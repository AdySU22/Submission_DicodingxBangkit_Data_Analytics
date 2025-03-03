# Submission_DicodingxBangkit_Data_Analytics

This project involves analyzing public e-commerce data from Brazil, visualized through a web interface using Streamlit.

## Project Overview

The project provides an analysis of Brazilian e-commerce public data, which is visualized on a website for easy access and interaction. The main goal is to offer valuable insights and a deeper understanding of the dataset.

## Data Source

The dataset is sourced from Kaggle: [Brazilian E-Commerce Dataset]([link to dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data))

## Project Directory Structure

- `/data`: This directory contains the dataset in `.csv` format.
- `/dashboard`: This folder includes the main `Dashboard.py`, `df.csv`, `geolocation_dataset.csv` file, which contains the core code for building the web-based visualization.
- `Proyek_Analisis_Data.ipynb`: This Jupyter Notebook is used for data analysis.
- `requirement.txt`: This is file format for all the libraries used in the project
- `url.txt`: This is file format for the url used for the project

## Installation

To clone the repository to your local machine, run the following command:
git clone https://github.com/AdySU22/Submission_DicodingxBangkit_Data_Analytics.git

## Setup Environment

## Setup Environment - Shell/Terminal
```
mkdir proyek_analisis_data
cd Submission_DicodingxBangkit_Data_Analytics\dashboard
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Run steamlit app
```
streamlit run dashboard.py
```