# Energy Consumption Prediction Using Machine Learning

## Project Overview

This project predicts hourly energy consumption using historical energy measurements, outdoor temperature, and time-based features.

Three machine learning regression algorithms are compared to evaluate their prediction performance.

## Objectives

* Predict hourly energy consumption in kWh.
* Compare multiple machine learning models.
* Evaluate prediction accuracy.
* Visualize actual versus predicted consumption.
* Identify important input features.
* Generate a next-hour energy forecast.
* Flag unusually high predicted consumption.

## Machine Learning Algorithms

1. Linear Regression
2. Random Forest Regressor
3. HistGradientBoosting Regressor

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Dataset

UCI Appliances Energy Prediction dataset:

https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

The dataset contains household appliance energy measurements and environmental information.

## Features

* Hourly data aggregation
* Time-based feature engineering
* Historical energy consumption features
* Chronological training and testing
* Model comparison using MAE, RMSE, and R²
* Energy consumption visualizations
* Next-hour prediction
* High-consumption alert

## Project Structure

```text
Energy_Consumption_Prediction/
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
└── outputs/
```

## Installation

Install Python and the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the dataset from the UCI link above and place `energydata_complete.csv` in the same folder as `main.py`.

## Run the Project

```bash
python main.py
```

The program trains and compares three models, generates graphs, and saves the results in the `outputs/` folder.

## Evaluation Metrics

* **MAE:** Mean Absolute Error
* **RMSE:** Root Mean Squared Error
* **R²:** Coefficient of Determination

The model with the lowest test MAE is selected for the demonstration forecast.

## Outputs

The program generates model comparison results, prediction graphs, feature importance results, and a next-hour forecast.

## Limitations

The next-hour forecast assumes the latest observed outdoor temperature remains unchanged. The high-consumption alert identifies unusual predicted usage but does not prove that energy is being wasted.

The project demonstrates prediction and decision support; actual energy savings must be verified using measured consumption.

## Author

RITUPARNA SAHU
