# Naive Bayes Classification Model Validation

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-✓-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-✓-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-✓-green.svg)

This repository contains Python code for validating the performance of a binary classification model (naive Bayes) through different evaluation metrics and visualizations.

## Features

- 📊 **Calibration Plot**
  - Compares predicted probabilities against observed event rates
  - Calculates R² value between predicted and observed rates
  - Includes 95% confidence intervals for each probability bin
  - Computes Brier Score and Expected Calibration Error (ECE)

- 📈 **ROC Curve Analysis**
  - Evaluates model discrimination ability
  - Calculates Area Under the Curve (AUC)
  - Determines optimal threshold using Youden's Index
  - Includes standard error calculations for AUC

- 🔢 **Performance Metrics**
  - Sample size (N)
  - Event count and proportion
  - R² value with standard error
  - AUC with standard error

## Usage

Input Data Requirements

1) Prepare your input dataframe with the following columns (an example is provided):
   - status_name: outcome/class labels
   - probability_percent: the model's predicted probabilities (0-100 scale)
2) Call the model_performance_plots() function with your dataframe
3) The function will:
   - Generate and display calibration and ROC plots
   - Save plots as both SVG and PNG files
   - Return a dataframe with performance metrics

## Outputs

The function generates:

- calibrationplot.svg and calibrationplot.png - Calibration plot visualization
- roccurve.svg and roccurve.png - ROC curve visualization
- Console output of frequency distribution of statuses, R² value, Brier Score, Expected Calibration Error (ECE), and encounter counts per probability bin
- A pandas DataFrame with performance metrics

## Notes
- The code assumes binary classification (event vs. non-event)
- Probability bins are hardcoded as 0-10%, 11-20%, etc.
- The example uses a simple threshold (50%) for predicted class assignment

## Potential Additions and Customizations

- Split up dataframe by different demographics to performs a fairness analysis (if applicable)
- Add a date or time column to the dataframe and track AUC or ECE over time

## License

MIT License - Free for academic and commercial use

(More to be added soon)
