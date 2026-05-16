# House Price Prediction - Linear Regression

This is my implementation for the first machine learning task. The goal was to
build a linear regression model that predicts house prices based on square
footage, number of bedrooms, and number of bathrooms.

## What it does

The script creates a dataset of 500 houses where the price depends on the three
features plus some random noise, then trains a linear regression model on it and
checks how well it learned the relationship.

I went with a generated dataset on purpose - since I know the real formula used
to build the prices, I can actually verify that the model figures out the right
coefficients on its own instead of just trusting a score.

## How to run

You'll need Python with a few libraries. I used a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn pandas numpy matplotlib
python3 task1.py
```

Running it prints a sample of the data, the coefficients the model learned, the
performance metrics, and predictions for a few example houses. It also opens and
saves the graphs.

## Results

The model lands at around 97% R2 on the test set. The coefficients it learns
come out very close to the real values used to build the data (about $150 per
square foot, $12k per bedroom, $18k per bathroom), which is the main thing I
wanted to confirm.

## Graphs

`regression_plots.png` has two plots:

- Predicted vs Actual - points sit close to the diagonal line, which means the
  predictions are accurate.
- Residuals - the errors scatter randomly around zero with no real pattern,
  which is a good sign that linear regression fits this data well.

## Files

- `task1.py` - the full code
- `regression_plots.png` - the output graphs
