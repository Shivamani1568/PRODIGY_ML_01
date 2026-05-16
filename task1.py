import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def generate_dataset(n=500, seed=42):
    rng = np.random.default_rng(seed)
    sqft = rng.integers(600, 4000, n)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)

    base = 50_000
    price = (
        base
        + sqft * 150
        + bedrooms * 12_000
        + bathrooms * 18_000
        + rng.normal(0, 25_000, n)
    )
    price = np.maximum(price, 40_000).round(-3)

    return pd.DataFrame(
        {
            "sqft": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "price": price.astype(int),
        }
    )


def main():
    df = generate_dataset()
    print("Sample of the dataset:")
    print(df.head(), "\n")

    X = df[["sqft", "bedrooms", "bathrooms"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Learned coefficients:")
    for name, coef in zip(X.columns, model.coef_):
        print(f"  {name:>10}: {coef:,.2f}")
    print(f"  {'intercept':>10}: {model.intercept_:,.2f}\n")

    print("Model performance on the test set:")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  R^2  : {r2:.4f}\n")

    new_houses = pd.DataFrame(
        {
            "sqft": [1200, 2500, 3500],
            "bedrooms": [2, 4, 5],
            "bathrooms": [1, 3, 3],
        }
    )
    predictions = model.predict(new_houses)

    print("Predictions for new houses:")
    for (_, row), pred in zip(new_houses.iterrows(), predictions):
        print(
            f"  {row.sqft} sqft, {row.bedrooms} bed, "
            f"{row.bathrooms} bath -> ${pred:,.0f}"
        )

    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.5, edgecolor="k", linewidth=0.3)
    lims = [y_test.min(), y_test.max()]
    axes[0].plot(lims, lims, "r--", linewidth=2)
    axes[0].set_xlabel("Actual price ($)")
    axes[0].set_ylabel("Predicted price ($)")
    axes[0].set_title("Predicted vs Actual")

    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolor="k", linewidth=0.3)
    axes[1].axhline(0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted price ($)")
    axes[1].set_ylabel("Residual ($)")
    axes[1].set_title("Residuals")

    plt.tight_layout()
    plt.savefig("regression_plots.png", dpi=120)
    plt.show()
    print("\nSaved plots to regression_plots.png")


if __name__ == "__main__":
    main()