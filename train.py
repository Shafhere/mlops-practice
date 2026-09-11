# import numpy as np
# import pandas as pd

# np.random.seed(42)

# n = 1000

# df = pd.DataFrame({
#     "area": np.random.randint(500, 5001, n),
#     "bedrooms": np.random.randint(1, 6, n),
#     "bathrooms": np.random.randint(1, 4, n),
#     "age": np.random.randint(0, 31, n),
#     "location_score": np.random.randint(1, 11, n)
# })

# df["price"] = (
#     df["area"] * 0.05
#     + df["bedrooms"] * 5
#     + df["bathrooms"] * 4
#     - df["age"] * 0.5
#     + df["location_score"] * 10
#     + np.random.normal(0, 10, n)
# )

# df["price"] = df["price"].round(2)

# df.to_csv("data/house_price.csv", index=False)

# print("Dataset created!")
# print(df.head())

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv("data/house_price.csv")

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R2:", r2)

joblib.dump(model, "models/model_v1.pkl")

print("Model v1 saved!")