import dotenv
import os
import keras
import matplotlib
from polygon import RESTClient

dotenv.load_dotenv()
POLYGON_API_KEY = os.getenv("API_KEY")
ticker = input("What Ticker Would You Like Predicted? ")
client = RESTClient(POLYGON_API_KEY)

aggs = []
for a in client.list_aggs(
    ticker,
    1,
    "minute",
    "2022-01-01",
    "2023-02-03",
    limit=50000,
):
    aggs.append(a)

print(aggs)

model = keras.Sequential([
    keras.layers.Input(shape=(1,1)),            # Input layer
    keras.layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation
    keras.layers.Dense(1),                      # Output layer with a single neuron (for regression)
    keras.layers.Dropout(0.20)
])

model.compile(optimizer='adam', loss='mean_squared_error')