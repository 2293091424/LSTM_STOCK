import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import argparse
import sys
import os

# Suppress warnings
warnings.filterwarnings("ignore")


class LSTMStockPredictor:
    def __init__(self, company, start_date, end_date, future_days, test_size=0.33, n_features=1, look_back=60):
        self._init_logger()
        self.company = company
        self.start_date = start_date
        self.end_date = end_date
        self.n_features = n_features
        self.look_back = look_back
        self.future_days = future_days
        self._load_and_preprocess_data()
        self._split_train_test_data(test_size)
        self._build_model()

    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def _load_and_preprocess_data(self):
        """Load stock data and normalize it."""
        try:
            data = yf.download(self.company, start=self.start_date, end=self.end_date)
        except IOError:
            print("Invalid stock selection. Please try again with a stock that is available on Yahoo finance.")
            sys.exit()

        self.dates = data.index  # Save the date index
        self.data = data[['Close']].values  # Only use 'Close' prices
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(self.data)

    def _split_train_test_data(self, test_size):
        """Split the data into train and test datasets."""
        train_size = int(len(self.data) * (1 - test_size))
        self.train_data = self.scaled_data[:train_size]
        self.test_data = self.scaled_data[train_size:]
        self.test_dates = self.dates[train_size:]  # Save test dates

    def _build_model(self):
        """Build LSTM model."""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.look_back, self.n_features)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def create_dataset(self, data):
        """Prepare dataset in the form of X (input) and y (output)."""
        X, y = [], []
        for i in range(self.look_back, len(data)):
            X.append(data[i - self.look_back:i, 0])  # Take previous look_back days as input
            y.append(data[i, 0])  # Predict the next day's price
        return np.array(X), np.array(y)

    def train(self):
        """Train the LSTM model."""
        X_train, y_train = self.create_dataset(self.train_data)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], self.n_features)
        self.model.fit(X_train, y_train, epochs=100, batch_size=32)

    def predict(self, data):
        """Predict stock prices."""
        X_test, y_test = self.create_dataset(data)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], self.n_features)
        predictions = self.model.predict(X_test)
        return predictions, y_test

    def predict_future_prices(self):
        """Predict future stock prices."""
        future_predictions = []
        last_data_point = self.test_data[-self.look_back:].reshape(1, self.look_back, self.n_features)
        for _ in range(self.future_days):
            prediction = self.model.predict(last_data_point)
            future_predictions.append(prediction[0, 0])
            last_data_point = np.append(last_data_point[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
        return future_predictions

    def plot_results(self, real_prices, predicted_prices, out_dir, stock_name):
        """Plot the results of actual vs predicted prices."""
        plt.figure(figsize=(10, 6))
        plt.plot(real_prices, label="Actual Prices", color='blue')
        plt.plot(predicted_prices, label="Predicted Prices", color='red')
        plt.title(f"{stock_name} Stock Price Prediction")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        save_path = f"{out_dir}/{stock_name}_LSTM_Predictions.png"
        plt.savefig(save_path)
        plt.show()

    def calculate_mape(self, real_prices, predicted_prices):
        """Calculate the Mean Absolute Percentage Error (MAPE)."""
        mape = np.mean(np.abs((real_prices - predicted_prices) / real_prices)) * 100
        return mape


def check_bool(boolean):
    """Helper function to convert boolean inputs."""
    if isinstance(boolean, bool):
        return boolean
    if boolean.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif boolean.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def use_stock_predictor(company_name, start, end, future, metrics, plot, out_dir):
    """Use the LSTMStockPredictor class to predict stock prices."""
    company_name = company_name.strip("'").strip('"')
    print(f"Predicting stock prices for {company_name} using LSTM...")

    stock_predictor = LSTMStockPredictor(company_name, start, end, future_days=future)
    stock_predictor.train()

    # Predict the stock prices for the testing period
    predicted_prices, real_prices = stock_predictor.predict(stock_predictor.test_data)

    # Inverse the scaling to get actual prices
    predicted_prices = stock_predictor.scaler.inverse_transform(predicted_prices.reshape(-1, 1))
    real_prices = stock_predictor.scaler.inverse_transform(real_prices.reshape(-1, 1))

    if metrics:
        # Calculate Mean Squared Error
        mse = mean_squared_error(real_prices, predicted_prices)
        print(f"Mean Squared Error: {mse}")

        # Calculate Mean Absolute Percentage Error
        mape = stock_predictor.calculate_mape(real_prices, predicted_prices)
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

        # Save the results to a file
        output_df = pd.DataFrame({
            'Date': stock_predictor.test_dates[stock_predictor.look_back:],
            'Actual_Close': real_prices.flatten(),
            'Predicted_Close': predicted_prices.flatten()
        })
        output_file = f"{out_dir}/{company_name}_LSTM_Predictions_{str(round(mse, 6))}.xlsx"
        output_df.to_excel(output_file)

    # Plot the results if requested
    if plot:
        stock_predictor.plot_results(real_prices, predicted_prices, out_dir, company_name)

    # Predict future stock prices if required
    if future:
        future_prices = stock_predictor.predict_future_prices()
        print(f"Future predicted prices for the next {future} days: {future_prices}")


def main():
    # Set up argument parser
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-n", "--stock_name", required=True, type=str, help="Stock symbol (e.g., 'AAPL')")
    arg_parser.add_argument("-s", "--start_date", required=True, type=str, help="Start date (yyyy-mm-dd)")
    arg_parser.add_argument("-e", "--end_date", required=True, type=str, help="End date (yyyy-mm-dd)")
    arg_parser.add_argument("-o", "--out_dir", type=str, default=None, help="Directory to save results")
    arg_parser.add_argument("-p", "--plot", type=check_bool, nargs="?", const=True, default=False, help="Plot results")
    arg_parser.add_argument("-f", "--future", type=int, default=None, help="Number of days to predict into the future")
    arg_parser.add_argument("-m", "--metrics", type=check_bool, nargs="?", const=True, default=False, help="Show metrics")

    args = arg_parser.parse_args()

    company_name = args.stock_name
    start = args.start_date
    end = args.end_date
    future = args.future
    metrics = args.metrics
    plot = args.plot
    out_dir = args.out_dir or os.getcwd()

    # Check if we need to do any predictions
    if not metrics and future is None:
        print("Please specify either -m for metrics or -f for future predictions.")
        sys.exit()

    use_stock_predictor(company_name, start, end, future, metrics, plot, out_dir)


if __name__ == "__main__":
    main()