# Import necessary modules
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define the Predictor class
class TempModel:
    def _init_(self, num_days, weight_values=None):
        self.num_days = num_days
        self.weight_values = [1.] * num_days if weight_values is None else weight_values

    def calculate_weighted_avg(self, temperatures):
        """Calculate weighted average of the given temperatures."""
        weighted_sum = sum(temperatures[i] * self.weight_values[i] for i in range(len(temperatures)))
        total_weights = sum(self.weight_values)
        return weighted_sum / total_weights

    def forecast(self, temp_seq):
        """Predict temperatures for upcoming days based on previous num_days."""
        if len(temp_seq) < self.num_days:
            raise ValueError("The sequence must be at least as long as num_days.")
        
        return [self.calculate_weighted_avg(temp_seq[i:i+self.num_days]) for i in range(len(temp_seq) - self.num_days)]

    def adjust_weights(self, updated_weights):
        if len(updated_weights) != self.num_days:
            raise ValueError("New weights must match num_days length.")
        self.weight_values = updated_weights

# Evaluation function for Mean Squared Error (MSE)
def calculate_mse(true_values, predictions):
    squared_errors = [(true_values[i] - predictions[i]) ** 2 for i in range(len(true_values))]
    return sum(squared_errors) / len(true_values)

# Load temperature datasets
file_reykjavik = "C:/Users/vasistha/Data,Stats and info/Assignment-1/HW1_data/Reykjavik_temps_2020.pkl"
file_capetown = "C:/Users/vasistha/Data,Stats and info/Assignment-1/HW1_data/CapeTown_temps_2020.pkl"

with open(file_reykjavik, 'rb') as f:
    temps_reykjavik = pickle.load(f)

with open(file_capetown, 'rb') as f:
    temps_capetown = pickle.load(f)

# Split data into training, validation, and testing sets
train_reykjavik, valid_reykjavik, test_reykjavik = temps_reykjavik[:50], temps_reykjavik[50:70], temps_reykjavik[70:]
train_capetown, valid_capetown, test_capetown = temps_capetown[:50], temps_capetown[50:70], temps_capetown[70:]

# Model setup and predictions
num_days = 3
model_reykjavik = TempModel(num_days)

# Reykjavik model training
reykjavik_preds = model_reykjavik.forecast(train_reykjavik)
reykjavik_train_mse = calculate_mse(train_reykjavik[num_days:], reykjavik_preds)
print(f"Reykjavik Training MSE: {round(reykjavik_train_mse, 2)}")

# Cape Town model training
capetown_preds = model_reykjavik.forecast(train_capetown)
capetown_train_mse = calculate_mse(train_capetown[num_days:], capetown_preds)
print(f"Cape Town Training MSE: {round(capetown_train_mse, 2)}")

# Visualization of predictions vs. actual data
def plot_temperature(data, predictions, num_days):
    plt.figure(figsize=(10,6))
    x_axis = range(len(data))
    plt.plot(x_axis, data, label='Actual', marker='o')
    plt.plot(x_axis[num_days:], predictions, label='Predicted', marker='x', linestyle='dashed')
    plt.xlabel("Days")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_temperature(train_reykjavik, reykjavik_preds, num_days)
plot_temperature(train_capetown, capetown_preds, num_days)

# Adjusting weights and retraining the model
model_reykjavik.adjust_weights([0.2, 0.3, 0.9])

reykjavik_preds_tuned = model_reykjavik.forecast(train_reykjavik)
reykjavik_tuned_mse = calculate_mse(train_reykjavik[num_days:], reykjavik_preds_tuned)
print(f"Tuned Reykjavik Training MSE: {round(reykjavik_tuned_mse, 2)}")

capetown_preds_tuned = model_reykjavik.forecast(train_capetown)
capetown_tuned_mse = calculate_mse(train_capetown[num_days:], capetown_preds_tuned)
print(f"Tuned Cape Town Training MSE: {round(capetown_tuned_mse, 2)}")

# Validation and Model Selection
print('Validating Reykjavik Model')
valid_reykjavik_preds = model_reykjavik.forecast(valid_reykjavik)
valid_reykjavik_mse = calculate_mse(valid_reykjavik[num_days:], valid_reykjavik_preds)
print(f"Reykjavik Validation MSE: {round(valid_reykjavik_mse, 2)}")

print('Validating Cape Town Model')
valid_capetown_preds = model_reykjavik.forecast(valid_capetown)
valid_capetown_mse = calculate_mse(valid_capetown[num_days:], valid_capetown_preds)
print(f"Cape Town Validation MSE: {round(valid_capetown_mse, 2)}")

# Testing the selected model on the test set
print('Testing Reykjavik Model')
test_reykjavik_preds = model_reykjavik.forecast(test_reykjavik)
test_reykjavik_mse = calculate_mse(test_reykjavik[num_days:], test_reykjavik_preds)
print(f"Reykjavik Test MSE: {round(test_reykjavik_mse, 2)}")

print('Testing Cape Town Model')
test_capetown_preds = model_reykjavik.forecast(test_capetown)
test_capetown_mse = calculate_mse(test_capetown[num_days:], test_capetown_preds)
print(f"Cape Town Test MSE: {round(test_capetown_mse, 2)}")

# Plot test results
plot_temperature(test_reykjavik, test_reykjavik_preds, num_days)
plot_temperature(test_capetown, test_capetown_preds, num_days)

