import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the dataset
data = pd.read_csv("D:/Education/PhD/ZZU/Research/FRP/On Going Papers/Paper ( 1 Multi Target RCA)/All Tables and graphs/Database1CSV.csv")

# Features (X) and Targets (y)
X = data.drop(['FC'], axis=1)  # Features
y = data['FC']  # Target (FC)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a regression model (Decision Tree Regressor)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predict on the training set
train_predictions = model.predict(X_train)
# Predict on the test set
test_predictions = model.predict(X_test)

# Calculate MAE, MSE, RMSE, R-squared for training
mae_train = mean_absolute_error(y_train, train_predictions)
mse_train = mean_squared_error(y_train, train_predictions)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, train_predictions)

# Calculate MAE, MSE, RMSE, R-squared for testing
mae_test = mean_absolute_error(y_test, test_predictions)
mse_test = mean_squared_error(y_test, test_predictions)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, test_predictions)

# Print the results
print("Training Metrics (FC):")
print(f'MAE: {mae_train}')
print(f'MSE: {mse_train}')
print(f'RMSE: {rmse_train}')
print(f'R-squared: {r2_train}')

print("\nTesting Metrics (FC):")
print(f'MAE: {mae_test}')
print(f'MSE: {mse_test}')
print(f'RMSE: {rmse_test}')
print(f'R-squared: {r2_test}')

# Ask the user whether to save the model
save_model = input("Do you want to save the trained model? Enter 'yes' or 'no': ").lower()

if save_model == 'yes':
    # Create a directory if it doesn't exist
    save_dir = r"D:/Education/PhD/ZZU/Research/FRP/On Going Papers/Paper ( 1 Multi Target RCA)/Algorithms/Decision tree results"
    os.makedirs(save_dir, exist_ok=True)

    # Save the trained model
    joblib.dump(model, os.path.join(save_dir, "your_saved_model_DecisionTree.joblib"))
    print("Model saved successfully.")

    # Save actual and predicted values to Excel for training and testing sets
    output_excel_path = os.path.join(save_dir, "actual_vs_predicted_results.xlsx")
    with pd.ExcelWriter(output_excel_path) as writer:
        # Training set
        pd.DataFrame({'Actual_FC': y_train, 'Predicted_FC': train_predictions}).to_excel(writer, sheet_name='Training_FC', index=False)
        # Testing set
        pd.DataFrame({'Actual_FC': y_test, 'Predicted_FC': test_predictions}).to_excel(writer, sheet_name='Testing_FC', index=False)
    print("Actual vs predicted results for both training and testing sets saved to Excel successfully.")

else:
    print("Model not saved.")
