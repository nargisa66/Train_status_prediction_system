import pandas as pd
data = pd.read_excel('Book1.xlsx', index_col=0)
print(data.head())


import numpy as np

# Define a function to engineer features
def engineer_features(data):
    # Compute the distance of each station from the starting point
    distances = np.arange(len(data))
    
    # Compute the cumulative delay for each station
    cumulative_delays = data.cumsum(axis=1)
    
    # Combine the features into a single DataFrame
    features = pd.DataFrame({
        'Distance': distances,
        'Cumulative_Delay': cumulative_delays.iloc[:, -1],
        'Delay_Sequence': data.values.tolist()
    }, index=data.index)
    
    return features

# Prepare the features
features = engineer_features(data)

# Display the engineered features
print(features.head())


from sklearn.linear_model import LinearRegression

# Create features for each delay in the sequence
X = pd.DataFrame({
    'Distance': np.repeat(features['Distance'], len(data.columns)),
    'Cumulative_Delay': np.repeat(features['Cumulative_Delay'], len(data.columns)),
    'Date': np.tile(data.columns, len(features))
})

# Flatten the target variable
y_flattened = np.concatenate(features['Delay_Sequence'].values)

# Train the linear regression model
model = LinearRegression()
model.fit(X[['Distance', 'Cumulative_Delay']], y_flattened)

# Once trained, you can use the model to make predictions
# For example, to predict the delay at a new station:
new_station_distance = 5  # Example: distance of a new station from the starting point
new_cumulative_delay = 200  # Example: cumulative delay of the train up to the new station
predicted_delay = model.predict([[new_station_distance, new_cumulative_delay]])

print("Predicted delay at the new station:", predicted_delay)



# Function to predict delay at next stations given current station and delay
def predict_delay(model, current_station, current_delay, features):
    # Find the index of the current station in the features DataFrame
    current_station_index = features.index.get_loc(current_station)
    
    # Get the features for stations after the current station
    features_after_current = features.iloc[current_station_index+1:]
    
    # Calculate the distance of each station from the current station
    distances = np.abs(np.arange(len(features_after_current)))
    
    # Calculate the cumulative delay for each station
    cumulative_delays = current_delay + features_after_current['Cumulative_Delay']
    
    # Prepare the input features for prediction
    X = pd.DataFrame({
        'Distance': distances,
        'Cumulative_Delay': cumulative_delays
    })
    
    # Predict the delay at each station using the trained model
    predicted_delays = model.predict(X)
    
    return predicted_delays, features_after_current.index

# Example usage:
#current_station = 'Waria'  # Example: current station reached by the train
#current_delay = 10  # Example: current delay of the train in minutes
current_station = input("Enter the current station reached by the train: ")
current_delay = float(input("Enter the current delay of the train in minutes: "))
# Predict the delay at the next stations
predicted_delays, next_stations = predict_delay(model, current_station, current_delay, features)

# Display the predicted delays for stations after the current station
print("Predicted delays at the next stations:")
for station, delay in zip(next_stations, predicted_delays):
    print(station + ":", delay, "minutes")
