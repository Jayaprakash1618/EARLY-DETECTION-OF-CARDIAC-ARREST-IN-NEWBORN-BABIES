import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix

# Function to calculate performance metrics
def calculate_metrics(true_labels, predicted_labels):
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    tn, fp, fn, tp = cm.ravel()

    # Delta-p value
    delta_p = abs((tp / (tp + fp)) - (tp / (tp + fn)))

    # False Discovery Rate (FDR)
    fdr = fp / (tp + fp)

    # False Omission Rate (FOR)
    for_ = fn / (fn + tn)

    # Prevalence threshold
    pt = (tp + fn) / (tp + tn + fp + fn)

    # Critical Success Index (CSI)
    csi = tp / (tp + fn + fp)

    return delta_p, fdr, for_, pt, csi

# Read the dataset into a dataframe
df = pd.read_csv('infantset.csv')

# Mapping of categorical variables
category_mapping = {
    'BirthWeight': {'WeightTooLow': 3, 'LowWeight': 2, 'NormalWeight': 1},
    'FamilyHistory': {'AboveTwoCases': 3, 'ZeroToTwoCases': 2, 'NoCases': 1},
    'PretermBirth': {'4orMoreWeeksEarlier': 3, '2To4weeksEarlier': 2, 'NotaPreTerm': 1},
    'HeartRate': {'RapidHeartRate': 3, 'HighHeartRate': 2, 'NormalHeartRate': 1},
    'BreathingDifficulty': {'HighBreathingDifficulty': 3, 'BreathingDifficulty': 2, 'NoBreathingDifficulty': 1},
    'SkinTinge': {'Bluish': 3, 'LightBluish': 2, 'NotBluish': 1},
    'Responsiveness': {'UnResponsive': 3, 'SemiResponsive': 2, 'Responsive': 1},
    'Movement': {'Diminished': 3, 'Decreased': 2, 'NormalMovement': 1},
    'DeliveryType': {'C_Section': 3, 'DifficultDelivery': 2, 'NormalDelivery': 1},
    'MothersBPHistory': {'VeryHighBP': 3, 'HighBP': 2, 'BPInRange': 1},
    'CardiacArrestChance': {'High': 1, 'Medium': 1, 'Low': 0}  # Adjusted mapping
}

# Apply mapping
for column, mapping in category_mapping.items():
    df[column] = df[column].map(mapping)

# Convert dataframe to numpy array
data = df.to_numpy()

# Split data into inputs and outputs
inputs = data[:, :-1]
outputs = data[:, -1]

# Split data into training and testing sets
training_inputs = inputs[:1000]
training_outputs = outputs[:1000]
testing_inputs = inputs[1000:]
testing_outputs = outputs[1000:]

# Initialize and train Bagging Classifier
classifier = BaggingClassifier()
classifier.fit(training_inputs, training_outputs)

# Evaluate performance metrics for training region
train_predictions = classifier.predict(training_inputs)
delta_p_train, fdr_train, for_train, pt_train, csi_train = calculate_metrics(training_outputs, train_predictions)

# Evaluate performance metrics for testing region
test_predictions = classifier.predict(testing_inputs)
delta_p_test, fdr_test, for_test, pt_test, csi_test = calculate_metrics(testing_outputs, test_predictions)

# Print performance metrics for training region
print("Performance Metrics for Training Region:")
print("Delta-p value (Tr):", delta_p_train)
print("False Discovery Rate (FDR) (Tr):", fdr_train)
print("False Omission Rate (FOR) (Tr):", for_train)
print("Prevalence Threshold (Tr):", pt_train)
print("Critical Success Index (CSI) (Tr):", csi_train)
print()

# Print performance metrics for testing region
print("Performance Metrics for Testing Region:")
print("Delta-p value (Ts):", delta_p_test)
print("False Discovery Rate (FDR) (Ts):", fdr_test)
print("False Omission Rate (FOR) (Ts):", for_test)
print("Prevalence Threshold (Ts):", pt_test)
print("Critical Success Index (CSI) (Ts):", csi_test)
