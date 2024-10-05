import numpy as np
import pandas as pd
from sklearn import *
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('infantset.csv')
#Read the dataset into a dataframe
df = pd.read_csv('infantset.csv')


#Mapping of Birth Weight
df["BirthWeight"] = df["BirthWeight"].map({'WeightTooLow':3 ,'LowWeight':2,'NormalWeight':1})


#Mapping of Family History
df["FamilyHistory"] = df["FamilyHistory"].map({'AboveTwoCases':3 ,'ZeroToTwoCases':2,'NoCases':1})


#Mapping of Preterm Birth
df["PretermBirth"] = df["PretermBirth"].map({'4orMoreWeeksEarlier':3 ,'2To4weeksEarlier':2,'NotaPreTerm':1})


#Mapping of Heart Rate
df["HeartRate"] = df["HeartRate"].map({'RapidHeartRate':3 ,'HighHeartRate':2,'NormalHeartRate':1})


#Mapping of Breathing Difficulty
df["BreathingDifficulty"] = df["BreathingDifficulty"].map({'HighBreathingDifficulty':3 ,'BreathingDifficulty':2,'NoBreathingDifficulty':1})


#Mapping of Skin Tinge
df["SkinTinge"] = df["SkinTinge"].map({'Bluish':3 ,'LightBluish':2,'NotBluish':1})


#Mapping of Responsiveness
df["Responsiveness"] = df["Responsiveness"].map({'UnResponsive':3 ,'SemiResponsive':2,'Responsive':1})


#Mapping of Movement
df["Movement"] = df["Movement"].map({'Diminished':3 ,'Decreased':2,'NormalMovement':1})


#Mapping of Delivery Type
df["DeliveryType"] = df["DeliveryType"].map({'C_Section':3 ,'DifficultDelivery':2,'NormalDelivery':1})


#Mapping of Mothers BP History
df["MothersBPHistory"] = df["MothersBPHistory"].map({'VeryHighBP':3 ,'HighBP':2,'BPInRange':1})


#Mapping of Cardiac Arrest Chance
df["CardiacArrestChance"] = df["CardiacArrestChance"].map({'High':2 ,'Medium':1,'Low':0})


#Creation of data as numpy array
data = df[["BirthWeight","FamilyHistory","PretermBirth","HeartRate","BreathingDifficulty","SkinTinge","Responsiveness","Movement","DeliveryType","MothersBPHistory","CardiacArrestChance"]].to_numpy()


#All columns except last column are considered as inputs
inputs = data[:,:-1]


#Last Column is considered as outputs
outputs = data[:, -1]

training_inputs = inputs[:1000]
training_outputs = outputs[:1000]
testing_inputs = inputs[1000:]
testing_outputs = outputs[1000:]
classifier = BaggingClassifier()
classifier.fit(training_inputs, training_outputs)
#testSet = [[3,1,1,1,2,1,1,1,1,2]]
#test = pd.DataFrame(testSet)
#predictions = classifier.predict(test)
#print('BC prediction on the first test set is:',predictions)
#testSet = [[2,2,1,2,3,1,2,3,1,1]]
#test = pd.DataFrame(testSet)
#predictions = classifier.predict(test)
#print('BC prediction on the second test set is:',predictions)
#testSet = [[3,2,2,1,3,3,1,3,3,3]]
#test = pd.DataFrame(testSet)
#predictions = classifier.predict(test)
#print('BC prediction on the third test set is:',predictions)
# Define a function to get dynamic test inputs
def get_dynamic_input():
    print("Mapping of Birth Weight:")
    print("- WeightTooLow:3, LowWeight:2, NormalWeight:1")

    print("\nMapping of Family History:")
    print("- AboveTwoCases:3, ZeroToTwoCases:2, NoCases:1")

    print("\nMapping of Preterm Birth:")
    print("- 4orMore WeeksEarlier:3, 2To4 weeksEarlier:2, Not a PreTerm:1")

    print("\nMapping of Heart Rate:")
    print("- RapidHeartRate:3, HighHeartRate:2, NormalHeartRate:1")

    print("\nMapping of Breathing Difficulty:")
    print("- HighBreathingDifficulty:3, BreathingDifficulty:2, NoBreathingDifficulty:1")

    print("\nMapping of Skin Tinge:")
    print("- Bluish:3, LightBluish:2, NotBluish:1")

    print("\nMapping of Responsiveness:")
    print("- UnResponsive:3, SemiResponsive:2, Responsive:1")

    print("\nMapping of Movement:")
    print("- Diminished:2, Decreased:2, NormalMovement:1")

    print("\nMapping of Delivery Type:")
    print("- C-Section:3, DifficultDelivery:2, NormalDelivery:1")

    print("\nMapping of Mothers BP History:")
    print("- VeryHighBP:3, HighBP:2, BPInRange:1")

    print("\nMapping of Cardiac Arrest Chance:")
    print("- High:3, Medium:2, Low:1")
    input_values = []
    for i in range(len(df.columns) - 1):
        value = input(f"Enter value for {df.columns[i]}: ")
        input_values.append(float(value))
    return [input_values]

# Make predictions on dynamic test inputs
testSet = get_dynamic_input()
test = pd.DataFrame(testSet)
# Define a dictionary to map numeric predictions to labels
prediction_labels = {0: 'Low', 1: 'Medium', 2: 'High'}

# Make predictions on dynamic test inputs
predictions = classifier.predict(test)

# Convert numeric predictions to labels
predictions_labels = [prediction_labels[prediction] for prediction in predictions]

# Print predictions
print('BC prediction on effect of cardiac arrest is', predictions_labels)