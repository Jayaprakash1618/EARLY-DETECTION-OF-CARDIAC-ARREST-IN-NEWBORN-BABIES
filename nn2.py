#Importing Numpy
import numpy as np


#Importing Pandas
import pandas as pd

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



#Importing Matpplotlib
import matplotlib.pyplot as plt



#Importing Tensorflow
import tensorflow as tf


#Importing Keras backend
import tensorflow.keras.backend as K



#Print Tensorflow Version
print(tf.__version__)



#Importing Warnings
import warnings

#Ignoring Warnings
warnings.filterwarnings("ignore")





#First Thousand rows are considered for training.
training_data = inputs[:1000]


#Training labels are set to the last column values of first thousand rows
training_labels = outputs[:1000]



#Remaining Rows, Beyond 1000 are considered for testing
test_data = inputs[1000:]


#Testing labels are set to the last column values of remaining rows
test_labels = outputs[1000:]


#Tensorflow Initiation
tf.keras.backend.clear_session()




#Configure the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.elu), 
                                    tf.keras.layers.Dense(64, activation=tf.nn.selu), 
                                    tf.keras.layers.Dense(32, activation=tf.nn.softmax), 
                                    tf.keras.layers.Dense(16, activation=tf.nn.softplus)])
									
									
#Comiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



#Creation of the model
model.fit(training_data, training_labels, epochs=100)


#First Test Set Assinment
'''testSet = [[3,1,1,1,2,1,1,1,1,2]]


#First Test Set Conversion to Pandas Data Frame
test = pd.DataFrame(testSet)


#Prediction on First Test Set Using the Model
predictions = model.predict(test)


#Finding the first test set label
classes=np.argmax(predictions,axis=1)


#printing the first test set label
print('DL Model Prediction on the first test set is:',classes)


#Second Test Set Assinment
testSet = [[2,2,1,2,3,1,2,3,1,1]]


#Second Test Set Conversion to Pandas Data Frame
test = pd.DataFrame(testSet)


#Prediction on Second Test Set Using the Model
predictions =  model.predict(test)


#Finding the second test set label
classes=np.argmax(predictions,axis=1)


#printing the second test set label
print('DL Model Prediction on the second test set is:',classes)

#Third Test Set Assinment
testSet = [[3,2,2,1,3,3,1,3,3,3]]


#Third Test Set Conversion to Pandas Data Frame
test = pd.DataFrame(testSet)


#Prediction on Third Test Set Using the Model
predictions =  model.predict(test)


#Finding the Third test set label
classes=np.argmax(predictions,axis=1)


#printing the Third test set label
print('DL Model Prediction on the Third test set is:',classes)'''
# Define the model (you need to define 'model' somewhere in your code)
# model = ...

# Iterate over each test set assignment
for i in range(3):
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
    # Prompt the user to input values for the test set
    print(f"Enter values for Test Set {i+1}:")
    testSet = []
'''  for j in range(10):
        value = float(input(f"Enter value for feature {j+1}: "))
        testSet.append(value)
    
    # Convert the test set to a Pandas DataFrame
    test = pd.DataFrame([testSet])
    
    # Make predictions on the test set using the model
    predictions = model.predict(test)
    
    # Find the predicted label for the test set
    classes = np.argmax(predictions, axis=1)
    
    # Print the predicted label
    print(f'DL Model Prediction on Test Set {i+1} is:', classes)'''
# Define a dictionary to map numeric predictions to labels
prediction_labels = {0: 'Low', 1: 'Medium', 2: 'High'}

# Iterate over each test set assignment
for i in range(3):
    # Prompt the user to input values for the test set
    print(f"Enter values for Test Set {i+1}:")
    testSet = []
    for j in range(10):
        value = float(input(f"Enter value for feature {j+1}: "))
        testSet.append(value)
    
    # Convert the test set to a Pandas DataFrame
    test = pd.DataFrame([testSet])
    
    # Make predictions on the test set using the model
    predictions = model.predict(test)
    
    # Find the predicted label for the test set
    classes = np.argmax(predictions, axis=1)
    
    # Convert numeric predictions to labels
    classes_labels = [prediction_labels[class_] for class_ in classes]
    
    # Print the predicted label
    print(f'DL Model Prediction on Test Set {i+1} is:', classes_labels)