import tkinter as tk
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

frame = tk.Tk()
frame.title("App for NN")
frame.geometry('400x200')

label = tk.Label(frame, text = "")
# pack() - showing the label to the user
label.pack()

# provide a new function where user will provide the no of epochs
def main():
    # input from the user (only takes input from the first object without new line mark)
    inp = int(numberofep.get(1.0,"end-1c"))
    label.config(text = "Number of iterations provided: {inp}")
    data = pd.read_csv("DSP_1.csv")

    cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    data = data[cols].copy()

    data["Age"].fillna((data["Age"].mean()), inplace=True)
    data.dropna(subset=['Embarked'], inplace=True)

    encoder = LabelEncoder()

    data.loc[:,"Sex"] = encoder.fit_transform(data.loc[:,"Sex"])
    data.loc[:,"Embarked"] = encoder.fit_transform(data.loc[:,"Embarked"])

    y_train = data.iloc[0:600:,0]
    y_test = data.iloc[600:700:,0]
    X_train = data.iloc[0:600:,1:8]
    X_test = data.iloc[600:700:,1:8]

# the fix that was needed, changing type to float and/or reshaping to a NumPy array
    X_train = X_train.to_numpy().astype('float32')
    y_train = y_train.values.reshape((-1, 1)).astype('float32')

    model = Sequential()

    if int(inp) > 300:
        model.add(Dense(20, kernel_initializer = "uniform", activation="relu", input_dim = 7))
        model.add(Dense(10, kernel_initializer = "uniform", activation="relu"))
        model.add(Dense(8, kernel_initializer = "uniform", activation="relu"))
        model.add(Dense(8, kernel_initializer = "uniform", activation="relu"))
        model.add(Dense(6, kernel_initializer = "uniform", activation="relu"))
        model.add(Dense(1, kernel_initializer = "uniform", activation="sigmoid"))
        model.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ['accuracy'])
        model.fit(X_train, y_train, batch_size = 30, epochs = int(inp))
        print(model.summary())
    else:
        model.add(Dense(10, kernel_initializer = "uniform", activation="relu", input_dim = 7))
        model.add(Dense(7, kernel_initializer = "uniform", activation="relu"))
        model.add(Dense(7, kernel_initializer = "uniform", activation="relu"))
        # it has to be 1 cause we only want to provide one thing (if survived)
        model.add(Dense(1, kernel_initializer = "uniform", activation="sigmoid"))
        model.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ['accuracy'])
        model.fit(X_train, y_train, batch_size = 30, epochs = int(inp))
        print(model.summary())
    model.save('model')

numberofep = tk.Text(
    frame,
    height = 5,
    width = 20
)
numberofep.pack()

newButton = tk.Button(frame, text = "Go!", command = main)
newButton.pack()
frame.mainloop()
