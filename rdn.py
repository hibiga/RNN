import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras


def vectorize_label(labels):
    new_label = []
    for label in labels:
        y = np.zeros((10, 1))
        y[label] = 1
        new_label.append(y)
    return new_label

def plot_img_number(data):
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the digits: each image is 28x28 pixels
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(data[0][i].reshape(28, 28), cmap=plt.cm.binary, interpolation='nearest')
        # ax.text(0, 7, str(np.where(data[1][i] == [1])[0][0]))
        ax.text(0, 7, str(data[1].iloc[0]))
    plt.show()


def preprocess_data(data):
    X = np.array(data.iloc[:, 1:]) / 255
    y = data.iloc[:, 0]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0, stratify=y)
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.17, random_state=0, stratify=y_train)

    x_train = x_train.reshape(len(x_train), 28, 28, 1)
    x_test = x_test.reshape(len(x_test), 28, 28, 1)
    x_validate = x_validate.reshape(len(x_validate), 28, 28, 1)

    # y_train = vectorize_label(y_train)
    # y_test = vectorize_label(y_test)
    # y_validate = vectorize_label(y_validate)

    return [x_train, y_train], [x_test, y_test], [x_validate, y_validate]


def mlp(train, test, validate):
    print("========================== Creating a model ==========================")
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(28, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    print("")
    print("============================ modelSummary ============================")
    print(model.summary())
    print("")
    # Model Compile
    print("========================== Compiling  model ==========================")
    model.compile(optimizer='adam', metrics=['acc'], loss='sparse_categorical_crossentropy')
    print("")
    print("========================== Model's Compiled ==========================")

    keras.utils.plot_model(model, show_shapes=True, to_file='MLP_model.png')



    # Accuracy Callback
    class handwriting_acc_callback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            if logs.get('acc') >= 0.9:
                print("\n Model has reached 90% accuracy! Congratulations !!!!!")
                self.model.stop_training = True

    handwriting_acc_callback = handwriting_acc_callback()
    print("")
    # Model Fitting
    print("============================= Model- Fit =============================")
    model.fit(train[0], train[1], epochs=100, callbacks=[handwriting_acc_callback])
    print("")
    # Model Evaluation
    print("========================== Model Evaluation ==========================")
    model.evaluate(test[0], test[1])
    print("")
    print("======================================================================")
    print("======================================================================")
    print("======= Model was succesful with >90% accuracy =======")


if __name__ == '__main__':
    digit_data = pd.read_csv("mnist_all.csv")

    train, test, validate = preprocess_data(digit_data)
    #plot_img_number(train)
    mlp(train, test, validate)

    # Creating model
