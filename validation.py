import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from statistics import *
import Breast_Cancer_Classification_ResNet50
import Breast_Cancer_Classification_VGG16


def validation(images_train_val, labels_train_val, images_test, labels_test, model_type, k):
    """Validate the model and plot the ROC curve.

           Parameters:
               images_train_val (np.array): Contains the pixel arrays of the training and validation images.
               images_test (np.array): Contains the pixel arrays of the test images.
               labels_train_val (np.array): Array of labels for the training and validation images.
               labels_test (np.array): Array of labels for the test images.
               model_type (str): Name of the model to be validated. One of "VGG16","ResNet50".
               k (int): number of different training runs.

           Returns:
                A numpy array with the mean AUC and accuracy of the model in the different runs with random
                train_test_splits.
    """

    results_auc = []
    results_acc = []
    
    for i in range(k):
        assert model_type in ['VGG16', 'ResNet50']
        if model_type == "VGG16":
            model = Breast_Cancer_Classification_VGG16.vgg16()
        else:
            # mode == 'ResNet50'
            model = Breast_Cancer_Classification_ResNet50.resnet50()

        X_train, X_val, Y_train, y_val = train_test_split(images_train_val, labels_train_val, test_size=0.125,
                                                          random_state=None)
        
        callback = EarlyStopping(monitor='val_loss', mode='auto', patience=3, verbose=1, start_from_epoch=5,
                                 restore_best_weights=True)

        if model_type == "VGG16":
            batch_size = 1
        else:
            # mode == 'ResNet50'
            batch_size = 32

        model.fit(
            X_train,
            Y_train,
            epochs=50,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[callback],
            verbose=2
        )

        predictions = model.predict(images_test)

        if model_type == "ResNet50":
            # Convert multilabel-indicator format to binary labels for the roc function
            # to_categorical([0, 1], 2) = [[1. 0.], [0. 1.]]
            # Store the probability estimates of the positive class
            predictions = [x[1] for x in predictions]
            # Convert one-hot encoded labels back to integer labels
            labels_test_binary = [0 if x[0] == 1 else 1 for x in labels_test]
        else:
            predictions = predictions.reshape(-1)
            labels_test_binary = labels_test

        fpr, tpr, thresholds = metrics.roc_curve(labels_test_binary, predictions)
        label = "model " + str(i)
        plt.plot(fpr, tpr, label=label)

        # compute mean accuracy and mean auc
        result = model.evaluate(images_test, labels_test)
        results_auc.append(result[2])
        results_acc.append(result[1])
        
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Random classifier')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    title = 'ROC Curve on the test set with ' + str(k) + ' different train/validation splits' 
    plt.title(title)
    plt.legend(loc='lower right')
    
    acc = fmean(results_acc)
    auc = fmean(results_auc)
    plt.savefig("./roc_test.pdf")
    print("accuracy of all runs:")
    print(results_acc)
    print("The mean accuracy is:")
    print(acc)
    print("auc of all runs:")
    print(results_auc)
    print("The mean auc is:")
    print(auc)
    return acc, auc
