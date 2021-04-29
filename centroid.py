import cv2
import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Extracting & reshaping image to 28 * 28 pixels aspect
def ExtractAndReshape(imagesPath, labelsPath):
    images, images_labels = loadlocal_mnist(
        images_path=imagesPath,
        labels_path=labelsPath)

    images = images.reshape(len(images), 28, 28)
    images = np.array(images)
    return images, images_labels

# Split a matrix into sub-matrices.
def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))

# The main function to get the 
def centerOfMass(matrix):
    Sumx, Sumy, num = 0, 0, 0
    cx, cy = 0, 0

    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
            if matrix[i][j] != 0:  # if pixel has a value
                Sumx += i  # add index of x-axis
                Sumy += j  # add index of y-axis
                num += 1  # get their number

    if num != 0:
        cx = Sumx / num  # get point of centroid on x-axis
        cy = Sumy / num  # get point of centroid on y-axis

    return cx, cy


def extractFeatures(imagesList):
    inputFeatures = []

    for image in imagesList:
        featureVector = []
        # split matrix into 16 sub matrices
        for matrix in split(image, 7, 7):
            cx, cy = centerOfMass(matrix)
            # collect centroid of sub matrices into one list
            featureVector.append((cx, cy))
        inputFeatures.append(featureVector)  # get feature vector of all images

    print(inputFeatures)
    return np.array(inputFeatures)


def applyKNN(trainFeatures, trainLabels, testFeatures):
    print('Wait for fitting knn classifier and testing it...')
    # Apply KNN classifier with k = 5
    knn = KNeighborsClassifier(5, metric='euclidean')
    knn.fit(trainFeatures, trainLabels)  # fit train data
    prediction = knn.predict(testFeatures)  # test data
    return prediction


def second_model(trainFeatures, testFeatures, trainLabels, testLabels):
    train_inputs = extractFeatures(trainFeatures)
    test_inputs = extractFeatures(testFeatures)

    train_inputs = train_inputs.reshape(60000, 32)
    test_inputs = test_inputs.reshape(10000, 32)

    prediction = applyKNN(train_inputs, trainLabels, test_inputs)
    return accuracy_score(testLabels, prediction)


def main():
    train_features, train_labels = ExtractAndReshape(
        "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    test_features, test_labels = ExtractAndReshape(
        "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

    accuracyScore = second_model(
        train_features, test_features, train_labels, test_labels)

    print("Accuracy Score =", accuracyScore * 100, "%")


main()
