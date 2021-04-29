import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import accuracy_score
from centroid import collectCentroids
# Extracting & reshaping image to 28 * 28 pixels aspect


def ExtractAndReshape(imagesPath, labelsPath):
    images, images_labels = loadlocal_mnist(
        images_path=imagesPath,
        labels_path=labelsPath)

    images = images.reshape(len(images), 28, 28)
    images = np.array(images)
    return images, images_labels


def applyKNN(trainFeatures, trainLabels, testFeatures):
    print('...')
    print('...')
    print('...')
    # Apply KNN classifier with k = 5
    knn = KNeighborsClassifier(5, metric='euclidean')
    knn.fit(trainFeatures, trainLabels)
    prediction = knn.predict(testFeatures)
    return prediction


def calculateAccuracy(trainFeatures, testFeatures, trainLabels, testLabels):
    train_inputs = collectCentroids(trainFeatures)
    test_inputs = collectCentroids(testFeatures)

    train_inputs = train_inputs.reshape(60000, 32)
    test_inputs = test_inputs.reshape(10000, 32)

    prediction = applyKNN(train_inputs, trainLabels, test_inputs)
    return accuracy_score(testLabels, prediction)


train_features, train_labels = ExtractAndReshape(
    "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
test_features, test_labels = ExtractAndReshape(
    "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

accuracyScore = calculateAccuracy(
    train_features, test_features, train_labels, test_labels)

print("Accuracy Score: ", accuracyScore * 100, "%")
