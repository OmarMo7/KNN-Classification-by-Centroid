import numpy as np

# Split a matrix into sub-matrices.


def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


# Getting matrices' centers of each image


def centerOfMass(matrix):
    Sumx, Sumy, num = 0, 0, 0
    cx, cy = 0, 0
    # Here we get the centriod of each matrix
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
            if matrix[i][j] != 0:  # if pixel has a value
                Sumx += i  # add index of x-axis
                Sumy += j  # add index of y-axis
                num += 1  # get their number

    # Here we calculate the centoid of all centroids of matrices [The whole image]
    if num != 0:
        cx = Sumx / num  # get point of centroid on x-axis
        cy = Sumy / num  # get point of centroid on y-axis

    return cx, cy


def collectCentroids(imagesList):
    inputFeatures = []

    for image in imagesList:
        featureVector = []
        # split matrix into 16 sub matrices
        for matrix in split(image, 7, 7):
            cx, cy = centerOfMass(matrix)
            # collect centroid of sub matrices into one list
            featureVector.append((cx, cy))
        inputFeatures.append(featureVector)  # get feature vector of all images

    # print(inputFeatures)
    return np.array(inputFeatures)
