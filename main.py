# <--------------------------------------------------------------------->
# <------Solution for the assignment of Machine Learning Internship----->
# <--------------------------------------------------------------------->

# Machine Learning model for classifying industrial equipment as defective or non-defective.
# I have chosen Support Vector Machine (SVM) algorithm for this problem, however, other algorithms
# like KNN, RandomForest or other algorithms have laso worked.


# importing necessary libraries
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from matplotlib import pyplot as plt


# prepare the data
input_dir = "./"
categories = ["defective", "non-defective"]

data = []
labels = []
for category_idx, category in enumerate(categories):
        # iterating through each image of folders `defective` and `non-defective`
    for file in os.listdir(os.path.join(input_dir, category)):
        # print(file)
        img_path = os.path.join(input_dir, category + "/" + file)
        # converting color images to gray-scale (64-bit floats),
        #  resizing it and then storing in a list
        img = imread(img_path)
        img = resize(img, (80, 80))
        data.append(img.flatten())
        labels.append(category_idx)


# converting that list to a numpy array for compatibilty of SVM input
data = np.asarray(data)
labels = np.asarray(labels)

# storing the input data to a pikle file to avoid further recomputation
pickle.dump(data, open("./data.pkl", "wb"))
pickle.dump(labels, open("./labels.pkl", "wb"))

# splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Model training from scratch --->

# creating an instance of Support Vector Classifier
classifier = SVC()

# creating 12 different models with defferent parametrs
parameters = [{"gamma": [0.01, 0.001, 0.0001], "C": [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)

# training the input data in created model
grid_search.fit(x_train, y_train)

# choosing best among 12 models created above
best_estimator = grid_search.best_estimator_

# test performance
y_prediction = best_estimator.predict(x_test)

# Checking the accuracy of our model, the highest I achieved was around 91%
score = accuracy_score(y_prediction, y_test)
precision = precision_score(y_test, y_prediction)
recall = recall_score(y_test, y_prediction)
print(str(score * 100), "% of samples were correctly classified")

# Saving the best trained model to a pickle file for reusability
pickle.dump(best_estimator, open("./model.pkl", "wb"))

# passing all the test data one by by one, we can also pass individual images manually
for i in range(0, 32):
    # we can see the image passed as input
    plt.imshow(x_test[i].reshape(80, 80, 3))
    plt.show()
    single_pred = best_estimator.predict(x_test[i].reshape(1, 19200))
    if(single_pred == 1):
        print("Its a non-defective image.")
    else:
        print("Its a defctive image.")    

# Due to lack of kind images/data the performance of model is not too good
# but the same model works far better on dog vs cat classification data
# from kaggle.


# <------------------------------------------------------------------------->
# <------------------------------------------------------------------------->
# <------------------------------------------------------------------------->


# <--------------------------------------------------->
# <--- After the model has been trained and saved ---->
# <--------------------------------------------------->


# We just have to uncomment below code and reload all the pickle file and can
# pass data for prediction

# data = pickle.load(open("data.pkl", "rb"))
# labels = pickle.load(open("labels.pkl", "rb"))

# # splitting the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(
#     data, labels, test_size=0.2, shuffle=True, stratify=labels
# )

# # loading the stored trained model
# best_estimator = pickle.load(open("model.pkl", "rb"))

# # # passing all the test data one by by one, we can also pass individual images manually
# for i in range(0, 32):
#     # we can see the image passed as input
#     # plt.imshow(x_test[i].reshape(80, 80, 3))
#     # plt.show()
#     single_pred = best_estimator.predict(x_test[i].reshape(1, 19200))
#     if single_pred == 1:
#         print("Its a non-defective image.")
#     else:
#         print("Its a defctive image.")


# # # test performance
# y_prediction = best_estimator.predict(x_test)

# # Evaluation of model's performance using accuracy, precision, and recall.

# score = accuracy_score(y_test, y_prediction )
# precision = precision_score(y_test, y_prediction)
# recall = recall_score(y_test, y_prediction)

# print(score, precision, recall)
# # Some of previous score, precision and recall in order
# # 0.90625 0.9411764705882353 0.8888888888888888
# # 0.84375 0.8823529411764706 0.8333333333333334
# # 0.84375 1.0 0.7222222222222222

# print(str(score * 100), "% of samples were correctly classified")


# <------------------------------------------------------------------------->
# <------------------------------------------------------------------------->
# <------------------------------------------------------------------------->
# <------------------------------------------------------------------------->


# Above problem could also have been solved with Deep Learnig by using CNN but
# the amount of data I have at his moment is not near sufficient for a Deep Learning
