import modelling as md
import pandas as pd
import feature_engineering as fe
import image_conversion as ic
import os
from tensorflow.keras.models import load_model


PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(PATH, 'data')
IMAGES_PATH = os.path.join(PATH, 'images/train')
MODEL_PATH = os.path.join(PATH, 'models')
data_name = 'DJI_1d_10y_signal.csv'

data = pd.read_csv(os.path.join(DATA_PATH, data_name))

# Create one more df from data for converting images
col_name = ['Date', 'Close']
data_cnn= data.copy()
data_cnn = data_cnn[col_name]
ic.data_to_gaf(data_cnn)


###### CNN Model ########
# train_generator, validation_generator, test_generator = fe.load_data(image_path=IMAGES_PATH)
#
# load = True
# if not load:
#     model = md.fit_cnn(EPOCHS=5, SPLIT=0.2, LR=0.001,
#                        train_generator=train_generator, validation_generator=validation_generator, test_generator=test_generator)
# else:
#     model = load_model('models/cnn_20210424155515_59.38%.h5')
#
# scores = model.evaluate(test_generator, steps=5)
# print("{0}s: {1:.2f}%".format(model.metrics_names[1], scores[1] * 100))


######  LSTM Model ########
train_generator, validation_generator, test_generator = fe.load_time_series_data(image_path=IMAGES_PATH)

model = md.fit_lstm(EPOCHS=1, SPLIT=0.2, LR=0.001,
                       train_generator=train_generator, validation_generator=validation_generator, test_generator=test_generator)

#model = load_model('models/lstm_20210424175024_56.25%.h5')
#scores = model.evaluate(test_generator, steps=5)
#print("{0}s: {1:.2f}%".format(model.metrics_names[1], scores[1] * 100))


# data = data.drop(["Date"], axis=1)
# data = data.ffill(axis=0)
# #print(data)
# xTrain, xTest, yTrain, yTest = fe.ordered_train_test_split(data, "Signal")
#
# results = pd.DataFrame()
# results["true_y"] = yTest


"""
Classification On Close Price
"""
#
# ada = md.fit_ada_boost(xTrain, yTrain, True)
# results["prediction"] = ada.predict(xTest)
# print("Adaboost Classifier")
# print(results["prediction"])
# print(md.get_results(results["true_y"], fe.generate_y(results, "prediction")))
#
# svm = md.fit_SVM(xTrain, yTrain)
# results["prediction"] = svm.predict(xTest)
# print("SVM Classifier")
# print(md.get_results(results["true_y"], results["prediction"]))
# print('Accuracy of the SVM on test set: {:.3f}'.format(svm.score(xTest, yTest)))
#
# knn = md.fit_KNN(xTrain, yTrain, True)
# results["prediction"] = knn.predict(xTest)
# print("KNN")
# print(md.get_results(results["true_y"], results["prediction"]))
# print('Accuracy of the KNN on test set: {:.3f}'.format(knn.score(xTest, yTest)))
#
# rf = md.fit_random_forest(xTrain, yTrain)
# results["prediction"] = rf.predict(xTest)
# print("Random Forest Classifier")
# print(md.get_results(results["true_y"], results["prediction"]))
#
# lr = md.fit_logistic_regression(xTrain, yTrain)
# results["prediction"] = lr.predict(xTest)
# print("Logistic Regression")
# print(md.get_results(results["true_y"], results["prediction"]))
# print('Accuracy of the Logistic Regression on test set: {:.3f}'.format(lr.score(xTest, yTest)))
#
# gb = md.fit_gradient_boosting(xTrain, yTrain, True)
# results["prediction"] = gb.predict(xTest)
# print("Gradient Boosting")
# print(md.get_results(results["true_y"], results["prediction"]))
# print('Accuracy of the GBM on test set: {:.3f}'.format(gb.score(xTest, yTest)))
#
# xTrain_fnn, xTest_fnn, yTrain_fnn, yTest_fnn = md.reconstruct(xTrain, xTest, yTrain, yTest)
# fnn = md.fit_FNN(xTrain_fnn, yTrain_fnn)
# results["prediction"] = fnn.predict(xTest_fnn).flatten()
# print("FNN")
# print(md.get_results(yTest_fnn, results["prediction"]))
# print('Accuracy of the FNN on test set: {:.3f}'.format(md.accuracy_score(yTest_fnn, results["prediction"])))
#
#
# """
# Regression On Close Price
# """
#
# data = pd.read_csv("data/data_reg_1d_10y.csv")
#
# data = data.drop(["Date"], axis=1)
# xTrain, xTest, yTrain, yTest = fe.ordered_train_test_split(data, "Y")
# results = pd.DataFrame()
# results["true_y"] = yTest
#
# knn_reg_close = md.fit_KNN_reg(xTrain, yTrain, True)
# results["prediction"] = knn_reg_close.predict(xTest)
# results["trueSignal"] = fe.generate_y(results, "true_y")
# results["signal"] = fe.generate_y(results, "prediction")
# print("KNN Regressor")
# print(md.get_results(results["trueSignal"], results["signal"]))
#
# gradient_reg_close = md.fit_gradient_boosting_reg(xTrain, yTrain, True)
# results["prediction"] = gradient_reg_close.predict(xTest)
# results["trueSignal"] = fe.generate_y(results, "true_y")
# results["signal"] = fe.generate_y(results, "prediction")
# print("Gradient Boosting Regressor")
# print(md.get_results(results["trueSignal"], results["signal"]))
