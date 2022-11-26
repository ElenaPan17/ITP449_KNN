""" Elena Pan
    ITP-449
    Assignment 8
    Diabetes and KNN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def main():
    # create a dataframe to store the diabetes data 
    file = 'diabetes.csv'
    df_diabetes = pd.read_csv(file)

    # determine the dimensions of df_diabetes: 768 rows x 9 columns
    print(df_diabetes.shape)
    # (768 , 9)

    # update the dataframe to account for missing values
    # print(df_diabetes.isnull().sum())
    # there are no missing values 

    # create the feature matrix (x) and Target Vector(y) 
    y = df_diabetes['Outcome']
    x = df_diabetes.drop(columns = 'Outcome')

    # standardize the attritues of feature matrix
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    
    # split the Feature Matrix and Target Vector into three partitions
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=42, stratify=y_temp)

    # develop a KNN model based on training for various k in the range of 1 to 30
    ks = range(1, 31)
    accuracy_train = []
    accuracy_valid = []
    for k in ks:
        model_knn = KNeighborsClassifier(n_neighbors=k)
        model_knn.fit(X_train, y_train)
        # compute the accuracy for both traning and validation for those ks
        accuracy_train.append(model_knn.score(X_train, y_train))
        accuracy_valid.append(model_knn.score(X_valid, y_valid))

    # plot the training and validation accuracy vs k and determine the best value of k
    plt.plot(ks, accuracy_train, label='Training accuracy')
    plt.plot(ks, accuracy_valid, label='Validation accuracy')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('KNN: Accuracy for various ks')
    plt.legend()
    plt.tight_layout()
    plt.savefig('KNN.png')

    # the best k is 7
    # use k = 7, score the test data set
    model_knn = KNeighborsClassifier(n_neighbors=7)
    model_knn.fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('The accuracy for the testing dataset is: ', accuracy)

    # plot the confusion matrix as a figuer
    cm = confusion_matrix(y_test, y_pred)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_knn.classes_)
    fig, ax1 = plt.subplots()
    cm_disp.plot(ax=ax1)
    plt.suptitle('Confusion Matrix of Diabetes Dataset')
    plt.savefig('confusion matrix.png')

    # predict the Outcome for a person 
    # 2 pregnancies, 150 glucose, 85 blood pressure, 22 skin thickness, 200 insulin
    # 30 BMI, 0.3 diabetes predigree, 55 age
    df_new = pd.DataFrame([[2, 150, 85, 22, 200, 30, 0.3, 55]], columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    df_new = pd.concat([df_new, x], ignore_index = True)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df_new), columns=df_new.columns)
    pred = model_knn.predict(X)
    print('the Outcome for a person with 2 pregnancies, 150 glucose, 85 blood pressure, 22 skin thickness, 200 insulin, 30 BMI, 0.3 diabetes pedigree, 55 age is: ', pred[0])



if __name__ == '__main__':
    main()
