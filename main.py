import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import urllib

# Wrap the existing code in a function
def heart_disease_classification(file_path):
    output = ""
    # Reading the data
    df = pd.read_csv(file_path)

    # Exploratory data analysis
    df.head()
    df.target.value_counts()
    sns.countplot(x="target", data=df)
    plt.savefig('countplot.png')
    plt.close()

    countNoDisease = len(df[df.target == 0])
    countHaveDisease = len(df[df.target == 1])
    output += "Percentage of Patients Without Heart Disease: {:.2f}%".format((countNoDisease / len(df.target)) * 100)
    output += "\n\nPercentage of Patients With Heart Disease: {:.2f}%".format((countHaveDisease / len(df.target)) * 100)

    sns.countplot(x='sex', data=df)
    plt.xlabel("Sex (0 = female, 1 = male)")
    plt.savefig('sex_countplot.png')
    plt.close()

    countFemale = len(df[df.sex == 0])
    countMale = len(df[df.sex == 1])
    output += "\n\nPercentage of Female Patients: {:.2f}%".format((countFemale / len(df.sex)) * 100)
    output += "\n\nPercentage of Male Patients: {:.2f}%".format((countMale / len(df.sex)) * 100)

    df.groupby('target').mean()
    plt.savefig('target_mean.png')
    plt.close()

    Age = pd.crosstab(df.age, df.target)
    Age.head()
    Age.plot(kind="bar", figsize=(20, 6))
    plt.title('Heart Disease Frequency for Ages')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('heartDiseaseAndAges.png')
    plt.close()

    pd.crosstab(df.sex, df.target).plot(kind="bar", figsize=(15, 6), color=['#1CA53B', '#AA1111'])
    plt.title('Heart Disease Frequency for Sex')
    plt.xlabel('Sex (0 = Female, 1 = Male)')
    plt.xticks(rotation=0)
    plt.legend(["Without Disease", "With Disease"])
    plt.ylabel('Frequency')
    plt.savefig('DiseaseSex_crosstab.png')
    plt.close()

    plt.scatter(x=df.age[df.target == 1], y=df.thalach[(df.target == 1)], c="red")
    plt.scatter(x=df.age[df.target == 0], y=df.thalach[(df.target == 0)])
    plt.legend(["Disease", "No Disease"])
    plt.xlabel("Age")
    plt.ylabel("Maximum Heart Rate")
    plt.savefig('scatterplot.png')
    plt.close()

    pd.crosstab(df.slope, df.target).plot(kind="bar", figsize=(15, 6), color=['#DAF7A6', '#FF5733'])
    plt.title('Heart Disease Frequency for Slope')
    plt.xlabel('The Slope of The Peak Exercise ST Segment')
    plt.xticks(rotation=0)
    plt.ylabel('Frequency')
    plt.savefig('slope_crosstab.png')
    plt.close()

    pd.crosstab(df.fbs, df.target).plot(kind="bar", figsize=(15, 6), color=['#FFC300', '#581845'])
    plt.title('Heart Disease Frequency According To FBS')
    plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
    plt.xticks(rotation=0)
    plt.legend(["Without Disease", "With Disease"])
    plt.ylabel('Frequency of Disease or Not')
    plt.savefig('fbs_crosstab.png')
    plt.close()

    pd.crosstab(df.cp, df.target).plot(kind="bar", figsize=(15, 6), color=['#11A5AA', '#AA1190'])
    plt.title('Heart Disease Frequency According To Chest Pain Type')
    plt.xlabel('Chest Pain Type')
    plt.xticks(rotation=0)
    plt.ylabel('Frequency of Disease or Not')
    plt.savefig('chest_crosstab.png')
    plt.close()

    a = pd.get_dummies(df['cp'], prefix="cp")
    b = pd.get_dummies(df['thal'], prefix="thal")
    c = pd.get_dummies(df['slope'], prefix="slope")
    frames = [df, a, b, c]
    df = pd.concat(frames, axis=1)
    df.head()

    df = df.drop(columns=['cp', 'thal', 'slope'])
    df.head()

    y = df.target.values
    x_data = df.drop(['target'], axis=1)

    # Normalize
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Transpose matrices
    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    # Initialize
    def initialize(dimension):
        weight = np.full((dimension, 1), 0.01)
        bias = 0.0
        return weight, bias

    def sigmoid(z):
        y_head = 1 / (1 + np.exp(-z))
        return y_head

    def forwardBackward(weight, bias, x_train, y_train):
        # Forward
        y_head = sigmoid(np.dot(weight.T, x_train) + bias)
        loss = -(y_train * np.log(y_head) + (1 - y_train) * np.log(1 - y_head))
        cost = np.sum(loss) / x_train.shape[1]

        # Backward
        derivative_weight = np.dot(x_train, ((y_head - y_train).T)) / x_train.shape[1]
        derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
        gradients = {"Derivative Weight": derivative_weight, "Derivative Bias": derivative_bias}

        return cost, gradients

    def update(weight, bias, x_train, y_train, learningRate, iteration):
        costList = []
        index = []

        # For each iteration, update weight and bias values
        for i in range(iteration):
            cost, gradients = forwardBackward(weight, bias, x_train, y_train)
            weight = weight - learningRate * gradients["Derivative Weight"]
            bias = bias - learningRate * gradients["Derivative Bias"]

            costList.append(cost)
            index.append(i)

        parameters = {"weight": weight, "bias": bias}
        output = ""
        output += "iteration: {}".format(iteration)
        output += "cost: {}".format(cost)

        plt.plot(index,costList)
        plt.xlabel("Number of Iteration")
        plt.ylabel("Cost")
        plt.savefig('iteration.png')
        plt.close()
        return parameters, gradients, output

    def predict(weight, bias, x_test):
        z = np.dot(weight.T, x_test) + bias
        y_head = sigmoid(z)

        y_prediction = np.zeros((1, x_test.shape[1]))

        for i in range(y_head.shape[1]):
            if y_head[0, i] <= 0.5:
                y_prediction[0, i] = 0
            else:
                y_prediction[0, i] = 1

        return y_prediction

    def logistic_regression(x_train, y_train, x_test, y_test, learningRate, iteration):
      dimension = x_train.shape[0]
      weight, bias = initialize(dimension)
      output = ""
      parameters, gradients, output = update(weight, bias, x_train, y_train, learningRate, iteration)

      y_prediction = predict(parameters["weight"], parameters["bias"], x_test)

      accuracy = (100 - np.mean(np.abs(y_prediction - y_test)) * 100) / 100 * 100
      output += "\n\nManual Test Accuracy: {:.2f}%".format(accuracy)
      return output


    logistic_regression(x_train,y_train,x_test,y_test,1,100)
    lr = LogisticRegression()
    lr.fit(x_train.T, y_train.T)
    test_accuracy = lr.score(x_test.T, y_test.T) * 100
    output += "\n\nTest Accuracy: {:.2f}%".format(test_accuracy)

    # KNN Model
    knn = KNeighborsClassifier(n_neighbors=2)  # n_neighbors means k
    knn.fit(x_train.T, y_train.T)
    prediction = knn.predict(x_test.T)

    knn_accuracy = knn.score(x_test.T, y_test.T) * 100
    output += "\n\n2 NN Score: {:.2f}%".format(knn_accuracy)

        # Try to find the best k value
    scoreList = []
    for i in range(1, 20):
      knn2 = KNeighborsClassifier(n_neighbors=i)  # n_neighbors means k
      knn2.fit(x_train.T, y_train.T)
      scoreList.append(knn2.score(x_test.T, y_test.T))

    plt.plot(range(1, 20), scoreList)
    plt.xticks(np.arange(1, 20, 1))
    plt.xlabel("K value")
    plt.ylabel("Score")
    plt.savefig('knn.png')
    plt.close()

    max_score = max(scoreList)
    output += "\n\nMaximum KNN Score is {:.2f}%".format(max_score * 100)

    svm = SVC(random_state=1)
    svm.fit(x_train.T, y_train.T)
    svm_accuracy = svm.score(x_test.T, y_test.T) * 100
    output += "\n\nTest Accuracy of SVM Algorithm: {:.2f}%".format(svm_accuracy)

    nb = GaussianNB()
    nb.fit(x_train.T, y_train.T)
    nb_accuracy = nb.score(x_test.T, y_test.T) * 100
    output += "\n\nAccuracy of Naive Bayes: {:.2f}%".format(nb_accuracy)

    dtc = DecisionTreeClassifier()
    dtc.fit(x_train.T, y_train.T)
    dtc_accuracy = dtc.score(x_test.T, y_test.T) * 100
    output += "\n\nDecision Tree Test Accuracy: {:.2f}%".format(dtc_accuracy)

    rf = RandomForestClassifier(n_estimators=1000, random_state=1)
    rf.fit(x_train.T, y_train.T)
    rf_accuracy = rf.score(x_test.T, y_test.T) * 100
    output += "\n\nRandom Forest Algorithm Accuracy Score: {:.2f}%".format(rf_accuracy)

    methods = ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Decision Tree", "Random Forest"]
    accuracy = [test_accuracy, knn_accuracy, svm_accuracy, nb_accuracy, dtc_accuracy, rf_accuracy]
    colors = ["purple", "green", "orange", "magenta", "#CFC60E", "#0FBBAE"]

    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 5))
    plt.ylabel("Accuracy %")
    plt.xlabel("Algorithms")
    sns.barplot(x=methods, y=accuracy, palette=colors)
    plt.savefig('accuracy.png')
    plt.close()

    output = "Heart disease classification result: \n"  + output
    return output


    # Define the Telegram bot handlers
def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Welcome to the Heart Disease Classification Bot! PLease upload your CSV file to get result.")

def handle_message(update, context):
    # Get the file from the message
    file = context.bot.get_file(update.message.document.file_id)
    file_url = file.file_path

    # Download the file and get the local file path
    file_path = urllib.request.urlretrieve(file_url)[0]

    # Call the heart_disease_classification function and get the result
    result = heart_disease_classification(file_path)

    # Send the result back to the user
    context.bot.send_message(chat_id=update.effective_chat.id, text=result)

    #giving out the plots
    context.bot.send_message(chat_id=update.effective_chat.id, text="Count Plot")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('countplot.png', 'rb'))
    context.bot.send_message(chat_id=update.effective_chat.id, text="Sex_ratio Countplot")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('sex_countplot.png', 'rb'))
    context.bot.send_message(chat_id=update.effective_chat.id, text="Heart diesease as per age")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('heartDiseaseAndAges.png', 'rb'))
    context.bot.send_message(chat_id=update.effective_chat.id, text="HeartDisease as per Sex")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('DiseaseSex_crosstab.png', 'rb'))
    context.bot.send_message(chat_id=update.effective_chat.id, text="Scatter Plot")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('scatterplot.png', 'rb'))
    context.bot.send_message(chat_id=update.effective_chat.id, text="Slope crosstab Plot")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('slope_crosstab.png', 'rb'))
    context.bot.send_message(chat_id=update.effective_chat.id, text="Fasting Blood Sugar Crosstab Plot")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('fbs_crosstab.png', 'rb'))
    context.bot.send_message(chat_id=update.effective_chat.id, text="Chest Pain Plot")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('chest_crosstab.png', 'rb'))
    context.bot.send_message(chat_id=update.effective_chat.id, text="Iteration Cost Plot")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('iteration.png', 'rb'))
    context.bot.send_message(chat_id=update.effective_chat.id, text="KNN Plot")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('knn.png', 'rb'))
    context.bot.send_message(chat_id=update.effective_chat.id, text="Accuracy comparision :")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('accuracy.png', 'rb'))

# Create an updater and pass your bot's API token
updater = Updater(token="<your bot token>", use_context=True)
dispatcher = updater.dispatcher

# Register the handlers
start_handler = CommandHandler('start', start)
message_handler = MessageHandler(Filters.document, handle_message)
dispatcher.add_handler(start_handler)
dispatcher.add_handler(message_handler)

# Start the bot
updater.start_polling()

updater.idle()
