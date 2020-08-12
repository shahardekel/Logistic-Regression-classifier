# Logistic-Regression-classifier
use Logistic Regression classifier in order to generate a multi-class classifier

we will use the sklearn's Iris dataset from which contains 150 observations of 3 Iris species. You will use Logistic Regression classifier in order to generate a multi-class classifier of type one-versus-rest.

a. Load the data using the following code:
  from sklearn import datasets
  iris = datasets.load_iris()
  X = iris.data
  y = iris.target

X is a np-array with 150 rows and 4 columns: Sepal Length, Sepal Width, Petal Length and Petal Width.
y is a np-array with 150 rows and 1 column which contains 0, 1, or 2 for each
Iris species: Setosa, Versicolour and Virginicacv (respectively).

b. Split the data to train and test data. Use the train_test_split function with random_state=1000.

c. A Logistic Regression classifier is a binary classifier, which also provides us a confidence level – the probability of belonging to each class of the two. 

In the following sections, you will build a one-versus-rest classifier using the
following scheme:
  o Create a new vector from the response vector (y_train) which contains 1 for the Setosa and -1 for the other species. Do the same for    y_test.
  o Build a Logistic Regression classifier for the species Setosa (using the training data). Use the new response vector you have created.

d. Repeat section c for the other two species.

e. Create a function which receives all the above classifiers and a collection of observations as np-array and returns a one-versus-rest classification vector, in which each observation is classified to the class for which it has the maximal probability to belong to.

f. Use the function you wrote to perform a one-versus-rest classification for the test data. Use the output of this function to create and plot a confusion matrix.
You are to display two different plots – one for the unnormalized (original) confusion matrix and one for the normalized matrix.

g. Choose one observation which is misclassified and explain why, in your opinion, did the one-versus-rest classifier misclassify it.
