from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

#A
iris = datasets.load_iris()
X = iris.data
y = iris.target

"""
X is a np-array with 150 rows and 4 columns: Sepal Length, Sepal Width, Petal
Length and Petal Width.
y is a np-array with 150 rows and 1 column which contains 0, 1, or 2 for each
Iris species: Setosa, Versicolour and Virginicacv (respectively).
"""

#B
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1000)

#C
def confidence_level(n):
    y_train_new=np.where(y_train==n,1,-1)
    y_test_new=np.where(y_test==n, 1,-1)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train_new)
    y_pred_prob = logreg.predict_proba(X_test)[:,1]
    return y_pred_prob

Setosa=confidence_level(0)


#D
Versicolour=confidence_level(1)
Virginicacv=confidence_level(2)

#E
def  one_versus_rest_classification(obs1,obs2,obs3):
    result=[]
    obs=zip(obs1,obs2,obs3)
    for i,j,k in obs:
        result.append(np.argmax([i,j,k]))
    return result

#F
classification=one_versus_rest_classification(Setosa, Versicolour, Virginicacv)
cm = confusion_matrix(y_test, classification)
sn.heatmap(cm, annot=True, cmap="tab20b")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('confusion matrix')
plt.show()

cm = cm.astype('float')
normalized_cm = cm / cm.sum(axis=1)[:, np.newaxis]
sn.heatmap(normalized_cm, annot=True, cmap="tab20b")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('confusion matrix- normalized')
plt.show()

#G
"""
We can see that observation 10 was classified as a mistake.
The observation was originally virginicacv (2) and with the classifier,
was classified as versicolour (1).
The values of observation 10: X = [5.4 3.7 1.5 0.2]

If we look at the values of the different observations in the database,
they can be characterized so that most observations are labeled as 2,
the second column has values less than 3, the third column has values less than 6,
and the fourth has values less than 2.
It can be seen that observation 10 has very borderline values (at the upper limit) for each of the above features,
and is very close to the values that characterize the type 1 flowers as well,
and therefore probably very close to the "dividing line" which divides the observations into types 1 and 2.
From the above explanation it can be concluded that 
it makes sense that the classifier was wrong in classifying observation.
"""





