from seaborn import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
iris=load_dataset('iris')# загружаем датасет с ирисами
X_iris=iris.drop('species',axis=1)# создаем матрицу признаков
y_iris=iris['species']# и вектор значений
X_train, X_test, y_train, y_test=train_test_split(X_iris,y_iris,random_state=1)# сплитим на тренеровачный набор и тестовый
model=GaussianNB()# создаем экземпляр модели наивного байеса
model.fit(X_train,y_train)# обучаем модель
y_predict=model.predict(X_test)# предсказываем
print(accuracy_score(y_predict,y_test)) # точность модели
