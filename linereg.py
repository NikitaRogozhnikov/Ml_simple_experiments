import matplotlib.pyplot as plot
import numpy as np
from sklearn.linear_model import LinearRegression
'''Простой пример линейной регрессии'''
rng=np.random.RandomState(1)# создаем экземпляр генератора случайных чисел
x=10*rng.rand(50)# массив случмйных чисел
y=2*x-1+rng.randn(50)# массив случайных чисель
X=x[:,np.newaxis]#преобразуем х в вектор стобец
model=LinearRegression(fit_intercept=True)# создаем экземпляр модели линейной регрессии
model.fit(X,y)# обучаем модель
xfit=np.linspace(-1,10)#массив распределенных чисел по которым будет производится предсказание
Xfit=xfit[:,np.newaxis]# преобразованный массив стобец
yfit=model.predict(Xfit)# массив предсказанных значений
plot.scatter(x,y) # график рассеяния
plot.plot(xfit,yfit)# график предсканной велечины
plot.grid()
plot.show()
