import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

iris=sns.load_dataset('iris')# загружаем датасет с ирисами
X_iris=iris.drop('species',axis=1)# создаем матрицу признаков
y_iris=iris['species']# и вектор значений
model=PCA(n_components=2) # создае экземпляр модели с 2 гиперпараметрами
							# т.е проецируем наши данные в 2d пространство
model.fit(X_iris) # обучаем
X_2d=model.transform(X_iris) #преобразуем данные в двумерные
iris['PCA1']=X_2d[:,0] # значения певого пространства
iris['PCA2']=X_2d[:,1] # значения второго пространства
#sns.lmplot('PCA1','PCA2', hue='species',data=iris,fit_reg=False)
#plot.show()


model_gmm=GaussianMixture (n_components=3,covariance_type ='full')
model_gmm.fit(X_iris)
y_predict=model_gmm.predict(X_iris)
iris['cluster']=y_predict
sns.lmplot("PCA1","PCA2",data=iris,hue='species',col='cluster',fit_reg=False)
plot.show()
