def grafica_polinomica(X,Y, grado):
  from sklearn.linear_model import LinearRegression
  from sklearn.metrics import r2_score, mean_squared_error
  from sklearn.preprocessing import PolynomialFeatures
  import numpy as np
  import matplotlib.pyplot as plt

  X=X[:,np.newaxis]
  Y=Y[:,np.newaxis]

  # Modificando los valores de X para considerar monomios con potencia
  polinomio=PolynomialFeatures(grado)
  x_poly=polinomio.fit_transform(X)

  model=LinearRegression()
  model.fit(x_poly,Y)
  y_pred_poly=model.predict(x_poly)

  zip_sorted=sorted(zip(X,y_pred_poly)) # Zip object sorting
  X,y_pred_poly=zip(*zip_sorted) # Tuple unpacking
  grafica= plt.plot(X,y_pred_poly, 
                  color=(np.random.rand(),np.random.rand(),np.random.rand(),1))
  r2=r2_score(Y,y_pred_poly)
  MSE=mean_squared_error(Y,y_pred_poly)
  RMSE=np.sqrt(MSE)
  return grafica,r2,MSE,RMSE