import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


datos = pd.read_excel(r'C:\Users\Jafet\Documents\clase\proyecto\data_mexico.xlsx')

datos.head(5)

datos['fecha'] = pd.to_datetime(datos['Date_reported'])
datos['year'] = datos['fecha'].dt.year

dataframe2020 = datos.loc[(datos['year'] == 2020)]
dataframe2021 = datos.loc[(datos['year'] == 2021)]

#2021 Y 2020
X_predict = np.c_[datos['day']]
y_predict = np.c_[datos['Cumulative_cases']]

#2020
X_2020 = np.c_[dataframe2020['day']]
y_2020 = np.c_[dataframe2020['Cumulative_cases']]

#2021
X_2021 = np.c_[dataframe2021['day']]
y_2021 = np.c_[dataframe2021['Cumulative_cases']]


#Grafica 1

fig, ax = plt.subplots()
ax.plot(X_2020, y_2020)

ax.set_xlabel('day', fontsize=15)
ax.set_ylabel('Cumulative_cases', fontsize=15)
ax.set_title('Mexico 2020')
ax.grid(True)
ax.legend('Mexico 2020')
fig.tight_layout()
plt.show()


modelo_polynomial = LinearRegression()
poly_features = PolynomialFeatures(degree=4)

X_poly = poly_features.fit_transform(X_2020)

#Filtro por dia
df = dataframe2021.loc[(datos['day'] == 700)]

y_test = np.c_[df['Cumulative_cases']]
X_test = np.c_[df['day']]

X_poly_predict = poly_features.fit_transform(X_test)
X_poly_score = poly_features.fit_transform(X_2021)
X_poly_all = poly_features.fit_transform(X_predict)

#Entreno el modelo
modelo_polynomial.fit(X_poly,y_2020)

#Saco el accuracy
print("Score ",modelo_polynomial.score(X_poly_score,y_2021))

#Imprimo el vlaora actual del dia
print("Valor del dia actual ",modelo_polynomial.predict(X_poly_predict))


#Grafica 2

fig, ax = plt.subplots()
ax.plot(X_predict, y_predict)

ax.set_xlabel('day', fontsize=15)
ax.set_ylabel('Cumulative_cases', fontsize=15)
plt.plot(X_predict, modelo_polynomial.predict(X_poly_all),color='red')
ax.set_title('Mexico 2020 y 2021')
ax.legend(['R','P'])

ax.grid(True)
fig.tight_layout()
plt.show()