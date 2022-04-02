########################
#    open your prompt  #
# and run this line :  #
########################
# streamlit run Teststreamlit.py #
########################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf # https://pypi.org/project/yfinance/

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from math import sqrt
from sklearn import linear_model


###########
# sidebar #
###########
ticker = ('AAPL', 'MSFT','SPY','WMT')
option = (50,100,200)
n = st.sidebar.selectbox('Combien de simulation ?', option)


today = datetime.date.today()
before = today - datetime.timedelta(days=1200)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Erreur : La date de fin doit tomber après la date de début.')
    
    
    
##############
# Stock data #
##############

# Download data
df = yf.download(ticker , start= start_date,end= end_date, progress=False)['Adj Close']
df = df.dropna()

df['portefeuille'] = df.sum(axis=1)
forecast_out = n # les jours à prédire
df['Prediction'] = df[["portefeuille"]].shift(-forecast_out)

X = np.array(df.drop(['Prediction'],1))
X = X[:-forecast_out]
y = np.array(df["Prediction"])
y = y[:-forecast_out]

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(x_train,y_train)
ridge_prediction = ridge_reg.predict(x_forecast)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(x_train,y_train)
ridge_prediction = ridge_reg.predict(x_forecast)

ridge_prediction = [float(np.round(x,2)) for x in ridge_prediction]
ridge = pd.DataFrame(ridge_prediction,columns=['ridge_prediction'])
somme2 = pd.DataFrame(df["portefeuille"])
dataframeridge = somme2.append(ridge ,ignore_index=True)
col_listridge = list(dataframeridge)
new_columridge = dataframeridge[col_listridge].sum(axis=1)
pd1 = pd.DataFrame(new_columridge, columns=["Ridge"])
new_pred = pd.concat([pd1], axis=1)
df = df.reset_index()
ridge['date'] = pd.date_range(df['Date'].max()+pd.Timedelta(1,unit='d'),periods=len(ridge))

date = df['Date'].append(ridge['date'])
date = pd.DataFrame(date)
date = date.reset_index()
del date["index"]

new_pred = pd.concat([pd1, date], axis=1) 
new_pred = new_pred.rename(columns={0: "date"})


#del new_pred["date"]
del df["Prediction"]
del df['Date']

df = pd.concat([df,new_pred], ignore_index=False, axis=1)
df = df.set_index(df['date'])
del df['date']
###################
# Set up main app #
###################

# Plot the prices and the bolinger bands
#for i in df.columns: 
#    st.write(df[i].name)
#    st.line_chart(df[i])
st.write('Prédiction sur le portefeuille') 
st.line_chart(df)   
progress_bar = st.progress(0)

st.write('Notre portefeuille') 
st.write(df.columns.difference(['Ridge']).name) 
st.area_chart(df[df.columns.difference(['Ridge'])])
    


st.write('Données récentes ')
st.dataframe(df.head(20))


### html / css


primaryColor="#d33682"
backgroundColor="#002b36"
secondaryBackgroundColor="#586e75"
textColor="#fafafa"
font="sans serif"