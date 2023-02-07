#Import Modules & Packages
import pandas as pd
import numpy as np
from sklearn import linear_model
import requests
from nba_api.stats import endpoints
from matplotlib import pyplot as plt
from IPython.display import display

# Gather data into csv format
data = endpoints.leagueleaders.LeagueLeaders()
df = data.league_leaders.get_data_frame()
#(df.head())
df.to_csv('nba_stats.csv')

# Variables
x = df.FGA/df.GP
y = df.PTS/df.GP

x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)


model = linear_model.LinearRegression()
model.fit(x,y)

r2 = round(model.score(x,y),2)
predicted_y = model.predict(x)

plt.scatter(x,y,s=15,alpha = 0.5,)
plt.plot(x,predicted_y,color = 'black')
plt.title('NBA - Relationship Between FGA and PPG')
plt.xlabel(' Avg Field Goal %')
plt.ylabel('Avg Points Per Game')
plt.text(10,25, f'R2 = {r2}')

for i in range(0,2):
    plt.annotate(df.PLAYER[i],
    (x[i],y[i]),
    (x[i]+3, y[i]-2), 
    arrowprops = dict(arrowstyle = '-'))

