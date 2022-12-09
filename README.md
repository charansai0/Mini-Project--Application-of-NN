# Mini-Project--Application-of-NN


(Expected the following details in the report )
## Project Title:
Rainfall Prediction.
## Project Description 
Rainfall Prediction is the application of science and technology to predict the amount of rainfall over a region. It is important to exactly determine the rainfall for effective use of water resources, crop productivity and pre-planning of water structures.
## Algorithm:
1.Import necessary libraries.

2.Apply the rainfall dataset to algoritm.

3.Read the dataset.

4.Plot the graph and correlation matrix.

5.Study the final output.
## Program:
~~~

Developed By Team Members:
1.A.Tharun
2.S.Sameer.
3.V.Charan sai.
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] 
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') 
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number])
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] 
    columnNames = list(df)
    if len(columnNames) > 10: 
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
nRowsRead = 1000
df2 = pd.read_csv('rainfall.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'rainfall in india 1901-2015.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 8)
plotScatterMatrix(df2, 20, 10)
~~~
## Output:
![205808275-90d8a736-5b79-46cc-be08-c849e1e6441a](https://user-images.githubusercontent.com/94296221/206646292-3373d6ac-908c-4314-bb4d-88d918bfd073.jpg)
![205808304-107e1e5e-810b-4e65-820e-a009ef1ccfb2](https://user-images.githubusercontent.com/94296221/206646310-4311853c-82f6-4143-b66b-e0a856d96513.jpg)
![205808324-2fdf098a-6344-45fc-a3e1-56c445578858](https://user-images.githubusercontent.com/94296221/206646326-46647de5-e030-4851-8a4f-6c06ba6db175.jpg)
![205808346-d951dc4e-4169-41a0-86cc-716bad0a5e98](https://user-images.githubusercontent.com/94296221/206646337-30e404c1-c6d6-4276-b604-0d3d958b9af2.jpg)
![205808352-84a049a7-fb93-4d06-adb6-b1d3d4cdbbb4](https://user-images.githubusercontent.com/94296221/206646350-76e8d33b-5767-4732-a09e-1b58a03d0daa.jpg)

## Advantage :
Rainfall prediction is important as heavy rainfall can lead to many disasters. The prediction helps people to take preventive measures and moreover the prediction should be accurate. There are two types of prediction short term rainfall prediction and long term rainfall.
## Result:
