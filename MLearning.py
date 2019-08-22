#Basic imports
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
#sklearn imports
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn.metrics import silhouette_score


#plotly imports
#import matplotlib.pyplot as plt
import plotly
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


#df is our original DataFrame
df = pd.read_csv("CC GENERAL.csv")

Data=df.copy()

# Temp is dataframe that contain columns name and number of their empty rows
Temp=pd.DataFrame(Data.isnull().sum())
Temp.columns=["Sum"]

# to drop the empty values columns and CUSTID
Data=pd.DataFrame(Data.drop(Temp.index[Temp['Sum']>0], axis=1))
Data=pd.DataFrame(Data.drop(['CUST_ID'],axis=1))

# check if there is any columns with empty values and check the list of remaining columns
print('Columns that has empty value' + str(Temp.index[Temp['Sum']>0]))
print(list(Data.columns))

# to Scale our Data
scaler = StandardScaler()
Data=pd.DataFrame(scaler.fit_transform(Data))

print(Data.head())

# # Use silhouette score to determine K
# range_n_clusters = list (range(2,15))
# print ("Number of clusters from 2 to 9: \n", range_n_clusters)
#
# maximum=0
#
# for n_clusters in range_n_clusters:
#     clusterer = KMeans (n_clusters=n_clusters)
#     preds = clusterer.fit_predict(Data)
#     centers = clusterer.cluster_centers_
#
#     score = silhouette_score (Data, preds, metric='euclidean')
#     if(score > maximum):
#         n_max=n_clusters
#         maximum=score
#     print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
#
# print(n_max)


# Elbow Method to decide value of K
# Sum_of_squared_distances = []
# K = range(1,15)
# for k in K:
#     km = KMeans(n_clusters=k)
#     km = km.fit(Data)
#     Sum_of_squared_distances.append(km.inertia_)
#
# fig = go.Figure(data=go.Scatter(x=K, y=Sum_of_squared_distances))
# fig.show()


# # k means determine k
# distortions = []
# K = range(1,10)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k).fit(Data)
#     kmeanModel.fit(Data)
#     distortions.append(sum(np.min(cdist(Data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / Data.shape[0])
#
#
# fig = go.Figure(data=go.Scatter(x=K, y=distortions))
# fig.show()
#


# Initialise K-means model
kmeans=KMeans(n_clusters=8)

#Fit our model
kmeans.fit(Data)

centroids = kmeans.cluster_centers_
print(centroids)

#Find which cluster each data-point belongs to
clusters = kmeans.predict(Data)

#Add the cluster vector to our DataFrame, X
Data["Cluster"] = clusters

print(Data.head())

#plotX is a DataFrame containing 70000 values
plotX = pd.DataFrame(np.array(Data))

#Rename plotX's columns since it was briefly converted to an np.array above
plotX.columns = Data.columns

print(plotX.head())

#PCA with one principal component
pca_1d = PCA(n_components=1)

#PCA with two principal components
pca_2d = PCA(n_components=2)

#PCA with three principal components
pca_3d = PCA(n_components=3)

#This DataFrame holds that single principal component mentioned above
PCs_1d = pd.DataFrame(pca_1d.fit_transform(plotX.drop(["Cluster"], axis=1)))

#This DataFrame contains the two principal components that will be used
#for the 2-D visualization mentioned above
PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))

#And this DataFrame contains three principal components that will aid us
#in visualizing our clusters in 3-D
PCs_3d = pd.DataFrame(pca_3d.fit_transform(plotX.drop(["Cluster"], axis=1)))

print(PCs_3d)
centroids=pd.DataFrame(pca_3d.transform(centroids))
print(centroids)

PCs_1d.columns = ["PC1_1d"]

#"PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
#And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
PCs_2d.columns = ["PC1_2d", "PC2_2d"]

PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]

plotX = pd.concat([plotX,PCs_1d,PCs_2d,PCs_3d], axis=1, join='inner')

plotX["dummy"] = 0

#Note that all of the DataFrames below are sub-DataFrames of 'plotX'.
#This is because we intend to plot the values contained within each of these DataFrames.


for i in range(0,9):
    globals()['cluster%s' % i]=plotX[plotX["Cluster"] == i]


# cluster0 = plotX[plotX["Cluster"] == 0]
# cluster1 = plotX[plotX["Cluster"] == 1]
# cluster2 = plotX[plotX["Cluster"] == 2]

# print(cluster0)
# print(cluster1)
# print(cluster2)


##//////////////////////////////////////////////// ##
##//////////       PCA 1 Dimension       ///////// ##
##//////////////////////////////////////////////// ##
#
# #trace1 is for 'Cluster 0'
# trace1 = go.Scatter(
#                     x = cluster0["PC1_1d"],
#                     y = cluster0["dummy"],
#                     mode = "markers",
#                     name = "Cluster 0",
#                     marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
#                     text = None)
#
# #trace2 is for 'Cluster 1'
# trace2 = go.Scatter(
#                     x = cluster1["PC1_1d"],
#                     y = cluster1["dummy"],
#                     mode = "markers",
#                     name = "Cluster 1",
#                     marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
#                     text = None)
#
# #trace3 is for 'Cluster 2'
# trace3 = go.Scatter(
#                     x = cluster2["PC1_1d"],
#                     y = cluster2["dummy"],
#                     mode = "markers",
#                     name = "Cluster 2",
#                     marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
#                     text = None)
#
# data = [trace1, trace2, trace3]
#
# title = "Visualizing Clusters in One Dimension Using PCA"
#
# layout = dict(title = title,
#               xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
#               yaxis= dict(title= '',ticklen= 5,zeroline= False)
#              )
#
# fig = dict(data = data, layout = layout)
#
# plotly.offline.plot(fig)

##//////////////////////////////////////////////// ##
##//////////       PCA 2 Dimension       ///////// ##
##//////////////////////////////////////////////// ##

# #trace1 is for 'Cluster 0'
# trace1 = go.Scatter(
#                     x = cluster0["PC1_2d"],
#                     y = cluster0["PC2_2d"],
#                     mode = "markers",
#                     name = "Cluster 0",
#                     marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
#                     text = None)
#
# #trace2 is for 'Cluster 1'
# trace2 = go.Scatter(
#                     x = cluster1["PC1_2d"],
#                     y = cluster1["PC2_2d"],
#                     mode = "markers",
#                     name = "Cluster 1",
#                     marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
#                     text = None)
#
# #trace3 is for 'Cluster 2'
# trace3 = go.Scatter(
#                     x = cluster2["PC1_2d"],
#                     y = cluster2["PC2_2d"],
#                     mode = "markers",
#                     name = "Cluster 2",
#                     marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
#                     text = None)
#
# data = [trace1, trace2, trace3]
#
# title = "Visualizing Clusters in Two Dimensions Using PCA"
#
# layout = dict(title = title,
#               xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
#               yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
#              )
#
# fig = dict(data = data, layout = layout)
#
# plotly.offline.plot(fig)

##//////////////////////////////////////////////// ##
##//////////       PCA 3 Dimension       ///////// ##
##//////////////////////////////////////////////// ##

#Instructions for building the 3-D plot

# for n in range(0,3):
#    trace= go.Scatter3d(
#                             x =["PC1_3d"],
#                             y = "PC2_3d"],
#                             z = ["PC3_3d"],
#                             mode = "markers",
#                             name = "Cluster 0",
#                             marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
#                             text = None)
#     )



# trace1 is for 'Cluster 0'
trace1 = go.Scatter3d(
                    x = cluster0["PC1_3d"],
                    y = cluster0["PC2_3d"],
                    z = cluster0["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

# trace2 is for 'Cluster 1'
trace2 = go.Scatter3d(
                    x = cluster1["PC1_3d"],
                    y = cluster1["PC2_3d"],
                    z = cluster1["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

# trace3 is for 'Cluster 2'
trace3 = go.Scatter3d(
                    x = cluster2["PC1_3d"],
                    y = cluster2["PC2_3d"],
                    z = cluster2["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'rgba(70, 100, 150, 0.8)'),
                    text = None)

# trace4 is for 'Cluster 3'
trace4 = go.Scatter3d(
                    x = cluster3["PC1_3d"],
                    y = cluster3["PC2_3d"],
                    z = cluster3["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 3",
                    marker = dict(color = 'rgba(100, 255, 200, 0.8)'),
                    text = None)

# trace5 is for 'Cluster 4'
trace5 = go.Scatter3d(
                    x = cluster4["PC1_3d"],
                    y = cluster4["PC2_3d"],
                    z = cluster4["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 4",
                    marker = dict(color = 'rgba(160, 200, 150, 0.8)'),
                    text = None)

# trace6 is for 'Cluster 5'
trace6 = go.Scatter3d(
                    x = cluster5["PC1_3d"],
                    y = cluster5["PC2_3d"],
                    z = cluster5["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 5",
                    marker = dict(color = 'rgba(50, 170, 60, 0.8)'),
                    text = None)

# trace7 is for 'Cluster 6'
trace7 = go.Scatter3d(
                    x = cluster6["PC1_3d"],
                    y = cluster6["PC2_3d"],
                    z = cluster6["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 6",
                    marker = dict(color = 'rgba(0, 0, 200, 0.8)'),
                    text = None)

# trace8 is for 'Cluster 7'
trace8 = go.Scatter3d(
                    x = cluster7["PC1_3d"],
                    y = cluster7["PC2_3d"],
                    z = cluster7["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 7",
                    marker = dict(color = 'rgba(100, 32, 20, 0.8)'),
                    text = None)


# trace9 is for 'Centroid'
trace9 = go.Scatter3d(
                    x = centroids.iloc[:,0],
                    y = centroids.iloc[:,1],
                    z = centroids.iloc[:,2],
                    mode = "markers+text",
                    name = "Centroids",
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),
                    text = ['Cluster 0','Cluster 1','Cluster 2','Cluster 3',
                            'Cluster 4','Cluster 5','Cluster 6','Cluster 7'])


data = [trace1, trace2, trace3, trace4, trace5,
        trace6, trace7, trace8, trace9 ]

title = "Visualizing Clusters in Three Dimensions Using PCA"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

plotly.offline.plot(fig)





