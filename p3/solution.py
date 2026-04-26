# Your solution goes here
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("p3/Mall_Customers.csv")

#Pre-processing including dropping irrelevant values and discretization
df = df.drop(columns="CustomerID")
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

#Define import values such as list of number of clusters and array for training
kvals = [2,3,5]
X = np.array(df)
bestSE = 99999999999
SSElist = [-1,-1,-1]
best_k = -1
i=0

#Train a new model for each number of clusters
for k in kvals:
    kmc = KMeans(k,random_state=0).fit(X)
    SSE = kmc.inertia_ #Calculate SSE for the current cluster and compare to the lowest SSE found thus far
    SSElist[i]=SSE
    if SSE < bestSE: 
        bestSE = SSE
        best_k = k
    i+=1
print(f"Best k for SSE: {best_k} with SSE: {bestSE:.2f}")

#Plot line chart
plt.figure(1)
plt.plot(kvals,SSElist)
plt.title("Number of Clusters vs. SSE")
plt.xlabel("# Clusters")
plt.ylabel("SSE")
plt.savefig("p3/problem3.png")

#Plot scatter plot
plt.figure(2)
pca = PCA(2)
X = pd.DataFrame(X)
new_X = pca.fit_transform(X)
plt.scatter(new_X[:,0],new_X[:,1],c=pd.Series(kmc.labels_).map({0: "blue", 1: "red", 2: "green", 3: "yellow", 4: "orange"}))
plt.title("Scatter Plot of Data After Principle Component Analysis")
plt.xlabel("PC 1")
plt.xlabel("PC 2")
plt.savefig("p3/clusters.png")

