# Your solution goes here
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("p1/train.csv")
dftest = pd.read_csv("p1/test.csv")
N = len(df) #This and the next 2 lines calculate the inverted matrix in the normal equation
sumxi = df["age"].sum()
Xinv = np.linalg.inv(np.array([[N,sumxi],[sumxi,(df["age"]**2).sum()]]))
sumyi = df["charges"].sum()
sumxyi = (df["charges"]*df["age"]).sum()
Y = np.array([[sumyi],[sumxyi]]) #This and the previous 2 lines helps find the second matrix in the normal equation with known values
res = np.dot(Xinv,Y) #Multipllying these 2 matrices allows us to find the slope and y intercept
b = res[0][0]
m = res[1][0]
Ntest = len(dftest)
SSE = 0
for i in range(1,Ntest+1): #Calculate sum of squared errors
    SSE+=(dftest["charges"].iloc[i-1]-(m*dftest["age"].iloc[i-1])-b)**2
print(f"Sum of squared errors: {SSE:.2f}")

dfplot = pd.DataFrame({"pred":(m*dftest["age"]+b), "actual": dftest["charges"]}) #Create new dataframe with one column for the prediced charge based on our earlier code and one for the actual charge
scat = dfplot.plot.scatter(title="Health Insurance Charges Predicted vs. Actual",x="pred",y="actual") #Plot testing data onto scatter plot
scat.set_xticks([0,10000,20000,30000,40000,50000,60000])
plt.savefig("p1/problem1.png")

