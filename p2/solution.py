# Your solution goes here
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("Telco-Customer-Churn-Ready.csv")

#Pre-processing: Make values numeric
df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
df["Partner"] = df["Partner"].map({"No": 0, "Yes": 1})
df["Dependents"] = df["Dependents"].map({"No": 0, "Yes": 1})
df["PhoneService"] = df["PhoneService"].map({"No": 0, "Yes": 1})
df["MultipleLines"] = df["MultipleLines"].map({"No": 0, "Yes": 1, "No phone service": 2})
df["InternetService"] = df["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2})
df["OnlineSecurity"] = df["OnlineSecurity"].map({"No": 0, "Yes": 1, "No internet service": 2})
df["OnlineBackup"] = df["OnlineBackup"].map({"No": 0, "Yes": 1, "No internet service": 2})
df["DeviceProtection"] = df["DeviceProtection"].map({"No": 0, "Yes": 1, "No internet service": 2})
df["TechSupport"] = df["TechSupport"].map({"No": 0, "Yes": 1, "No internet service": 2})
df["StreamingTV"] = df["StreamingTV"].map({"No": 0, "Yes": 1, "No internet service": 2})
df["StreamingMovies"] = df["StreamingMovies"].map({"No": 0, "Yes": 1, "No internet service": 2})
df["Contract"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two Year": 2})
df["PaperlessBilling"] = df["PaperlessBilling"].map({"No": 0, "Yes": 1})
df["PaymentMethod"] = df["PaymentMethod"].map({"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card": 3})
df["Contract"] = df["Contract"].fillna(df["Contract"].mode()[0])
df["PaymentMethod"] = df["PaymentMethod"].fillna(df["PaymentMethod"].mode()[0])
df["MonthlyCharges"] = (df["MonthlyCharges"]-df["MonthlyCharges"].mean())/df["MonthlyCharges"].std()

kvals = [5,25,51,75] #Define important components such as the k values and the data frame that we will be plotting
max_prec = -1
best_k = -1
dfplot = pd.DataFrame({"prec": [-1,-1,-1,-1], "k": kvals})
index = 0
crosserr = np.array_split(df.sample(frac=1),10)
besttp = -1
bestfp = -1

#Find average precision for each k using cross validation
for k in kvals:
    prec = pd.Series([0,0,0,0,0,0,0,0,0,0])
    for i in range(1,11):
        trainset = pd.DataFrame()
        for j in range(1,11):
            if j!=i:
                trainset = pd.concat([trainset,crosserr[j-1]])
        X = trainset.drop(columns=["Churn"])
        Y = trainset["Churn"]
        Xtest = crosserr[i-1].drop(columns=["Churn"])
        Ytest = crosserr[i-1]["Churn"]
        knnc = KNeighborsClassifier(k).fit(X,Y)
        predictions = knnc.predict(Xtest)
        tp = ((predictions == 1) & (Ytest == 1)).sum()
        fp = ((predictions == 1) & (Ytest == 0)).sum()
        if tp + fp!=0:
            prec[i-1] = tp/(tp + fp)
    precavg = sum(prec) / len(prec)
    dfplot.loc[index,"prec"] = precavg
    if precavg > max_prec:
        max_prec = precavg
        best_k = k
    index+=1

print(f"Best precision: {max_prec:.2f} when k: {best_k}")

plt.figure(1)
plt.plot(dfplot["k"],dfplot["prec"]) #Plot line chart
plt.savefig("problem2.png")

plt.figure(2) #Plot ROC curve
plt.plot()
plt.savefig("roc.png")