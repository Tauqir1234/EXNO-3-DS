## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
import pandas as pd

df=pd.read_csv("/content/Encoding Data.csv")

df

![Screenshot 2025-04-15 103224](https://github.com/user-attachments/assets/66e6bb22-473a-47ca-a50d-30a5ccb5bbe8)

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

![Screenshot 2025-04-15 103424](https://github.com/user-attachments/assets/e235e21d-97cb-4d34-8ceb-0e05421ab5a5)

df['bo2']=e1.fit_transform(df[["ord_2"]])

df

![Screenshot 2025-04-15 103504](https://github.com/user-attachments/assets/0c895f9c-2148-476e-b879-740e786aac53)

le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc

![Screenshot 2025-04-15 103618](https://github.com/user-attachments/assets/e1e7b463-2ab8-4662-b6ca-cd985c18310a)

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse=False)

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

![Screenshot 2025-04-15 103717](https://github.com/user-attachments/assets/1a866233-c7c0-47d2-8f58-c9288bf8970b)

df2=pd.concat([df2,enc],axis=1)

df2 

![Screenshot 2025-04-15 103802](https://github.com/user-attachments/assets/540d27ef-f7a6-4784-8aeb-6ad5a574baa7)

pd.get_dummies(df2,columns=["nom_0"])

![Screenshot 2025-04-15 103840](https://github.com/user-attachments/assets/da6da528-d51c-4997-b244-a15fcddd7e80)

pip install --upgrade category_encoders

![Screenshot 2025-04-15 103916](https://github.com/user-attachments/assets/229efd40-22a3-4db9-95fb-2800ff5ff85b)

from category_encoders import BinaryEncoder

df=pd.read_csv("/content/data.csv")

df

![Screenshot 2025-04-15 104032](https://github.com/user-attachments/assets/3bcd71df-a8df-4396-b1b6-7d574d5429a0)

be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

dfb=pd.concat([df,nd],axis=1)

dfb1=df.copy()

dfb 

![Screenshot 2025-04-15 104113](https://github.com/user-attachments/assets/6f1af2cc-26d3-4ee9-aca4-4fbe1e166522)

from category_encoders import TargetEncoder

te=TargetEncoder()

CC=df.copy()

new=te.fit_transform(X=CC["City"],y=CC["Target"])

CC=pd.concat([CC,new],axis=1)

CC

![Screenshot 2025-04-15 104148](https://github.com/user-attachments/assets/7ced45b5-08b8-4bf0-85da-ed1ce2ebace7)

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("/content/Data_to_Transform.csv")

df

![Screenshot 2025-04-15 104231](https://github.com/user-attachments/assets/6028cb49-7aea-449d-b367-4148ddae267f)

df.skew() 

![Screenshot 2025-04-15 104332](https://github.com/user-attachments/assets/4e5d4ff7-0ff9-4afe-9dc2-64dedbcb502d)

np.log(df["Highly Positive Skew"])

![Screenshot 2025-04-15 104358](https://github.com/user-attachments/assets/9b71bd4c-c4b1-4d4e-b40b-908bb3479cdb)

np.reciprocal(df["Moderate Positive Skew"])

![Screenshot 2025-04-15 104431](https://github.com/user-attachments/assets/c1e6ce56-b13e-4a19-bc14-b73d25b245f3)

np.sqrt(df["Highly Positive Skew"])

![Screenshot 2025-04-15 104500](https://github.com/user-attachments/assets/0f8f16dc-1b47-4147-8b56-6d58ebc91641)

np.square(df["Highly Positive Skew"])

![Screenshot 2025-04-15 104537](https://github.com/user-attachments/assets/24968a57-16fa-42e7-8c37-da41cbbd5956)

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])

df

![Screenshot 2025-04-15 104703](https://github.com/user-attachments/assets/69c3c847-e448-46c5-b4ae-450a894d7665)

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])

![Screenshot 2025-04-15 104732](https://github.com/user-attachments/assets/b0a925c9-77b0-46a0-a3a4-8efc94219fa5)

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

![Screenshot 2025-04-15 104811](https://github.com/user-attachments/assets/dfb4595f-aca4-4007-91b0-35cc8a2b4c60)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')

plt.show()

![Screenshot 2025-04-15 104845](https://github.com/user-attachments/assets/4291236f-a019-4aa3-8376-873e0db57832)

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

![Screenshot 2025-04-15 104945](https://github.com/user-attachments/assets/fea10e0f-79e5-421f-b475-85e1499284c2)

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df["Highly Negative Skew"],line='45')

plt.show()   

![Screenshot 2025-04-15 105026](https://github.com/user-attachments/assets/04fa4f39-a9d8-4830-a912-23eed96d9afd)

sm.qqplot(df["Highly Negative Skew_1"],line='45')

plt.show()

![Screenshot 2025-04-15 105110](https://github.com/user-attachments/assets/84ba5626-266d-4bac-8579-84d159ebc5e1)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

plt.show()

![Screenshot 2025-04-15 105144](https://github.com/user-attachments/assets/e80b4f8d-00d8-44c2-b2d2-f7ca3ae233ef)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
