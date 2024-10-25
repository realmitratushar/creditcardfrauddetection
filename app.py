import pandas as pd

df=pd.read_csv("creditcardfrauddataset.csv")
df.head()

df.columns

df.info()

df.describe()

df.describe().T

from ydata_profiling import ProfileReport
profile = ProfileReport(df, title = "Pandas Profiling Report")
profile.to_file("pandas_profiling_report.html")

df.isnull().sum()

df.duplicated().sum()

df.skew()

x = df.drop(labels=['default payment next month'], axis = 1)
y = df[['default payment next month']]

x

y

import seaborn as sns
sns.pairplot(df)

import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot = True)
plt.show()

df.shape

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

pipeline=Pipeline(
    steps=[
      ('imputer',SimpleImputer(strategy='median')),
      ('scaler',StandardScaler())
    ]
)

preprocessor=ColumnTransformer([
    ('num_pipeline',pipeline,x.columns)
    ]
)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=30)

x_train=pd.DataFrame(preprocessor.fit_transform(x_train),columns=preprocessor.get_feature_names_out())
x_test=pd.DataFrame(preprocessor.transform(x_test),columns=preprocessor.get_feature_names_out())

import numpy as np
def evaluate_model(true, predicted):
    ac = accuracy_score(true,predicted)
    cm = confusion_matrix(true,predicted)
    cr = classification_report(true,predicted)
    return ac,cm,cr


models={
    'DecisionTreeClassifier':DecisionTreeClassifier(),
    'SVC':SVC(),
    'RandomForestClassifier':RandomForestClassifier(),
    'Gaussian Naive Bayes':GaussianNB()
}
trained_model_list=[]
model_list=[]
ac_list=[]

for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(x_train,y_train)

    #Make Predictions
    y_pred=model.predict(x_test)

    ac,cm,cr=evaluate_model(y_test,y_pred)

    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print('Model Training Performance')
    print("Accuracy Score:",ac)
    print("Confusion Matrix:",cm)
    print("Classification Report:",cr)

    ac_list.append(ac)

    print('='*35)
    print('\n')

from sklearn.model_selection import GridSearchCV
param_gridDTC = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
    }
param_gridSVC = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
    }
param_gridGNB = {
    'var_smoothing': [1e-3, 1e-4, 1e-5, 1e-6]
    }
param_gridRFC = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
    }

gridDTC = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = param_gridDTC, cv = 5, verbose = 3)

gridSVC = GridSearchCV(estimator = SVC(), param_grid = param_gridSVC, cv = 5, verbose = 3)

gridGNB = GridSearchCV(estimator = GaussianNB(), param_grid = param_gridGNB, cv = 5, verbose = 3)

gridRFC = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_gridRFC, cv = 5, verbose = 3)

gridDTC.fit(x_train,y_train)

gridSVC.fit(x_train,y_train)

gridGNB.fit(x_train,y_train)

gridRFC.fit(x_train,y_train)

gridDTC.best_estimator_

gridSVC.best_estimator_

gridGNB.best_estimator_

gridRFC.best_estimator_

gridDTC.best_params_

gridSVC.best_params_

gridGNB.best_params_

gridRFC.best_params_

dtc=DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features='sqrt',
                       min_samples_leaf=2, min_samples_split=10)

svc=SVC(C=1, degree=2, gamma='auto', kernel='rbf')

gnb=GaussianNB(var_smoothing=0.001)

rfc=RandomForestClassifier(bootstrap=False, max_depth=20, max_features='log2',
                       min_samples_leaf=4, min_samples_split=10, n_estimators=200)

import numpy as np
def evaluate_model(true, predicted):
    ac = accuracy_score(true,predicted)
    cm = confusion_matrix(true,predicted)
    cr = classification_report(true,predicted)
    return ac,cm,cr


models={
    'DecisionTreeClassifier':dtc,
    'SVC':svc,
    'RandomForestClassifier':rfc,
    'Gaussian Naive Bayes':gnb
}
trained_model_list=[]
model_list=[]
ac_list=[]

for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(x_train,y_train)

    #Make Predictions
    y_pred=model.predict(x_test)

    ac,cm,cr=evaluate_model(y_test,y_pred)

    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print('Model Training Performance')
    print("Accuracy Score:",ac)
    print("Confusion Matrix:",cm)
    print("Classification Report:",cr)

    ac_list.append(ac)

    print('='*35)
    print('\n')

import gradio as gr

def default_prediction(LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6):
    data = {
        'LIMIT_BAL': [LIMIT_BAL],
        'SEX': [SEX],
        'EDUCATION': [EDUCATION],
        'MARRIAGE': [MARRIAGE],
        'AGE': [AGE],
        'PAY_0': [PAY_0],
        'PAY_2': [PAY_2],
        'PAY_3': [PAY_3],
        'PAY_4': [PAY_4],
        'PAY_5': [PAY_5],
        'PAY_6': [PAY_6],
        'BILL_AMT1': [BILL_AMT1],
        'BILL_AMT2': [BILL_AMT2],
        'BILL_AMT3': [BILL_AMT3],
        'BILL_AMT4': [BILL_AMT4],
        'BILL_AMT5': [BILL_AMT5],
        'BILL_AMT6': [BILL_AMT6],
        'PAY_AMT1': [PAY_AMT1],
        'PAY_AMT2': [PAY_AMT2],
        'PAY_AMT3': [PAY_AMT3],
        'PAY_AMT4': [PAY_AMT4],
        'PAY_AMT5': [PAY_AMT5],
        'PAY_AMT6': [PAY_AMT6]
    }

    data_df = pd.DataFrame(data)

    processed_data = preprocessor.transform(data_df)

    default_predict = rfc.predict(processed_data)
    if default_predict[0]==0:
      return "The person will not default"
    else:
      return "The person will default"

iface = gr.Interface(
    fn=default_prediction,
    inputs=[
        gr.Number(label='Balance Limit(Maximum Credit Limit for the User)'),
        gr.Dropdown(choices=['1','2'],label='Sex(1 for Male & 2 for Female)'),
        gr.Dropdown(choices=['1', '2', '3', '4', '5', '6'],label='Education(1(Graduate School),2(University),3(High School), 4(Middle School), 5(Elementary School), 6(Illiterate))'),
        gr.Dropdown(choices=['0','1','2','3'],label='Marriage(0(Others),1(Married),2(Single),3(Divorced))'),
        gr.Number(label='Age(Enter your Age in Years)'),
        gr.Dropdown(choices=['-2','-1','0','1','2','3','4','5','6','7','8'],label='Pay_0(Payment status for last month)'),
        gr.Dropdown(choices=['-2','-1','0','1','2','3','4','5','6','7','8'],label='Pay_2(Payment status for 2 months ago)'),
        gr.Dropdown(choices=['-2','-1','0','1','2','3','4','5','6','7','8'],label='Pay_3(Payment status for 3 months ago'),
        gr.Dropdown(choices=['-2','-1','0','1','2','3','4','5','6','7','8'],label='Pay_4(Payment status for 4 months ago)'),
        gr.Dropdown(choices=['-2','-1','0','1','2','3','4','5','6','7','8'],label='Pay_5(Payment status for 5 months ago)'),
        gr.Dropdown(choices=['-2','-1','0','1','2','3','4','5','6','7','8'],label='Pay_6(Payment status for 6 months ago)'),
        gr.Number(label='Bill Amount 1(Bill Amount for last month)'),
        gr.Number(label='Bill Amount 2(Bill Amount for 2 months ago)'),
        gr.Number(label='Bill Amount 3(Bill Amount for 3 months ago)'),
        gr.Number(label='Bill Amount 4(Bill Amount for 4 months ago)'),
        gr.Number(label='Bill Amount 5(Bill Amount for 5 months ago)'),
        gr.Number(label='Bill Amount 6(Bill Amount for 6 months ago)'),
        gr.Number(label='Pay Amount 1(Amount paid for last month)'),
        gr.Number(label='Pay Amount 2(Amount paid for 2 months ago)'),
        gr.Number(label='Pay Amount 3(Amount paid for 3 months ago)'),
        gr.Number(label='Pay Amount 4(Amount paid for 4 months ago)'),
        gr.Number(label='Pay Amount 5(Amount paid for 5 months ago)'),
        gr.Number(label='Pay Amount 6(Amount paid for 6 months ago)')
    ],
    outputs="text",
    title="Credit Card Default Predictor",
    description="Enter credit card related details to to predict default"
)

iface.launch(debug=True,share=True)