import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import numpy as np
import pandas as pd
from dash.dependencies import Input,Output
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('diabetes.csv')
from scipy import stats

z=np.abs(stats.zscore(df.Pregnancies))
threshold=3
data=df[(z<3)]
clean_data=df.drop(['SkinThickness','Insulin'],axis=1)
df1=clean_data.drop(['DiabetesPedigreeFunction','BloodPressure'],axis=1)
# dealing with imbalanced data

count_class_0,count_class_1=df1.Outcome.value_counts()
target_class_0=df1[df1['Outcome']==0]
target_class_1=df1[df1['Outcome']==1]
data_class_1_over=target_class_1.sample(count_class_0,replace=True)
data_test_over=pd.concat([target_class_0,data_class_1_over],axis=0)

# Create X (all the feature columns)
x=data_test_over.drop("Outcome",axis=1)

# Create y (the target column)
y=data_test_over["Outcome"]
# Split the data into training and test sets
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier(n_neighbors=1)
model.fit(x_train,y_train)

# saving model using joblib
import joblib

joblib.dump(model,'model_save2')

model1=joblib.load('model_save2')

external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']

fig=px.scatter(df,x="BMI",y="Age",
               size="Pregnancies",color="Glucose",
               title='RELATIONSHIP ANALYSIS WITHIN THE VARIABLES')
fig2=px.bar(df,x="Age",y="DiabetesPedigreeFunction",color="Glucose",barmode="group")
app=dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.layout=html.Div([
    html.Div([
        html.H1(
            children=" DIABETIC ANALYSIS DASHBOARD",
            style={
                'color':'white',
                'backgroundColor':'grey',
            }
        )
    ],className='row'),
    html.Div([
        html.Div([
            html.P(
                'Diabetes is a disease in which the body ability to produce or respond to the hormone insulin is impaired resulting in abnormal metabolism of carbs and elevated levels of glucose in the blood and urine. ')
        ])

    ]),
    html.Div([
        html.Div([
            dcc.Graph(id='relationship with variables',
                      figure=fig),

        ],className='six columns'),
        html.Div([
            dcc.Graph(id='analysis on glucose levels and age',
                      figure=fig2),
        ],className='six columns'),
    ],className='row'),
    html.Div([
        html.Div([
            html.P(
                'From the analysis above on the chart at our right we can conclude that woman who have a body mass index (BMI)of between 20 to 40and also of age 20 to 30 have moderate levels of glucose meaning that their chances of being diabetic are low,and it also shows that woman of old age have high levels of glucose in their blood stream which increases their chances of being diabetic.On the other chart we can see that woman who have high levels of diabetesPedigreeFunction are of age 21 to 34 years .Diabetes Pedigree Function provides history of relatives and genetic relationship of those relatives with patients.Higher HPF means patient is more likely to have diabetes ')
        ])
    ]),
    html.Div([
        html.Div([
            html.H1(children="MACHINE LEARNING COMPONENT",style={
                'color':'white','backgroundColor':'grey'
            })
        ])
    ]),
    html.Div(children=[
        html.H1(children='DIABETES ANALYSIS ',style={'textAlign':'center'}),
        html.Div(children=[
            html.Label('ENTER NUMBER OF PREGNANCIES CARRIED'),
            dcc.Input(id='input_1',placeholder='NUMBER OF PREGNANCIES CARRIED',type='number'),

        ]

        ),
        html.Div(children=[
            html.Label('ENTER GLUCOSE LEVEL'),
            dcc.Input(id='input_2',placeholder='GLUCOSE LEVELS',type='number')

        ]

        ),
        html.Div(children=[
            html.Label('ENTER YOUR BODY MASS INDEX'),
            dcc.Input(id='input_3',placeholder='BMI',type='number')
        ])

    ]

    ),
    html.Div(children=[
        html.Label('ENTER YOUR AGE'),
        dcc.Input(id='input_4',placeholder='AGE',type='number'),
        html.Div(id='prediction_result')

    ]),
    html.Div([
        html.Div([
            html.P(children='IF THE RESULT IS 1 ,PLEASE CONSULT THE DOCTOR AS SOON AS POSSIBLE!')


        ])
    ]),
    html.Div([
        html.Div([
            html.P(children='IF THE RESULT IS 0 , CONGRATS YOU ARE DIABETIES FREE')
        ])
    ])

])


@app.callback(Output(component_id='prediction_result',component_property='children'),
              [Input(component_id='input_1',component_property='value'),
               Input(component_id='input_2',component_property='value'),
               Input(component_id='input_3',component_property='value'),
               Input(component_id='input_4',component_property='value')],
               prevent_initial_call=False


              )
def prediction(value_1,value_2,value_3,value_4):
    input_X=np.array([value_1,
                      value_2,
                      value_3,
                      value_4

                      ]).reshape(1,-1)
    scaler=MinMaxScaler(feature_range=(0,1))
    scaler.fit(input_X)
    diabetes=scaler.transform(input_X)
    if prediction is not None:
        try:
            prediction_result=model1.predict(diabetes)
            return prediction_result

        except ValueError:
            'unable to process'


if __name__=='__main__':
    app.run_server(debug=True)
