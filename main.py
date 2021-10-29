import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


#source: https://pypi.org/project/streamlit-analytics/
import streamlit_analytics

# We use streamlit_analytics to track the site like in Google Analytics
streamlit_analytics.start_tracking()

# configuring the page and the logo
st.set_page_config(page_title='Mohamed Gabr - House Price Prediction', page_icon ='logo.png', layout = 'wide', initial_sidebar_state = 'auto')


import os
import base64

# the functions to prepare the image to be a hyperlink
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


# preparing the layout for the top section of the app
# dividing the layout vertically (dividing the first row)
row1_1, row1_2, row1_3 = st.columns((1, 4, 5))

# first row first column
with row1_1:
    gif_html = get_img_with_href('logo.png', 'https://golytics.github.io/')
    st.markdown(gif_html, unsafe_allow_html=True)

with row1_2:
    # st.image('logo.png')
    st.title("Predicting The Class of the Iris Flower")
    st.markdown("<h2>A Famous Machine Learning Project (Practical Project for Students)</h2>", unsafe_allow_html=True)

# first row second column
with row1_3:
    st.info(
        """
        ##
        This data product has been prepared to be used as a practical project in the training courses provided by Dr. Mohamed Gabr. Developing the final model required
        many steps following the CRISP-DM methodology. After building the model we used it to predict the iris folower type in this application.
        """)



st.write("""

This app predicts the **Iris flower** type!
""")
st.write("""We have 3 types as shown in the below image""")

images_list=['setosa.png', 'virginica.png', 'versicolor.png']
st.image(images_list, width=200)

st.subheader('How to use the model?')
'''
You can use the model by modifying the User Input Parameters on the left. The parameters will be passed to the classification
model and the model will run each time you modify the parameters.

1- You will see the values of the features/ parameters in the **'User Input Parameters'** section in the table below.

2- The names of the classes are Setosa, Versicolor, and Virginica. These classes are listed under the **Class Labels and Their Corresponding Index Numbers** section below.

3- You will see the **'prediction propability'** (: the propability that the user input parameters match one of the 3 classes).

4- You will see the prediction result (the type of Iris flower) under the **'Prediction'** section below.

'''


st.sidebar.header("""User input features/ parameters: 

Select/ modify the combination of features below to predict the Iris flower type
                """)


# Here, we create a custom function to accept all of the input parameters from the sidebar to create a dictionary that will be
# passed to a Pandas dataframe and show it in the User Input part of the screen
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4) # deafult value is 5.4, min is 4.3, max is 7.9
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features
# we create a Pandas dataframe to put the dictionary in
df = user_input_features()



st.subheader('User Input parameters')
st.write(df)

# Here, we load the Iris dataset then assign the predictors (Sepal length, Sepal width, Petal length, Petal length) to X and
# the target (the class index number : 0,1,2) to Y
iris = datasets.load_iris()
# print(iris)
X = iris.data
Y = iris.target

# create the classifier and apply it on X to predict Y
clf = RandomForestClassifier()
clf.fit(X, Y)

# Here, we make the prediction and calculate the prediction propability (the propability that the user input parameters is in one of the
# 3 classes (0= setoza, 1= versicolor, 2= virginica)
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


# Note: everytime we change a parameter, the code builds the prediction model again and again. This uses the resources. We print this below
# statement to see how the model runs everytime we change the parameters. So, we should think about enhancing the performance.
# To avoid this, we can save the model in a file (*.pkl file using the module PICKLE) and then read this file with our code.
#print("new iteration")

# print the classes (class index number and class names) in the tables
st.subheader('Class Labels and Their Corresponding Index Numbers')
st.write(iris.target_names)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Prediction')
#st.write(iris.target_names[prediction])

predicted_class=iris.target_names[prediction]

html_str = f"""
<h3 style="color:lightgreen;">{predicted_class[0]}</h3>
"""

st.markdown(html_str, unsafe_allow_html=True)
st.image(predicted_class[0]+'.png', width=200)
#st.write(prediction)



st.info("""**Note: ** [The data source is]: ** (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris). the following steps have been applied till we reached the model:

        1- Data Acquisition/ Data Collection (reading data, adding headers)

        2- Data Cleaning / Data Wrangling / Data Pre-processing (handling missing values, correcting data fromat/ data standardization 
        or transformation/ data normalization/ data binning/ Preparing Indicator or binary or dummy variables for Regression Analysis/ 
        Saving the dataframe as ".csv" after Data Cleaning & Wrangling)

        3- Exploratory Data Analysis (Analyzing Individual Feature Patterns using Visualizations/ Descriptive statistical Analysis/ 
        Basics of Grouping/ Correlation for continuous numerical variables/ Analysis of Variance-ANOVA for ctaegorical or nominal or 
        ordinal variables/ What are the important variables that will be used in the model?)

        4- Model Development (Single Linear Regression and Multiple Linear Regression Models/ Model Evaluation using Visualization)

        5- Polynomial Regression Using Pipelines (one-dimensional polynomial regession/ multi-dimensional or multivariate polynomial 
        regession/ Pipeline : Simplifying the code and the steps)

        6- Evaluating the model numerically: Measures for in-sample evaluation (Model 1: Simple Linear Regression/ 
        Model 2: Multiple Linear Regression/ Model 3: Polynomial Fit)

        7- Predicting and Decision Making (Prediction/ Decision Making: Determining a Good Model Fit)

        8- Model Evaluation and Refinement (Model Evaluation/ cross-validation score/ over-fitting, under-fitting and model selection)

""")

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

footer = """<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>&copy; <a href="https://golytics.github.io/" target="_blank">GoLytics</a><br>Developed By: <a href="https://golytics.github.io/" target="_blank">Dr. Mohamed Gabr</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

streamlit_analytics.stop_tracking(unsafe_password="forwardgbrbreen12")