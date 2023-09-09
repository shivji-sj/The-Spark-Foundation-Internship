import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
import streamlit as st


def predict(study_time=0):

	df = pd.read_csv(r"C:\Users\SHIV-G\Desktop\Desk\SPARK INTERNSHIP\TASK - 1 PREDICTION USING SUPERVISED LEARNING\data.txt" )

	# splitting the dataset into the dependent and independent set
	X = np.array(df.Hours).reshape(-1,1)     # independent set
	Y = np.array(df.Scores)                  # dependent set

	X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

	# linear model
	lr = LinearRegression()
	lr.fit(X_train,y_train)

	# making prediction
	hours = study_time

	new_df = pd.DataFrame({"Hours": hours}, index=[0])
	pred_score = lr.predict(new_df)

	return pred_score



########################################################
st.title('Student Scores Prediction')  ## title of app
st.write("""This is a Student Score Predictor prject using Linear Regression Algorithm. """)   ### description for app
study_hour = st.number_input("Study Hours")

ps = predict(study_time=study_hour)


if st.button("Predict"):
	with st.spinner():
		st.subheader(f'Predicted Score : { np.round(ps, 2)}%')


# Follow me link
github_url = "https://github.com/shivji-sj"
linkedin_url ="https://www.linkedin.com/in/shivji-881449205/recent-activity/"
medium_url = "https://medium.com/@sjshivji"

st.header("Follow Me : ")

col1, col2, col3 = st.columns(3)

with col1:
	st.markdown("[GitHub](%s)" % github_url)
with col2:
	st.markdown("[LinkedIn](%s)" % linkedin_url)
with col3:
	st.markdown("[Medium](%s)" % medium_url)