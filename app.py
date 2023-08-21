

import streamlit as st
import joblib,os
from joblib import dump, load
import scipy
import spacy
import pandas as pd
nlp = spacy.load("en_core_web_sm")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from wordcloud import WordCloud


# load Vectorizer
complaints_cv = load("models/tfidf_vect.joblib")

def load_prediction_models(model_file):

	loaded_model = load(model_file)
	return loaded_model

# Get the Keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key




def main():

	"""Telecom Complaints Classifier"""
	st.title("Comcast Telecom Complaints App")
	
	# Layout Templates
	html_temp = """
	<div style="background-color:#D5CC8F;padding:10px;border-radius:10px;margin:10px;">
	<h1 style="color:white;text-align:center;"> ML - Telecom Complaints Classifier </h1>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;width: 50px;height: 50px;border-radius: 50%;" >
	<p style="text-align:justify">{}</p>
	</div>
	"""
	title_temp ="""
	<div style="background-color:#D5CC8F;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">{Debmalya Ray}</h1>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<h6>Author:{Debmalya Ray}</h6>
	<br/>
	<br/>
	<p style="text-align:justify">{}</p>
	</div>
	"""
	article_temp ="""
	<div style="background-color:#D5CC8F;padding:10px;border-radius:5px;margin:10px;">
	<h4 style="color:white;text-align:center;">{Debmalya Ray}</h1>
	<h6>Author:{Debmalya Ray}</h6>
	<h6>Post Date: {}</h6>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>
	<p style="text-align:justify">{}</p>
	</div>
	"""


	st.markdown(html_temp,unsafe_allow_html=True)

	activity = ['Prediction','NLP','About']
	choice = st.sidebar.selectbox("Select Activity",activity)


	if choice == 'Prediction':
		st.info("Prediction with ML")
		complaints_text = st.text_area("Enter Complaints Here","Type Here")
		all_ml_models = ["Decision Tree", "GradientBoost", "RandomForest", "Adaboost"]
		model_choice = st.selectbox("Select Model",all_ml_models)

		prediction_labels = {'Closed': 0, 'Open': 1, 'Pending': 2, 'Solved': 3}
		if st.button("Classify"):
			st.text("Original Text:\n{}".format(complaints_text))
			vect_text = complaints_cv.transform([complaints_text]).toarray()
			if model_choice == 'Decision Tree':
				predictor = load_prediction_models("models/dtcpred.joblib")
				prediction = predictor.predict(vect_text)
				st.write(prediction)
			elif model_choice == 'GradientBoost':
				predictor = load_prediction_models("models/gbcpred.joblib")
				prediction = predictor.predict(vect_text)
				st.write(prediction)
			elif model_choice == 'RandomForest':
				predictor = load_prediction_models("models/rbcpred.joblib")
				prediction = predictor.predict(vect_text)
				st.write(prediction)
			elif model_choice == 'Adaboost':
				predictor = load_prediction_models("models/adacpred.joblib")
				prediction = predictor.predict(vect_text)
				st.write(prediction)

			final_result = get_key(prediction,prediction_labels)
			st.success("Complaints Categorized as: {}".format(final_result))

	elif choice == 'NLP':
		st.info("Natural Language Processing of Text")
		raw_text = st.text_area("Enter Customer Complaints Here","Type Here")
		nlp_task = ["Tokenization", "Parts-of-Speech(POS) Tags"]
		task_choice = st.selectbox("Choose NLP Task",nlp_task)
		if st.button("Analyze"):
			st.info("Original Text:\n{}".format(raw_text))

			docx = nlp(raw_text)
			if task_choice == 'Tokenization':
				result = [token.text for token in docx ]
			elif task_choice == 'Parts-of-Speech(POS) Tags':
				result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]

			st.json(result)


		if st.checkbox("WordCloud"):
			c_text = raw_text
			wordcloud = WordCloud().generate(c_text)
			plt.imshow(wordcloud,interpolation='bilinear')
			plt.axis("off")
			st.set_option('deprecation.showPyplotGlobalUse', False)
			st.pyplot()


	else:
		st.write("")
		st.subheader("About")
		st.write("""**************************************************************************""")
		st.markdown("""
        ### NLP Complaints Classifier With Different Models (With Streamlit)
        ###### Python Tools Used: spacy, pandas, matplotlib, wordcloud, Pillow(PIL), Joblib
        """)
		st.write("""**************************************************************************""")
		st.write("""
        361148 || Throttling service and unreasonable data caps	 || 24-06-2015 || Acworth || Georgia || 30101 || Pending  
        """)
		st.write("""
        359792 || Comcast refuses to help troubleshoot and correct my service. || 23-06-2015 || Adrian || Michigan || 49221 || Solved 
        """)
		st.write("""
        371214 || Comcast Raising Prices and Not Being Available To Ask Why || 28-06-2015 || Alameda || California || 94501 || Open  
        """)
		st.write("""
        242732 || Speed and Service || 18-04-2015 || Acworth || Georgia || 30101 || Closed
        """)
		st.write("""**************************************************************************""")


if __name__ == '__main__':
	main()


