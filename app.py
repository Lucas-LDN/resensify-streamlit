import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import csv
from Resensify.data import get_data, clean_df
from Resensify.predict import download_model_local, generate_predictions_from_list


####################
# MOHAMAD FUNCTIONS
####################

@st.cache(suppress_st_warning=True)
def load_reviews(nrows=1000):
  df = get_data(nrows)
  #df = clean_df(df)
  return df

def load_model_sentiment():
  model = download_model_local(2)
  return model

def load_model_star():
  model = download_model_local(5)
  return model

#sidebars
st.sidebar.header("About the team")
st.sidebar.markdown("""
  We're a great bunch of data scientists.
  """)
resensify_team_sidebar = Image.open('resensify_meet_the_team.PNG')
st.sidebar.image(resensify_team_sidebar, caption='Lucas, Mohammed and Paulette', use_column_width=True)

st.sidebar.header("What is NLP?")
st.sidebar.markdown("""
  Natural Language Processing (NLP) is a way for
  computers to analyse human language and
  derive useful meaning from it
  """)

st.sidebar.header("Our goal")
st.sidebar.markdown("""
  Implement a NLP Sentiment Analysis model:
  * To predict whether a Yelp review is positive or negative
  * Used a learning labeled set of 8m Yelp reviews.
  """)

st.sidebar.header("Our solution")
st.sidebar.markdown("""
  We chose BERT to model our data:
  * BERT is reason 1
  * BERT is reason 2
  * BERT is reason 3
  """)
#####################################
# Main page
#####################################
st.title("Resensify...make sense of sentiment")

st.header("Meet the team")
if st.checkbox("Show/Hide"):
  resensify_team_image = Image.open('resensify_meet_the_team.PNG')
  st.text("What a great bunch we are!")
  st.image(resensify_team_image, caption='(left to right: Lucas, Mohammed and Paulette)', use_column_width=True)

st.header("Choose a presentation")
status = st.radio(" ", ("Original pitch", "Yelp Sentiment Analysis model"))
url = 'https://docs.google.com/presentation/d/1RnE1slwuDa4GWgzOedUXr0hjqC9LdFwWLYuJov2eDOA/edit?usp=sharing'
if status == "Yelp Sentiment Analysis model":
  st.success("This will launch a link")
  import webbrowser
  webbrowser.open_new_tab(url)
else:
  st.warning("To view today's presentation, click Yelp Sentiment Analysis model")


# MOHAMAD PREDICT
st.title("Let BERT predict")
predict_df = load_reviews(1000)
model_star = load_model_star()
model_sentiment = load_model_sentiment()
st.header("Load sample reviews from original data")
if st.button('Get sample data'):
    fake_reviews = predict_df.text.sample(5).tolist()
    y_star_pred = generate_predictions_from_list(fake_reviews, model_star,5)
    y_sentiment_pred = generate_predictions_from_list(fake_reviews, model_sentiment,2)
    my_dict = {0:'Neg', 1:'Pos'}
    y_sent_pred = np.vectorize(my_dict.__getitem__)(y_sentiment_pred)
    df_out = pd.DataFrame(
      {'review': fake_reviews,
       'Sentiment': y_sent_pred.tolist(),
       'Star Rating': y_star_pred.tolist()
      })
    st.table(df_out)


st.header("Input your own review")
message = st.text_area("Enter your review to predict","....")

if st.button('BERT predict'):
  fake_review = [message]
  y_star_pred = generate_predictions_from_list(fake_review, model_star,5)
  y_sentiment_pred = generate_predictions_from_list(fake_review, model_sentiment,2)
  my_dict = {0:'Neg', 1:'Pos'}
  y_sent_pred = np.vectorize(my_dict.__getitem__)(y_sentiment_pred)
  if y_sent_pred[0] == 'Neg':
    st.success("This review is: :thumbsdown:")
  elif  y_sent_pred[0] == 'Pos':
    st.success("This review is: :thumbsup:")
  st.success("This review gets: "+"⭐"*y_star_pred[0])

if __name__ == '__main__':
    # df = load_yelp_sample_AZ_NV(CSV_FILEPATH_YELP_SAMPLE)
    # df = pd.DataFrame({'reviews': df['text'], 'rating': df['stars_x']})
  if st.button('Presentation complete'):
    st.balloons()
