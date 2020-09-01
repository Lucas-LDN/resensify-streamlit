import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import csv

CSV_FILEPATH_YELP_SAMPLE = os.path.join(
                                        os.getcwd(),
                                        "..",
                                        "data",
                                        "data_sample_AZ_NV.csv"
                                        )

####################
# functions
####################
def load_yelp_sample_AZ_NV(file):
    import os
    import csv
    import pandas as pd
    df = pd.read_csv(file)
    df['text'] = df['text'].astype('str')
    print(f"\n{df.info()}\nUPLOAD COMPLETE\n")
    return df


#sidebars
st.sidebar.header("About the team")
st.sidebar.markdown("""
  We're a great bunch of data scientists.
  """)
resensify_team_sidebar = Image.open('resensify_meet_the_team.PNG')
st.sidebar.image(
                 resensify_team_sidebar,
                 caption='Lucas, Mohammed and Paulette',
                 use_column_width=True
                 )

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
  st.image(
           resensify_team_image,
           caption='(left to right: Lucas, Mohammed and Paulette)',
           use_column_width=True
           )

st.header("Choose a presentation")
status = st.radio(
                  " ",
                  (
                    "Original presentation",
                    "Yelp Sentiment Analysis model")
                  )
url = 'https://docs.google.com/presentation/d/1RnE1slwuDa4GWgzOedUXr0hjqC9LdFwWLYuJov2eDOA/edit?usp=sharing'
if status == "Yelp Sentiment Analysis model":
  st.success("This will launch a link")
  import webbrowser
  webbrowser.open_new_tab(url)
else:
  st.info("To view today's presentation, click Yelp Sentiment Analysis model")


# Dummy to show how to load the data with a button
st.header("Upload your file")
selectbox_choose_file = st.selectbox(
                                     "Select the location of file to be uploaded:",
                                     ["Local: data/Yelp_training_data.json",
                                     "GCP: Yelp_training_data.json",
                                     "Local: Yelp_test_data.json",
                                     "GCP: Yelp_test_data.json"]
                                     )
st.write("You selected: ", selectbox_choose_file)
if st.button('Load data'):
  st.markdown("""
    The following has been carried out on the data:
    ** New column added: Sentiment** - this converts the star rating to a sentiment (1-3: Negative, 4-5: Positive)
    """)
  df = pd.read_csv('sample_reviews.csv')
  line_count = st.slider('Select a line count', 1, 10, 3)
  # and used in order to select the displayed lines
  head_df = df.head(line_count)
  head_df

  # progress bar
  import time
  my_bar=st.progress(0)
  for p in range(0,10):
    my_bar.progress(p+2)

  #spinner
  with st.spinner("Waiting.."):
    time.sleep(5)
  st.success("Load complete")

#Text area
st.title("Let BERT predict")
predict_df = pd.read_csv('sample_reviews.csv')
review1 = predict_df.text[0]
review2 = predict_df.text[1]
status = st.radio(" ", (review1, "hard coded review"))
message = st.text_area("Enter your review to predict", review1)

if st.button('BERT predict sentiment'):
  st.success("This review is: NEGATIVE")

if st.button('BERT predict rating'):
  st.success("This rating would be: 1 star")




if __name__ == '__main__':
    # df = load_yelp_sample_AZ_NV(CSV_FILEPATH_YELP_SAMPLE)
    # df = pd.DataFrame({'reviews': df['text'], 'rating': df['stars_x']})
  st.success("Application loaded")
  if st.button('Presentation complete'):
    st.balloons()
