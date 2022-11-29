#!/usr/bin/env python3

import streamlit as st
from streamlit_chat import message
import requests
import pandas as pd
import deepl
from ftlangdetect import detect


if "char_usage" not in st.session_state:
    st.session_state.char_usage = 0

if "char_limit" not in st.session_state:
    st.session_state.char_limit = 350000.0

## count the remaining characters from the session states 
if "remaining_characters" not in st.session_state:
    st.session_state.remaining_characters = 0

## access the secrets from streamlit 
DEEPL_AUTH_KEY = st.secrets["DEEPL_AUTH_KEY"]
HF_AUTH_KEY = st.secrets["HF_AUTH_KEY"]

def get_character_usage(translator: deepl.Translator, limit_pcnt: float = 0.7) -> int:
    """"
    Accessing the Usage object from the Translator object, the character count 
    is retrieved and if the usage is close to the limit, a warning is printed halting the execution
    """
    ## first get the translator object 
    usage = translator.get_usage()
    ## then get the character count
    character_count = usage._character.count
    ## add to the car usage
    st.session_state.char_usage += character_count
    ## if the character count is greater than the limit, then print a warning
    if character_count > st.session_state.char_limit:
        print(f"Warning: character count {character_count} is greater than the limit {st.session_state.char_limit}")
    ## calculate the remaining characters
    st.session_state.remaining_characters = st.session_state.char_limit - st.session_state.char_usage
    return 

def spanish_to_english(text: str, target: str = "EN-US"):
    """
    Translate the text from spanish to english 
    """
    translator = deepl.Translator(DEEPL_AUTH_KEY) 
    ## get the character count 
    get_character_usage(translator)
    print(f"Character count: {st.session_state['char_usage']}", "Character Limit:", st.session_state['char_limit'], "Remaining Characters:", st.session_state['remaining_characters'])
    result = translator.translate_text(text, target_lang=target) 
    translated_text = result.text
    return translated_text

def english_to_spanish(text: str, target: str = "ES"):
    """
    Translate the text from spanish to english 
    """
    translator = deepl.Translator(DEEPL_AUTH_KEY)
    get_character_usage(translator)
    print(f"Character count: {st.session_state['char_usage']}", "Character Limit:", st.session_state['char_limit'], "Remaining Characters:", st.session_state['remaining_characters'])
    result = translator.translate_text(text, target_lang=target) 
    translated_text = result.text
    return translated_text
## set the page config

st.set_page_config(page_title="Diageo - Chatbot - Demo", page_icon=":robot:")


## dictionary to hold the models with their respective sizes 
CHATBOT_MODELS = {
        # "Small":{
        #     "90M":"https://api-inference.huggingface.co/models/facebook/blenderbot_small-90M",
        # },
        "Medium":{
            "90M":"https://api-inference.huggingface.co/models/facebook/blenderbot-90M",
            "400M_Distilled":"https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill",

        },
        "Large":{
            "1B_Distilled":"https://api-inference.huggingface.co/models/facebook/blenderbot-1B-distill",
            "3B":"https://api-inference.huggingface.co/models/facebook/blenderbot-3B",  ## this one is the best one, so let's use it
            "9B":"https://api-inference.huggingface.co/models/hyunwoongko/blenderbot-9B",
        }   
    }



## get the API to connect
# API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-3B"

## set some headers and info
st.header("Diageo - Chatbot - Demo")
st.subheader("Powered by Locaria")

model_size = "Large"
model = "3B"

## load the lookup 
df = pd.read_csv("./test_data/test_audios.csv") ## show 3 columns, which will contain 3 different spanish dialects: Argentine, Colombian, Venezuelan

## load the audio file 
def load_audio(audio_file):
    audio_file = open(audio_file, 'rb')
    audio_bytes = audio_file.read()
    return audio_bytes

## display the male and female audio, under it, show the transcript 
def display_audio(audio_file, transcript, col):
    ## get the name of the variable name that is passed as audio_file
    audio_bytes = load_audio(audio_file)
    col.audio(audio_bytes, format='audio/wav')
    col.write(transcript)

## header : Sample Dialects 
st.markdown("This section shows the possible variations of the dialect that we'd be able to synthesize. There are 3 different dialects: Argentine, Colombian, Venezuelan")
st.markdown("Our dialect-classification algorithm is also able to identify the different dialects with high-accuracy and precision.")
st.markdown("Two genders are shown, to demonstrate the different dialects that can be generated. The audio is played and the transcript is shown below it.")

st.header("Sample Dialects")
with st.expander("Show the audio files"):
    arg, col, ven = st.columns(3)
    ## Argentine Dialects 
    arg_male = df[df['target'] == 'es_ar_male']['file_path'].values[0]
    arg_male_tr = df[df['file_path'] == arg_male]['transcript'].values[0]

    arg_female = df[df['target'] == 'es_ar_female']['file_path'].values[0]
    arg_fem_tr = df[df['file_path'] == arg_female]['transcript'].values[0]

    ## Colombian Dialects 
    col_male = df[df['target'] == 'es_co_male']['file_path'].values[0]
    col_male_tr = df[df['file_path'] == col_male]['transcript'].values[0]

    col_female = df[df['target'] == 'es_co_female']['file_path'].values[0]
    col_fem_tr = df[df['file_path'] == col_female]['transcript'].values[0]

    ## Venezuelan Dialects 
    ven_male = df[df['target'] == 'es_ve_male']['file_path'].values[0]
    ven_male_tr = df[df['file_path'] == ven_male]['transcript'].values[0]

    ven_female = df[df['target'] == 'es_ve_female']['file_path'].values[0]
    ven_fem_tr = df[df['file_path'] == ven_female]['transcript'].values[0]


    ## display the audio and transcript
    arg.markdown("#### Argentine")
    arg.markdown("##### Female")
    display_audio(arg_female, arg_fem_tr, arg)
    arg.markdown("##### Male")
    display_audio(arg_male, arg_male_tr, arg)

    col.markdown("#### Colombian")
    col.markdown("##### Female")
    display_audio(col_female, col_fem_tr, col)
    col.markdown("##### Male")
    display_audio(col_male, col_male_tr, col)

    ven.markdown("### Venezuela")
    ven.markdown("##### Female")
    display_audio(ven_female, ven_fem_tr, ven)
    ven.markdown("##### Male")
    display_audio(ven_male, ven_male_tr, ven)

## an audio file needs to be played, as well as showing the corresponding transcript 
st.markdown("The chatbot below is Locaria's custom chatbot, which is able to receive and respond to spanish and english. The current version will consider a single language. If the language is changed in the chat itself, the entire chat will be translated to the new language.")
## INFO
API_URL = CHATBOT_MODELS[model_size][model]
headers = {"Authorization": f"Bearer {HF_AUTH_KEY}"}

## keep session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []

## iuf the past is not empty, show it
if "past" not in st.session_state:
    st.session_state["past"] = []

## define function to halt any further execution if the character limit is reached
def check_character_limit():
    if st.session_state['char_usage'] >= st.session_state['char_limit']:
        st.error("Time limit & Processing Limit Reached. No further interaction will be allowed.")
        st.stop()


def query(payload):
    check_character_limit()
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


## the get response function
def get_response():
    ## open an input text
    input_text = st.text_input("Type your message here", key="input_text")
    return input_text


## all of this assumes that the language is english
## consistency is key, so we need to keep the same languages! 
## the user input
## add a separator 
st.markdown("---")
## add a header
st.header("Chatbot")
## add a description of the chatbot 

st.markdown("This chatbot is able to reply in english and in spanish. For the demonstration purposes, a single language is used (if the language is changed, the entire chat history will be translated)")

user_input =  st.text_input("Type your message here", key="input_text")

input_language = detect(user_input, low_memory=True)['lang']


if user_input:
    output = query(
        {
            "inputs": {
                "past_user_inputs": st.session_state["past"],
                "generated_responses": st.session_state["generated"],
                "text": user_input if input_language == 'en' else spanish_to_english(user_input),
            },
            "parameters": {"repetition_penalty": 1.9},
        }
    )
    ## update the sessions state with the past history and the generated respons
    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(output['generated_text'])    

## check if response has been generated
if st.session_state["generated"]:
    ## now generate all the messages
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        ## if the language is in english, translate it 
        generated_messages = st.session_state["generated"][i]
        past_messages = st.session_state["past"][i]
        if input_language == 'es':
            generated_messages = english_to_spanish(generated_messages)
            past_messages = english_to_spanish(past_messages)
        message(generated_messages, key=str(i))
        message(past_messages, is_user=True, key=str(i) + "_user")
