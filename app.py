import joblib
import streamlit as st 
from PIL import Image
import pandas as pd
import base64

# Load Model & Scaler & Polynomial Features

model = joblib.load('model.pkl')
sc = joblib.load('sc.pkl')
pf = joblib.load('pf.pkl')

# Load dataset

df_final = pd.read_csv('test.csv')
df_main = pd.read_csv('main.csv')

# Load Image

image = Image.open('img.png')

# Multilingual labels and content
labels = {
    "en": {
        "country_label": "Country",
        "crop_label": "Crop",
        "rainfall_label": "Average Rainfall (mm-per-year)",
        "pesticides_label": "Pesticides per Tonnes Use (tonnes of active ingredients)",
        "temp_label": "Average Temperature (degree celcius)",
        "submit_button": "Predict",
        "result_label": "The Production of Crop Yields: "
    },
    "fr": {
        "country_label": "Pays",
        "crop_label": "Culture",
        "rainfall_label": "Précipitations moyennes (mm par an)",
        "pesticides_label": "Pesticides par tonne utilisée (tonnes d'ingrédients actifs)",
        "temp_label": "Température moyenne (degré Celsius)",
        "submit_button": "Prédire",
        "result_label": "La production de rendement des cultures : "
    },
    "mr": {
        "country_label": "देश",
        "crop_label": "फसल",
        "rainfall_label": "सरासर पाऊस (मिमी-प्रतिवर्ष)",
        "pesticides_label": "टन वापरलेल्या प्रति किटकनाशक (सक्रिय घटकांच्या टन)",
        "temp_label": "सरासर तापमान (डिग्री सेल्सियस)",
        "submit_button": "अंदाज",
        "result_label": "फसलाच्या उत्पादनाचे उत्पादन: "
    },
    "hi": {
        "country_label": "देश",
        "crop_label": "फसल",
        "rainfall_label": "औसत वर्षिक वर्षा (मिमी प्रतिवर्ष)",
        "pesticides_label": "प्रति टन उपयोग किए गए कीटनाशक (सक्रिय तत्वों के टन)",
        "temp_label": "औसत तापमान (डिग्री सेल्सियस)",
        "submit_button": "अग्रिम",
        "result_label": "फसल उत्पादन: "
    },
    "ta": {
        "country_label": "நாடு",
        "crop_label": "பயிர்",
        "rainfall_label": "சராசரி மழைத்தூறல் (மி.மீ-ஆண்டு)",
        "pesticides_label": "பெருக்குத் தொகுப்பு எண்ணிக்கை (செயற்கை உற்பத்தியின் டன்னுக்கு)",
        "temp_label": "சராசரி வெப்பநிலை (டிகிரி செல்சியஸ்)",
        "submit_button": "முன்னேற்று",
        "result_label": "பயிர் உதவி உத்தியாக்கல்: "
    }
}

# Streamlit Function For Building Button & app.

def main():
    st.markdown('<h1 style="color:gold">Crop Yield Prediction</h1>', unsafe_allow_html=True)
    st.image(image, width=700)
    set_png_as_page_bg('img.png')

    # Language selection
    language = st.selectbox("Select Language", ["English", "French", "Marathi", "Hindi", "Tamil"])
    if language == "English":
        lang = "en"
    elif language == "French":
        lang = "fr"
    elif language == "Marathi":
        lang = "mr"
    elif language == "Hindi":
        lang = "hi"
    else:
        lang = "ta"
    
    country = st.selectbox(labels[lang]["country_label"], df_main['area'].unique()) 
    crop = st.selectbox(labels[lang]["crop_label"], df_main['item'].unique()) 
    average_rainfall = st.number_input(labels[lang]["rainfall_label"], value=None)
    presticides = st.number_input(labels[lang]["pesticides_label"], value=None)
    avg_temp = st.number_input(labels[lang]["temp_label"], value=None)
    input = [country, crop, average_rainfall, presticides, avg_temp]
    st.markdown('<h3 style="color:gold">1 hg = 100 grams</h3>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:gold">1 ha = 2 acres</h3>', unsafe_allow_html=True)
    result = ''
    if st.button(labels[lang]["submit_button"], ''):
        result = prediction(input)
    temp = '''
     <div style='background-color:rgb(14 17 23); padding:10px'>
     <h1 style='color: #ff9900  ; text-align: center;'>{}</h1>
     </div>
     '''.format(labels[lang]["result_label"] + str(result))
    st.markdown(temp, unsafe_allow_html=True)

# Prediction Function to predict from model.
# Albania	Soybeans	1990	7000	1485.0	121.00	16.37
# input=['Albania','Soybeans',1485.0,121.00,16.37]
def update_columns(df, true_columns):
    df[true_columns] = True
    other_columns = df.columns.difference(true_columns)
    df[other_columns] = False
    return df
def prediction(input):
    categorical_col = input[:2]
    input_df = pd.DataFrame({'average_rainfall': input[2], 'presticides_tonnes': input[3], 'avg_temp': input[4]}, index=[0])
    input_df1 = df_final.head(1)
    input_df1 = input_df1.iloc[:, 3:]
    true_columns = [f'Country_{categorical_col[0]}', f'Item_{categorical_col[1]}']
    input_df2 = update_columns(input_df1, true_columns)
    final_df = pd.concat([input_df, input_df2], axis=1)
    final_df = final_df.values
    test_input = sc.transform(final_df)
    test_input1 = pf.transform(test_input)
    predict = model.predict(test_input1)
    return (f'The Production of Crop Yields: {predict} hg/ha yield')


           
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('img.png')

if __name__=='__main__':    
    main()


