import pickle
import pandas as pd
import numpy as np
import streamlit as st

model = pickle.load(open("model.pkl", 'rb'))


crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}


html_code = '''
<h1 style="color:blue; text-align:center">Crop Recommendation System</h1>
'''
## Function to predict which crop is best suited for particular region
def CropRecommendation(input_data):
    
    input_data=np.array(input_data).reshape(1,-1)
    recommend=model.predict(input_data)
    # print(recommend)
    print(crop_dict[recommend[0]])
    return crop_dict[recommend[0]]




def main():
    st.markdown(html_code,unsafe_allow_html=True)



    #Required Data 
# Nitrogen, Phosphorous,Potassium,Temperature,Rainfall,Ph
    nitrogen=st.text_input("Enter Nitrogen content in soil ")
    phosphorous=st.text_input("Enter Phosphorous content in soil ")
    potassium=st.text_input("Enter Potassium content in soil ")
    temperature=st.text_input("Enter Temperature in Celsius")
    humidity=st.text_input("Enter relative humidity in %")
    ph=st.text_input("Enter ph value of the soil")
    rainfall=st.text_input("Enter rainfall in mm")
    
    BestCrop=""
    if st.button("Recommend Crop"):

        print(nitrogen)
        if( nitrogen and  phosphorous and  potassium and  temperature and  rainfall and ph and humidity):
            BestCrop=CropRecommendation([int(nitrogen),int(phosphorous),int(potassium),float(temperature),float(humidity),float(ph),float(rainfall)])
            st.success(f"{BestCrop} is best crop to grow.")
        else :
            st.write("Enter Correct Values")
if __name__=='__main__':
    main()
