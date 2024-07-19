from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import numpy as np
#from tensorflow.keras.models import load_model
import pandas as pd
#import logging

app = Flask(__name__)

# Load your trained model (assuming you have it saved as 'model.pkl')
xgb = pickle.load(open('xgb.pkl', 'rb'))
ctt = joblib.load("datact") #loading column transfer object

#sc = pickle.load(open("churnscaler.pkl","rb"))
#model = load_model("churn.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/aboutus')
def about_us():
    return render_template('aboutus.html')

@app.route('/contactus')
def contact_us():
    return render_template('contactus.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

# Setup logging
#logging.basicConfig(level=logging.DEBUG)

@app.route('/predict', methods=['POST'])



def predict():
    data = request.form
    
    
    
    
    #logging.debug(f'Raw input data: {data}')
    #hemoglobin = float(data['hemoglobin'])
    # Extract other features similarly
    # features = [hemoglobin, ...]  # Add other features

    # Make prediction
   # prediction = model.predict([[hemoglobin]])  # Modify this line according to your feature extraction
    #return jsonify({'prediction': prediction[0]})
    
    
    # Convert categorical features to numerical values if necessary

    # Combine features into a single list
    '''
    features =[
        age,gender, place, duration_of_alcohol, quantity_of_alcohol,
        type_of_alcohol,hepatitis_b,hepatitis_c,
        diabetes,obesity, family_history,tch,tg,ldl,hdl, hemoglobin, pcv, rbc, mcv, mch, mchc, total_count,
        polymorphs, lymphocytes,monocytes, eosinophils, basophils, platelet_count, total_bilirubin,direct,
        indirect, total_protein, albumin, globulin, al_phosphatase, sgot_ast,sgpt_alt, 
        usg_abdomen,blood_pressure_sys,blood_pressure_dia]
'''
    '''
    features_df = pd.DataFrame([features],columns=['age', 'gender', 'place', 'duration_of_alcohol', 'quantity_of_alcohol',
    'type_of_alcohol', 'hepatitis_b', 'hepatitis_c', 'diabetes', 'obesity',
    'family_history', 'tch', 'tg', 'ldl', 'hdl', 'hemoglobin', 'pcv', 'rbc',
    'mcv', 'mch', 'mchc', 'total_count', 'polymorphs', 'lymphocytes',
    'monocytes', 'eosinophils', 'basophils', 'platelet_count',
    'total_bilirubin', 'direct', 'indirect', 'total_protein', 'albumin',
    'globulin', 'al_phosphatase', 'sgot_ast', 'sgpt_alt', 'usg_abdomen',
    'blood_pressure_sys', 'blood_pressure_dia'])
'''
    # Apply the same preprocessing pipeline to the input data
    #preprocessed_input = preprocessor.transform(input_df)


    #p =ct.transform(features_df)
    #p = p.astype(np.float32)
    
 
   # prediction = model.predict(p)

     # new things here:-
    '''
    data_o = {
    'Age': data['age'],
    'Gender': data['gender'],
    'Place(location where the patient lives)': data['place'],
    'Duration of alcohol consumption(years)': data['alcohol-duration'],
    'Quantity of alcohol consumption (quarters/day)': data['alcohol-quantity'],
    'Type of alcohol consumed': data['alctype'],
    'Hepatitis B infection': data['hepatitis-b'],
    'Hepatitis C infection': data['hepatitis-c'],
    'Diabetes Result': data['diabetes'],
    'Obesity': data['obesity'],
    'Family history of cirrhosis/ hereditary': data['family-history'],
    'TCH': data['tch'],
    'TG': data['tg'],
    'LDL': data['ldl'],
    'HDL': data['hdl'],
    'Hemoglobin  (g/dl)': data['hemoglobin'],
    'PCV  (%)': data['pcv'],
    'RBC  (million cells/microliter)': data['rbc'],
    'MCV   (femtoliters/cell)': data['mcv'],
    'MCH  (picograms/cell)': data['mch'],
    'MCHC  (grams/deciliter)': data['mchc'],
    'Total Count': data['total-count'],
    'Polymorphs  (%) ': data['polymorphs'],
    'Lymphocytes  (%)': data['lymphocytes'],
    'Monocytes   (%)': data['monocytes'],
    'Eosinophils   (%)': data['eosinophils'],
    'Basophils  (%)': data['basophils'],
    'Platelet Count  (lakhs/mm)': data['platelet-count'],
    'Total Bilirubin    (mg/dl)': data['total-bilirubin'],
    'Direct    (mg/dl)': data['direct'],
    'Indirect     (mg/dl)': data['indirect'],
    'Total Protein     (g/dl)': data['total-protein'],
    'Albumin   (g/dl)': data['albumin'],
    'Globulin  (g/dl)': data['globulin'],
    'AL.Phosphatase      (U/L)': data['al-phosphatase'],
    'SGOT/AST      (U/L)': data['sgot-ast'],
    'SGPT/ALT (U/L)': data['sgpt-alt'],
    'USG Abdomen (diffuse liver or  not)': data['usg-abdomen'],
    'Systolic': data['blood-pressure1'],
    'Diastolic': data['blood-pressure2']
}

'''
   
   
   
    data_o = {
    'Age': int(data['age']),
    'Gender': data['gender'],
    'Place(location where the patient lives)': data['place'],
    'Duration of alcohol consumption(years)': int(data['alcohol-duration']),
    'Quantity of alcohol consumption (quarters/day)': int(data['alcohol-quantity']),
    'Type of alcohol consumed': data['alctype'],
    'Hepatitis B infection': data['hepatitis-b'],
    'Hepatitis C infection': data['hepatitis-c'],
    'Diabetes Result': data['diabetes'],
    'Obesity': data['obesity'],
    'Family history of cirrhosis/ hereditary': data['family-history'],
    'TCH': float(data['tch']),
    'TG': float(data['tg']),
    'LDL': float(data['ldl']),
    'HDL': float(data['hdl']),
    'Hemoglobin  (g/dl)': float(data['hemoglobin']),
    'PCV  (%)': float(data['pcv']),
    'RBC  (million cells/microliter)': float(data['rbc']),
    'MCV   (femtoliters/cell)': float(data['mcv']),
    'MCH  (picograms/cell)': float(data['mch']),
    'MCHC  (grams/deciliter)': float(data['mchc']),
    'Total Count': float(data['total-count']),
    'Polymorphs  (%) ': float(data['polymorphs']),
    'Lymphocytes  (%)': float(data['lymphocytes']),
    'Monocytes   (%)': float(data['monocytes']),
    'Eosinophils   (%)': float(data['eosinophils']),
    'Basophils  (%)': float(data['basophils']),
    'Platelet Count  (lakhs/mm)': float(data['platelet-count']),
    'Total Bilirubin    (mg/dl)': float(data['total-bilirubin']),
    'Direct    (mg/dl)': float(data['direct']),
    'Indirect     (mg/dl)': float(data['indirect']),
    'Total Protein     (g/dl)': float(data['total-protein']),
    'Albumin   (g/dl)': float(data['albumin']),
    'Globulin  (g/dl)': float(data['globulin']),
    'AL.Phosphatase      (U/L)': float(data['al-phosphatase']),
    'SGOT/AST      (U/L)': int(data['sgot-ast']),
    'SGPT/ALT (U/L)': int(data['sgpt-alt']),
    'USG Abdomen (diffuse liver or  not)': data['usg-abdomen'],
    'Systolic': int(data['blood-pressure1']),
    'Diastolic': int(data['blood-pressure2'])
}

# Convert the dictionary into a pandas DataFrame
    input_df = pd.DataFrame([data_o])
    input_df = input_df.fillna(0)
    
    # Apply the same preprocessing pipeline to the input data
    p = ctt.transform(input_df)
    #logging.debug(f'Transformed data: {p}')
    #p_array = p.toarray()

# Handle potential NaN values in the transformed data
    #p = np.nan_to_num(p)
     
    prediction = xgb.predict(p)
     
    # Make prediction
    #prediction = model.predict([features])[0]
    #print(prediction)
    #logging.debug(f'Prediction: {prediction}')
    result = 'Liver Cirrhosis' if prediction[0]== 0 else 'No Liver Cirrhosis'
    
    return render_template('prediction.html', predict=result,prediction_value=prediction,
                           innput=input_df,pp=p)
    #return render_template('prediction.html', predict='Predicted Value: {}'.format(prediction[0]))
    #return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)


