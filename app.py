from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)
encoder = pickle.load(open('encoder.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

    # get features and values from the user
    data = request.form.to_dict()
    str_features = {}
    num_features = {}
    # separate strings for hot encoding and numners to convert to integer
    for k, v in data.items():
        if v.isdigit():
            num_features[k] = [int(v)]
        else:
            str_features[k] = [v]
    # create datafrafes to pass them to encoder and model 
    # (that's how they were trained in jupyter notebook)
    df1 = pd.DataFrame(data=str_features)  
    df2 = pd.DataFrame(data=num_features)

    # encode strings
    enc_arr = encoder.transform(df1) 
    # add encoded values to dataframe
    df2[encoder.get_feature_names_out()] = enc_arr.toarray()
    
    # make predictions on dataframe
    prediction = model.predict(df2)
    # get the output
    output = prediction[0]

    return render_template('predictions.html', prediction= output, data=data)

if __name__ == "__main__":
    app.run(debug=True)