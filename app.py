from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open('wine.pkl','rb'))

app=Flask('Wine Quality')

@app.route('/')
def ping():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    return render_template('home.html',prediction_text='Wine quality will be $ {}'.format(prediction))


if __name__=='__main__':
    app.run(debug=True)
