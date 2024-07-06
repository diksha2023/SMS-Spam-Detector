from flask import Flask, request, render_template
import sklearn
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)

        if prediction == 1:
            result = 'Spam'
        else:
            result = 'Not Spam'

        return render_template('index.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)