from flask import Flask, request, render_template
import pickle


import nltk
import nltk
nltk.download('all')
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


tfidf = pickle.load(open('Vectorizer_model.sav', 'rb'))

# load the model
model = pickle.load(open('model.sav', 'rb'))


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        text = re.sub('[^A-za-z0-9]', ' ',  message)
        text = text.lower()
        text = text.split(' ')
        text = [WordNetLemmatizer().lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        sentiment_id = model.predict(tfidf.transform([text]))
        return render_template('result.html', prediction=sentiment_id)

if __name__ == '__main__':
    app.run(debug=True)