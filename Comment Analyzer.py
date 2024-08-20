from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

app = Flask(__name__)

# Load the CSV file into a DataFrame
df = pd.read_csv('shuffled_file.csv', encoding='ISO-8859-1')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to predict sentiment using the model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df['cumments'], df['senti'])


def predict_sentiment_model(comment):
    prediction = model.predict([comment])
    return prediction[0]


# Function to predict sentiment using VADER
def predict_sentiment_vader(comment):
    scores = sid.polarity_scores(comment)
    if scores['compound'] >= 0.05:
        return 1  # Positive
    elif scores['compound'] <= -0.05:
        return -1  # Negative
    else:
        return 0  # Neutral


def generate_charts(sentiment_counts):
    # Pie chart
    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.title('Sentiment Distribution')
    plt.tight_layout()

    # Save pie chart to a base64 encoded string
    pie_chart = get_base64_encoded_image(fig1)

    # Bar chart
    fig2, ax2 = plt.subplots()
    ax2.bar(labels, sizes)
    plt.title('Sentiment Counts')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()

    # Save bar chart to a base64 encoded string
    bar_chart = get_base64_encoded_image(fig2)

    return pie_chart, bar_chart


def get_base64_encoded_image(fig):
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)  # Close the figure to release memory
    return img_base64


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        option = request.form['option']
        if option == 'comment':
            return render_template('comment.html')
        elif option == 'file':
            return render_template('upload_file.html')

    return render_template('index.html')


@app.route('/analyze_comment', methods=['POST'])
def analyze_comment():
    user_comment = request.form['comment']
    if not user_comment:
        return render_template('comment.html', prediction=None, comment=user_comment)

    # Split the user's comment into words
    user_words = set(user_comment.split())

    # Split the comments in 'cumments' column into words and create a set
    cumment_words = set(" ".join(df['cumments']).split())

    # Check if there is any intersection of words between user's comment and 'cumments' column
    if user_words.intersection(cumment_words):
        # Use model prediction
        sentiment_model = predict_sentiment_model(user_comment)
        prediction = "Negative" if sentiment_model == -1 else "Neutral" if sentiment_model == 0 else "Positive"

        return render_template('comment.html', prediction=prediction, comment=user_comment)
    else:
        # Use VADER prediction
        sentiment_vader = predict_sentiment_vader(user_comment)
        prediction = "Negative" if sentiment_vader == -1 else "Neutral" if sentiment_vader == 0 else "Positive"

        return render_template('comment.html', prediction=prediction, comment=user_comment)


@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        return render_template('upload_file.html', error='No file part')

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return render_template('upload_file.html', error='No selected file')

    # Read the file and process the comments
    comments = []
    for line in file:
        comments.append(line.decode('utf-8').strip())

    sentiment_counts = {
        'Positive': 0,
        'Negative': 0,
        'Neutral': 0
    }

    for comment in comments:
        sentiment_vader = predict_sentiment_vader(comment)
        if sentiment_vader == 1:
            sentiment_counts['Positive'] += 1
        elif sentiment_vader == -1:
            sentiment_counts['Negative'] += 1
        else:
            sentiment_counts['Neutral'] += 1

    pie_chart, bar_chart = generate_charts(sentiment_counts)

    return render_template('result.html', comments=comments, pie_chart=pie_chart, bar_chart=bar_chart,
                           sentiment_counts=sentiment_counts)


if __name__ == '_main_':
    app.run(debug=True)w
