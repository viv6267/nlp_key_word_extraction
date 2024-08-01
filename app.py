from flask import Flask,render_template,request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


app=Flask(__name__,template_folder='template')

#Load pickel file
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_name = pickle.load(f)
with open('tfidf_transformer.pkl', 'rb') as f:
    tfidf_transformer = pickle.load(f)


# cleaning words
reserved_stop_words = set(stopwords.words('english')) # set >> to get the unique vals
extra_stop_words=['one','two','three','four','five','six' "seven",
                  "eight","nine",'ten','using','sample','fig','figure','image','using']

# Add extra stop words to the reserved stop words

stop_words=reserved_stop_words.update(extra_stop_words)

# Preprocessing Text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def processing_text(text):

    # Lower case
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub('[^a-zA-Z]+'," ", text)
    
    # Convert to list from string
    #text = text.split()
    text=nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    
    # Remove words less than three letters
    text = [word for word in text if len(word) >= 3]
    
    # Stemming
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    
    return " ".join(text)

def get_keywords(corpus,topN):
    
    doc_word_count=tfidf_transformer.transform(cv.transform([corpus]))
    create_coordinates=doc_word_count.tocoo()
    tuple=zip(create_coordinates.col,create_coordinates.data)
    sorted_items=sorted(tuple,key=lambda x:(x[1],x[0]),reverse=True)

    sorted_items=sorted_items[:topN]

    feature_val=[]
    score_val=[]
    for idx, score in sorted_items:
        feature_val.append(cv.get_feature_names_out()[idx])
        score_val.append(round(score,3))

    result={}

    for i in range(len(feature_val)):
        result[feature_val[i]]=score_val[i]

    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    documents =request.files['file']
    if documents ==' ':
        return render_template('index.html', error='No document selected')
    
    if documents:
        text = documents.read().decode('utf-8', errors='ignore')
        processed_text=processing_text(text)
        keywords=get_keywords(processed_text,topN=15)
        return render_template('keywords.html', keywords=keywords)
    return render_template('index.html')

@app.route('/search_keywords',methods=['POST'])
def search_keywords():
    search_keyword = request.form['search']
    if search_keyword:
        keywords=[]
        for keyword in cv.get_feature_names_out():
            if search_keyword.lower() in keyword.lower():
                keywords.append(keyword)
                if len(keywords)==15:
                    break
        return render_template('keywordslist.html', keywords=keywords)
    return render_template('index.html')

if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8080)
    app.run(debug=True)
