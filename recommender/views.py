from django.shortcuts import render
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from rapidfuzz import process
from django.http import JsonResponse

# Global variables for caching
DATA = None
TFIDF_DESC = None
TFIDF_GENRE = None

# Load cleaned dataset from SQLite and preprocess
def load_data():
    global DATA, TFIDF_DESC, TFIDF_GENRE

    if DATA is None:
        sqlite_db_path = 'cleaned_books.db'
        conn = sqlite3.connect(sqlite_db_path)
        data = pd.read_sql_query("SELECT * FROM books", conn)
        conn.close()

        # Preprocess text data
        data['desc'] = data['desc'].fillna('').str.strip()
        data['genre'] = data['genre'].fillna('').str.strip()
        data['title'] = data['title'].fillna('').str.strip()
        data['author'] = data['author'].fillna('').str.strip()

        # Normalize rating and totalratings
        scaler = MinMaxScaler()
        data[['rating', 'totalratings']] = scaler.fit_transform(data[['rating', 'totalratings']])

        # Cache the data and precomputed matrices
        DATA = data
        TFIDF_DESC = TfidfVectorizer(stop_words='english').fit_transform(DATA['desc'])
        TFIDF_GENRE = TfidfVectorizer(stop_words='english').fit_transform(DATA['genre'])
    
    return DATA

# Autocomplete view to provide book title suggestions
def autocomplete(request):
    query = request.GET.get('query', '').strip()
    if query:
        data = load_data()
        
        # Use rapidfuzz for fuzzy matching on full dataset
        suggestions = process.extract(query, data['title'].values, limit=5)
        results = [title[0] for title in suggestions if title[1] > 60]
    else:
        results = []
    
    return JsonResponse({'suggestions': results})

# Recommend books
def recommend_books(input_title, data, top_n=10):
    global TFIDF_DESC, TFIDF_GENRE

    if input_title not in data['title'].values:
        similar_titles = process.extract(input_title, data['title'].values, limit=5)
        suggestions = [title[0] for title in similar_titles if title[1] > 60]
        return suggestions  # Return a list of similar titles if the book is not found

    idx = data[data['title'] == input_title].index[0]

    # Calculate cosine similarity for description and genre
    desc_similarity = cosine_similarity(TFIDF_DESC[idx], TFIDF_DESC).flatten()
    genre_similarity = cosine_similarity(TFIDF_GENRE[idx], TFIDF_GENRE).flatten()

    # Calculate author similarity (same author or not)
    author_similarity = (data['author'] == data.loc[idx, 'author']).astype(float)

    # Popularity score (combining rating and totalratings)
    popularity_score = 0.15 * data['rating'].values + 0.15 * data['totalratings'].values

    # Combine all similarity scores into one final score
    combined_similarity = (
        0.4 * desc_similarity + 
        0.3 * genre_similarity + 
        0.2 * author_similarity + 
        popularity_score
    )

    # Add similarity score to the dataframe
    data['similarity'] = combined_similarity

    # Sort by similarity score, excluding the input book itself
    recommendations = data.sort_values('similarity', ascending=False).iloc[1:top_n+1]

    # Return the top recommendations as a list of dictionaries
    return recommendations[['title', 'author', 'img', 'link']].to_dict(orient='records')

# Index view to handle the form and display recommendations
def index(request):
    recommendations = []
    if request.method == 'POST':
        input_title = request.POST.get('book_title').strip()
        data = load_data()
        recommendations = recommend_books(input_title, data)
    
    return render(request, 'recommender/index.html', {'recommendations': recommendations})
