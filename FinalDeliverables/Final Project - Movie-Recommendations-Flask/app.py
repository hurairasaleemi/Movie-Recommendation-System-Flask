import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
genres = pd.read_csv('movies.csv', usecols=['movieId', 'genres'])

# Clean the data
ratings.drop(['timestamp'], axis=1, inplace=True)

# Select features to be focused on
df = pd.merge(movies, ratings, on='movieId')
df = df[['userId', 'title', 'rating']]

# Find correlation of the features
corr_matrix = df.pivot_table(index='title', columns='userId', values='rating').corr()

# User Model Development
movies_users = df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
mat_movies_users = csr_matrix(movies_users.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=15)
model_knn.fit(mat_movies_users)

# Enlist the Movies w.r.t the collaborative filtering properties filters
def get_recommendations(movie_name, n_recommendations=10):
    idx = process.extractOne(movie_name, movies['title'])[2]
    distances, indices = model_knn.kneighbors(mat_movies_users[idx], n_neighbors=n_recommendations+1)
    rec_movies = [(movies_users.index[indices.flatten()[i]], 
                   genres.loc[genres['movieId'] == movies.loc[movies['title'] == movies_users.index[indices.flatten()[i]]].iloc[0]['movieId']]['genres'].iloc[0], 
                   movies_users.iloc[indices.flatten()[i]].values.max(), 
                   distances.flatten()[i]) 
                  for i in range(1, len(indices.flatten()))]
    return rec_movies

# Create WebPage where Movies list will be shown according to the filters applied
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        try:
            recommendations = get_recommendations(movie_name)
            return render_template('index.html', recommendations=recommendations, movie_name=movie_name)
        except:
            error_message = 'Movie not found. Please try again.'
            return render_template('index.html', error_message=error_message)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
