import pandas as pd
import numpy as np
import ast
# Load datasets
movies = pd.read_csv('dataset/movies.csv')
credits = pd.read_csv('dataset/credits.csv')
movies = movies.merge(credits, on=['movie_id', 'title'])

def extract_names(data, key='name', limit=None):
    try:
        items = ast.literal_eval(data)
        names = [item[key] for item in items]
        return names[:limit] if limit else names
    except:
        return []

def get_director(crew_data):
    try:
        crew = ast.literal_eval(crew_data)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name', '')
    except:
        return ''
    return ''

movies['cast'] = movies['cast'].apply(lambda x: extract_names(x, limit=5))
movies['genres'] = movies['genres'].apply(lambda x: extract_names(x))
movies['keywords'] = movies['keywords'].apply(lambda x: extract_names(x))
movies['director'] = movies['crew'].apply(get_director)

for feature in ['cast', 'genres', 'keywords']:
    movies[feature] = movies[feature].apply(lambda x: [i.replace(" ", "") for i in x])

movies['director'] = movies['director'].apply(lambda x: x.replace(" ", ""))

# Combine all features into one string
movies['details'] = movies['cast'].apply(lambda x: " ".join(x)) + ' ' + \
                    movies['genres'].apply(lambda x: " ".join(x)) + ' ' + \
                    movies['keywords'].apply(lambda x: " ".join(x)) + ' ' + \
                    movies['director'].str.lower()
movies.to_csv('dataset/processed_movies.csv', index=False)