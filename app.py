import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

movies=pd.read_csv('dataset/processed_movies.csv')

# Prepare for similarity search
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['details']).toarray()
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend_and_plot(movie_name):
    if movie_name not in movies['title'].values:
        print("Movie not found in dataset.")
        return

    idx = movies[movies['title'] == movie_name].index[0]
    distances = list(enumerate(similarity[idx]))
    top = sorted(distances, key=lambda x: x[1], reverse=True)[1:11]

    titles = [movies.iloc[i[0]].title for i in top]
    scores = [round(i[1]*100, 2) for i in top]

    print(f"\nTop 10 movies similar to '{movie_name}':\n")
    for i, (t, s) in enumerate(zip(titles, scores), 1):
        print(f"{i}. {t} ({s}%)")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=titles, palette='inferno')
    plt.xlabel("Similarity Score (%)")
    plt.title(f"Top 10 Similar Movies to '{movie_name}'")
    xmax = 100 if max(scores) >= 60 else min(100, round(max(scores) + 5))
    plt.tight_layout()
    plt.show()

user_movie = input("Enter a movie name: ").strip().title()
recommend_and_plot(user_movie)
