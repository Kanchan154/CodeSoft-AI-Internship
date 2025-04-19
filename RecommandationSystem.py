import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class MovieRecommender:
    def __init__(self):
        # Sample movie dataset (in a real app, you'd load this from a file/database)
        self.movies = pd.DataFrame([
            {"title": "The Shawshank Redemption", "genre": "drama", "tags": "prison escape hope friendship"},
            {"title": "The Godfather", "genre": "crime", "tags": "mafia family power betrayal"},
            {"title": "Inception", "genre": "sci-fi", "tags": "dreams reality heist mind"},
            {"title": "Pulp Fiction", "genre": "crime", "tags": "nonlinear violence dark humor"},
            {"title": "The Dark Knight", "genre": "action", "tags": "superhero batman joker chaos"},
            {"title": "Forrest Gump", "genre": "drama", "tags": "life journey love historical"},
            {"title": "The Matrix", "genre": "sci-fi", "tags": "simulation reality chosen one"},
            {"title": "Goodfellas", "genre": "crime", "tags": "mafia rise fall violence"},
            {"title": "Interstellar", "genre": "sci-fi", "tags": "space time love blackhole"},
            {"title": "Fight Club", "genre": "drama", "tags": "alter ego anarchy self-destruction"}
        ])
        
        # Create TF-IDF matrix
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['tags'])
        
        # Compute cosine similarity matrix
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
    
    def recommend_movies(self, movie_title, num_recommendations=5):
        """Get recommendations based on movie title"""
        try:
            # Get the index of the movie
            idx = self.movies[self.movies['title'] == movie_title].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            
            # Sort movies by similarity score
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top recommendations (excluding the input movie itself)
            sim_scores = sim_scores[1:num_recommendations+1]
            
            # Get movie indices
            movie_indices = [i[0] for i in sim_scores]
            
            # Return recommended movies
            return self.movies.iloc[movie_indices]['title'].tolist()
        except:
            return ["Movie not found in our database. Try another title."]

def main():
    recommender = MovieRecommender()
    
    print("Movie Recommendation System")
    print("Available movies:")
    print(recommender.movies['title'].to_string(index=False))
    
    while True:
        print("\nEnter a movie title to get recommendations (or 'quit' to exit):")
        user_input = input("> ")
        
        if user_input.lower() == 'quit':
            break
            
        recommendations = recommender.recommend_movies(user_input)
        
        print("\nRecommendations:")
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie}")

if __name__ == "__main__":
    main()