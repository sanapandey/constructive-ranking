from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CoalitionAnalyzer:
    min_score = float('inf')  # Global minimum score across all threads
    max_score = float('-inf')  # Global maximum score across all threads
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the coalition analyzer with a sentence transformer model
        
        Args:
            model_name (str): Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (List[str]): List of text comments
        
        Returns:
            np.ndarray: Array of embeddings
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, is_split_into_words=True)
        return embeddings
    
    def cluster_comments(self, embeddings: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """
        Cluster comments using K-means
        
        Args:
            embeddings (np.ndarray): Array of comment embeddings
            n_clusters (int): Number of clusters to form
        
        Returns:
            np.ndarray: Cluster labels for each comment
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        return cluster_labels
    
    def calculate_coalition_centroids(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """
        Calculate centroid for each coalition
        
        Args:
            embeddings (np.ndarray): Array of comment embeddings
            cluster_labels (np.ndarray): Cluster labels for each comment
        
        Returns:
            np.ndarray: Centroids for each coalition
        """
        unique_clusters = np.unique(cluster_labels)
        centroids = np.array([
            embeddings[cluster_labels == i].mean(axis=0) 
            for i in unique_clusters
        ])
        return centroids

    
    def calculate_comment_score(self, 
                              comment_embedding: np.ndarray, 
                              coalition_centroids: np.ndarray, 
                              comment_coalition: int) -> float:
        """
        Calculate coalition-building score for a single comment
        
        Args:
            comment_embedding (np.ndarray): Embedding of the comment
            coalition_centroids (np.ndarray): Centroids of all coalitions
            comment_coalition (int): Coalition index of the comment
        
        Returns:
            float: Coalition-building score
        """
        # Intra-coalition score
        intra_similarities = cosine_similarity(
            comment_embedding.reshape(1, -1), 
            coalition_centroids[comment_coalition].reshape(1, -1)
        )[0][0]
        intra_score = 1 / (intra_similarities + 1e-10)  # Prevent division by zero

        
        # Inter-coalition score
        other_centroids = np.delete(coalition_centroids, comment_coalition, axis=0)
        if len(other_centroids) > 0:
            inter_similarities = cosine_similarity(
                comment_embedding.reshape(1, -1), 
                other_centroids
            )[0]
            inter_score = -np.max(cosine_similarity(
                comment_embedding.reshape(1, -1), 
                coalition_centroids[comment_coalition].reshape(1, -1)
            )[0]) - np.sum(inter_similarities)
            print(f"Inter Score: {intra_score}") 
        else:
            inter_score = 0
            print("Went into inter else statement.")
        final_score = inter_score/intra_score  # Direct difference instead of division
        print(f"Comment Score: {final_score}")  # Debugging print
        return final_score    
    
    def analyze_thread(self, comments: List[str], n_clusters: int = 3) -> Dict[str, Any]:
        """
        Perform full coalition analysis on a thread
        
        Args:
            comments (List[str]): List of comment texts
            n_clusters (int): Number of coalitions to identify
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Check if we have enough comments to analyze
        if len(comments) < n_clusters:
            # Adjust n_clusters if we don't have enough comments
            n_clusters = max(2, len(comments) - 1) if len(comments) > 1 else 1

        # Generate embeddings
        embeddings = self.get_embeddings(comments)
        
        # If we have only one comment or n_clusters is 1, we can't do meaningful clustering
        if len(comments) < 10 or n_clusters <= 1:
            return {
                'cluster_labels': np.zeros(len(comments), dtype=int),
                'comment_scores': np.zeros(len(comments)),
                'overall_coalition_diversity': float('NaN') # return nan if comments are too few or clusters are too few 
            }
        
        # Measure overall similarity of all comments
        all_pairwise_similarities = cosine_similarity(embeddings)
        avg_pairwise_similarity = np.mean(all_pairwise_similarities)

        # If similarity is very high (above 0.8), force a low diversity score
        if avg_pairwise_similarity > 0.8:  
            return {
                'cluster_labels': np.zeros(len(comments), dtype=int),
                'comment_scores': np.zeros(len(comments)),
                'overall_coalition_diversity': 0.0  # Explicitly set diversity to 0
            }
        
        # Cluster comments
        cluster_labels = self.cluster_comments(embeddings, n_clusters)
        
        # Calculate coalition centroids
        coalition_centroids = self.calculate_coalition_centroids(embeddings, cluster_labels)
        
        # Calculate scores for each comment
        comment_scores = [
            self.calculate_comment_score(emb, coalition_centroids, cluster)
            for emb, cluster in zip(embeddings, cluster_labels)
        ]


        diversity_score = np.std(comment_scores)

        if np.std(comment_scores) < 0.05:
            normalized_score = 0.0  # Very low diversity
        else:
            if not np.isnan(diversity_score):  # Avoid updating if diversity_score is NaN
                CoalitionAnalyzer.min_score = min(CoalitionAnalyzer.min_score, diversity_score)
                CoalitionAnalyzer.max_score = max(CoalitionAnalyzer.max_score, diversity_score)

        # Normalize using Min-Max Scaling (only if min and max are different)
            if CoalitionAnalyzer.max_score > CoalitionAnalyzer.min_score:
                normalized_score = (diversity_score - CoalitionAnalyzer.min_score) / \
                       (CoalitionAnalyzer.max_score - CoalitionAnalyzer.min_score)
            else:
                normalized_score = 0.0  # Avoid division by zero
        print(f"Overall Diversity Score: {normalized_score}")  # Debugging print
        
        return {
            'cluster_labels': cluster_labels,
            'comment_scores': comment_scores,
            'overall_coalition_diversity': 1 - normalized_score #try 1-normalized_score
        }


def extract_comments_from_forest(comment_forest: List[Dict]) -> List[str]:
    """
    Extract comment texts from a comment forest structure
    
    Args:
        comment_forest (List[Dict]): A nested comment structure where each comment
                                     has 'content' and optionally 'replies'
    
    Returns:
        List[str]: Flattened list of comment texts
    """
    comment_texts = []
    
    def extract_recursive(comments):
        for comment in comments['comments']:
            if 'body' in comment:
                comment_texts.append(comment['body'])
            if 'replies' in comment and comment['replies']:
                extract_recursive(comment['replies'])
    
    extract_recursive(comment_forest)
    return comment_texts


def get_coalition_score(comment_forest, n_clusters: int = 3) -> float:
    """
    Calculate the coalition score for a comment forest JSON
    
    Args:
        comment_forest (List[Dict]): A nested comment structure where each comment
                                     has 'content' and optionally 'replies'
        n_clusters (int): Number of coalitions to identify
    
    Returns:
        float: Coalition score indicating viewpoint diversity (higher is more diverse)
               Returns 0.0 if analysis cannot be performed
    """
     # Extract comment texts from the forest structure
    comment_texts = extract_comments_from_forest(comment_forest)
        
    # Check if we have enough comments to analyze
    if len(comment_texts) < 10:  # Threshold for minimum comments
        return float("NaN")
        
    # Initialize analyzer and run analysis
    analyzer = CoalitionAnalyzer()
    results = analyzer.analyze_thread(comment_texts, n_clusters=min(n_clusters, len(comment_texts)))
        
    # Return the overall coalition diversity as the score
    return float(results['overall_coalition_diversity'])
    
