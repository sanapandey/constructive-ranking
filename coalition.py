import json
import numpy as np
from typing import List, Dict, Any, Union, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class CoalitionAnalyzer:
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
        return self.model.encode(texts, convert_to_numpy=True)
    
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
        return kmeans.fit_predict(embeddings)
    
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
        return np.array([
            embeddings[cluster_labels == i].mean(axis=0) 
            for i in unique_clusters
        ])
    
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
        else:
            inter_score = 0
            
        return inter_score/intra_score
    
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
        if len(comments) <= 1 or n_clusters <= 1:
            return {
                'cluster_labels': np.zeros(len(comments), dtype=int),
                'comment_scores': np.zeros(len(comments)),
                'overall_coalition_diversity': 0.0
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
        
        return {
            'cluster_labels': cluster_labels,
            'comment_scores': comment_scores,
            'overall_coalition_diversity': np.std(comment_scores)
        }


#THIS IS THE REFACTORED CODE THAT TAKES IN A JSON AND OUTPUTS A NUMERICAL SCORE 
def get_coalition_score(json_path: str, n_clusters: int = 3) -> Optional[float]:
    """
    Extract and analyze a Reddit thread from a JSON file and return a single coalition-building score.
    
    Args:
        json_path (str): Path to the JSON file containing Reddit thread data
        n_clusters (int): Number of coalitions to identify
    
    Returns:
        float: Coalition-building score (higher values indicate more diverse coalitions)
              Returns None if analysis cannot be performed
    """
    try:
        # Load and parse JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract post content
        post_title = data.get('title', '')
        post_body = data.get('selftext', '')
        post_content = f"{post_title} {post_body}".strip()
        
        # Extract comment texts, filtering out empty or deleted comments
        comments = []
        for comment in data.get('comments', []):
            body = comment.get('body', '').strip()
            if body and body not in ['[removed]', '[deleted]']:
                comments.append(body)
        
        # Add the post itself as the first "comment" if it exists
        if post_content:
            comments.insert(0, post_content)
        
        # Check if we have enough comments to analyze
        if len(comments) < 10: # I think we agreed on this number?
            return 0.0  # Return 0 for threads with insufficient comments
        
        # Initialize analyzer and run analysis
        analyzer = CoalitionAnalyzer()
        results = analyzer.analyze_thread(comments, n_clusters=min(n_clusters, len(comments)))
        
        # Return the overall coalition diversity as the score
        return float(results['overall_coalition_diversity'])
    
    except Exception as e:
        print(f"Error analyzing {json_path}: {e}")
        return None


# This is some old code that I've refactored in case we want to do some debugging 

def get_detailed_coalition_analysis(json_path: str, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Perform detailed coalition analysis on a Reddit thread and return full results.
    
    Args:
        json_path (str): Path to the JSON file containing Reddit thread data
        n_clusters (int): Number of coalitions to identify
    
    Returns:
        Dict: Detailed analysis results including per-comment scores and cluster assignments
    """
    try:
        # Load and parse JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract post content
        post_title = data.get('title', '')
        post_body = data.get('selftext', '')
        post_content = f"{post_title} {post_body}".strip()
        
        # Extract comment texts
        comments = []
        comment_ids = []
        for comment in data.get('comments', []):
            body = comment.get('body', '').strip()
            comment_id = comment.get('id', '')
            if body and body not in ['[removed]', '[deleted]']:
                comments.append(body)
                comment_ids.append(comment_id)
        
        # Add the post itself as the first "comment" if it exists
        post_id = data.get('id', '')
        if post_content:
            comments.insert(0, post_content)
            comment_ids.insert(0, f"post_{post_id}")
        
        # Check if we have enough comments to analyze
        if len(comments) < 2:
            return {
                'status': 'insufficient_data',
                'message': 'Not enough valid comments for analysis',
                'coalition_score': 0.0
            }
        
        # Initialize analyzer and run analysis
        analyzer = CoalitionAnalyzer()
        results = analyzer.analyze_thread(comments, n_clusters=min(n_clusters, len(comments)))
        
        # Prepare detailed results
        comment_details = []
        for i, (comment_id, comment_text, cluster, score) in enumerate(zip(
            comment_ids, comments, results['cluster_labels'], results['comment_scores'])):
            
            comment_details.append({
                'id': comment_id,
                'is_post': i == 0 and post_content,
                'text': comment_text,
                'cluster': int(cluster),
                'coalition_score': float(score)
            })
        
        # Prepare final results
        analysis_results = {
            'status': 'success',
            'post_id': post_id,
            'subreddit': data.get('subreddit', ''),
            'title': post_title,
            'comment_count': len(comments),
            'coalition_score': float(results['overall_coalition_diversity']),
            'comment_details': comment_details
        }
        
        return analysis_results
    
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'coalition_score': None
        }

# Still working on this, but the goal is that we will be able to analyze multiple threads with the same function 
def analyze_multiple_threads(json_paths: List[str], n_clusters: int = 3) -> Dict[str, float]:
    """
    Analyze multiple Reddit threads and return their coalition scores.
    
    Args:
        json_paths (List[str]): List of paths to JSON files containing Reddit thread data
        n_clusters (int): Number of coalitions to identify
    
    Returns:
        Dict[str, float]: Dictionary mapping thread IDs to coalition scores
    """
    results = {}
    
    for path in json_paths:
        score = get_coalition_score(path, n_clusters=n_clusters)
        
        # Extract thread ID from the path
        thread_id = path.split('id -- ')[-1].replace('.json', '') if 'id --' in path else path
        
        results[thread_id] = score
    
    return results


# For command-line usage (still need to figure out this integration)
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python coalition_analyzer.py <path_to_reddit_json_file>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    score = get_coalition_score(json_path)
    
    print(f"Coalition building score for {json_path}: {score:.3f}")