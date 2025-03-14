"""
Clustering strategies for organizing crawled content.
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

class BaseClusteringStrategy:
    """Base class for all clustering strategies."""
    
    def __init__(self, n_clusters: int = 5):
        """Initialize the clustering strategy.
        
        Args:
            n_clusters: Number of clusters to create
        """
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for clustering.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Convert texts to feature vectors.
        
        Args:
            texts: List of text documents
            
        Returns:
            Feature vectors
        """
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize texts
        try:
            vectors = self.vectorizer.fit_transform(preprocessed_texts)
            return vectors
        except Exception as e:
            logger.error(f"Error vectorizing texts: {e}")
            # Return a zero matrix as fallback
            return np.zeros((len(texts), 1))
    
    def cluster(self, texts: List[str]) -> Tuple[List[int], Any]:
        """Cluster texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Tuple of (cluster_labels, clustering_model)
        """
        raise NotImplementedError("Subclasses must implement the cluster method")
    
    def get_cluster_keywords(self, cluster_model: Any, cluster_idx: int, top_n: int = 5) -> List[str]:
        """Get top keywords for a cluster.
        
        Args:
            cluster_model: Trained clustering model
            cluster_idx: Cluster index
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        raise NotImplementedError("Subclasses must implement the get_cluster_keywords method")
    
    def get_cluster_summary(self, texts: List[str], labels: List[int]) -> Dict[int, Dict[str, Any]]:
        """Generate summary information for each cluster.
        
        Args:
            texts: List of text documents
            labels: Cluster labels for each document
            
        Returns:
            Dictionary of cluster summaries
        """
        cluster_counts = {}
        cluster_texts = {}
        
        # Group texts by cluster
        for i, label in enumerate(labels):
            if label not in cluster_counts:
                cluster_counts[label] = 0
                cluster_texts[label] = []
            
            cluster_counts[label] += 1
            cluster_texts[label].append(texts[i])
        
        # Create summaries
        summaries = {}
        for label in sorted(cluster_counts.keys()):
            text_samples = cluster_texts[label][:3]  # Take first 3 texts as samples
            
            summaries[label] = {
                "count": cluster_counts[label],
                "samples": text_samples,
                "avg_length": sum(len(t) for t in cluster_texts[label]) / len(cluster_texts[label])
            }
        
        return summaries


class KMeansClusteringStrategy(BaseClusteringStrategy):
    """K-means clustering strategy."""
    
    def cluster(self, texts: List[str]) -> Tuple[List[int], Any]:
        """Cluster texts using K-means.
        
        Args:
            texts: List of text documents
            
        Returns:
            Tuple of (cluster_labels, clustering_model)
        """
        # Handle empty input
        if not texts:
            return [], None
        
        # Handle case with fewer documents than clusters
        n_clusters = min(self.n_clusters, len(texts))
        
        # Vectorize texts
        vectors = self.vectorize(texts)
        
        # Apply dimensionality reduction if needed
        if vectors.shape[1] > 100:
            svd = TruncatedSVD(n_components=100)
            vectors = svd.fit_transform(vectors)
        
        # Cluster vectors
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(vectors)
        
        return labels.tolist(), kmeans
    
    def get_cluster_keywords(self, cluster_model: KMeans, cluster_idx: int, top_n: int = 5) -> List[str]:
        """Get top keywords for a K-means cluster.
        
        Args:
            cluster_model: Trained K-means model
            cluster_idx: Cluster index
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        if not hasattr(self.vectorizer, 'get_feature_names_out'):
            return []
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get cluster center
        center = cluster_model.cluster_centers_[cluster_idx]
        
        # Get indices of features with highest values
        indices = np.argsort(center)[::-1][:top_n]
        
        # Get keywords
        keywords = [feature_names[i] for i in indices]
        
        return keywords


class HierarchicalClusteringStrategy(BaseClusteringStrategy):
    """Hierarchical clustering strategy."""
    
    def __init__(self, n_clusters: int = 5, linkage: str = 'ward'):
        """Initialize the hierarchical clustering strategy.
        
        Args:
            n_clusters: Number of clusters to create
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        """
        super().__init__(n_clusters)
        self.linkage = linkage
    
    def cluster(self, texts: List[str]) -> Tuple[List[int], Tuple[Any, np.ndarray]]:
        """Cluster texts using hierarchical clustering.
        
        Args:
            texts: List of text documents
            
        Returns:
            Tuple of (cluster_labels, (clustering_model, vectors))
        """
        # Handle empty input
        if not texts:
            return [], (None, None)
        
        # Handle case with fewer documents than clusters
        n_clusters = min(self.n_clusters, len(texts))
        
        # Vectorize texts
        vectors = self.vectorize(texts)
        
        # Convert to dense matrix if needed
        if hasattr(vectors, 'toarray'):
            vectors_dense = vectors.toarray()
        else:
            vectors_dense = vectors
        
        # Cluster vectors
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=self.linkage)
        labels = model.fit_predict(vectors_dense)
        
        return labels.tolist(), (model, vectors)
    
    def get_cluster_keywords(self, cluster_data: Tuple[Any, np.ndarray], cluster_idx: int, top_n: int = 5) -> List[str]:
        """Get top keywords for a hierarchical cluster.
        
        Args:
            cluster_data: Tuple of (clustering_model, vectors)
            cluster_idx: Cluster index
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        if not hasattr(self.vectorizer, 'get_feature_names_out'):
            return []
        
        # Unpack cluster data
        model, vectors = cluster_data
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get document indices for this cluster
        cluster_indices = np.where(model.labels_ == cluster_idx)[0]
        
        # Get vectors for this cluster
        if hasattr(vectors, 'toarray'):
            cluster_vectors = vectors[cluster_indices].toarray()
        else:
            cluster_vectors = vectors[cluster_indices]
        
        # Get mean vector for this cluster
        if len(cluster_vectors) > 0:
            mean_vector = np.mean(cluster_vectors, axis=0)
            
            # Get indices of features with highest values
            indices = np.argsort(mean_vector)[::-1][:top_n]
            
            # Get keywords
            keywords = [feature_names[i] for i in indices]
            
            return keywords
        
        return []


def get_clustering_strategy(strategy_name: str, n_clusters: int = 5) -> BaseClusteringStrategy:
    """Get a clustering strategy by name.
    
    Args:
        strategy_name: Name of the strategy ('kmeans' or 'hierarchical')
        n_clusters: Number of clusters to create
        
    Returns:
        Clustering strategy
    """
    strategies = {
        'kmeans': KMeansClusteringStrategy(n_clusters),
        'hierarchical': HierarchicalClusteringStrategy(n_clusters)
    }
    
    return strategies.get(strategy_name.lower(), KMeansClusteringStrategy(n_clusters)) 