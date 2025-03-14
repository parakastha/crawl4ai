"""
Clustering strategies for content analysis.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logger = logging.getLogger(__name__)

class BaseClusteringStrategy:
    """Base class for clustering strategies."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the clustering strategy.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
    
    def cluster(self, texts: List[str]) -> Tuple[List[int], Any]:
        """Cluster the provided texts.
        
        Args:
            texts: List of text to cluster
            
        Returns:
            Tuple of (cluster_labels, cluster_model)
        """
        raise NotImplementedError("Subclasses must implement cluster method")
    
    def get_cluster_summary(self, texts: List[str], labels: List[int]) -> Dict[int, List[str]]:
        """Get a summary of clusters.
        
        Args:
            texts: List of texts
            labels: Cluster labels for each text
            
        Returns:
            Dictionary mapping cluster IDs to lists of texts
        """
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[i])
        return clusters
    
    def get_cluster_keywords(self, model: Any, cluster_id: int, top_n: int = 5) -> List[str]:
        """Get top keywords for a cluster.
        
        Args:
            model: Clustering model or feature extractor
            cluster_id: ID of the cluster
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        raise NotImplementedError("Subclasses must implement get_cluster_keywords method")


class CosineStrategy(BaseClusteringStrategy):
    """Clustering strategy based on cosine similarity."""
    
    def __init__(
        self,
        semantic_filter: Optional[str] = None,
        word_count_threshold: int = 10,
        sim_threshold: float = 0.3,
        max_dist: float = 0.2,
        linkage_method: str = 'ward',
        top_k: int = 3,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        verbose: bool = False
    ):
        """Initialize the cosine similarity clustering strategy.
        
        Args:
            semantic_filter: Topic/keyword filter
            word_count_threshold: Minimum words per cluster
            sim_threshold: Similarity threshold
            max_dist: Maximum cluster distance
            linkage_method: Clustering method ('ward', 'complete', 'average', 'single')
            top_k: Top clusters to return
            model_name: Name of the sentence-transformers model to use
            verbose: Enable verbose logging
        """
        super().__init__(verbose=verbose)
        self.semantic_filter = semantic_filter
        self.word_count_threshold = word_count_threshold
        self.sim_threshold = sim_threshold
        self.max_dist = max_dist
        self.linkage_method = linkage_method
        self.top_k = top_k
        self.model_name = model_name
        
        # Load the model
        try:
            self.model = SentenceTransformer(model_name)
            if verbose:
                logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.model = None
    
    def _filter_by_word_count(self, texts: List[str]) -> List[int]:
        """Filter texts by word count.
        
        Args:
            texts: List of texts to filter
            
        Returns:
            Indices of texts that meet the word count threshold
        """
        valid_indices = []
        for i, text in enumerate(texts):
            if len(text.split()) >= self.word_count_threshold:
                valid_indices.append(i)
        
        if self.verbose:
            logger.info(f"Filtered {len(texts) - len(valid_indices)} texts below word count threshold")
        
        return valid_indices
    
    def _filter_by_semantic_relevance(self, embeddings: np.ndarray) -> List[int]:
        """Filter texts by semantic relevance to the filter query.
        
        Args:
            embeddings: Text embeddings
            
        Returns:
            Indices of semantically relevant texts
        """
        if not self.semantic_filter or not self.model:
            # No filtering if no semantic filter or model
            return list(range(len(embeddings)))
        
        # Get embedding for the filter query
        filter_embedding = self.model.encode([self.semantic_filter])[0]
        
        # Calculate similarity with all text embeddings
        similarities = cosine_similarity([filter_embedding], embeddings)[0]
        
        # Filter by threshold
        valid_indices = [i for i, sim in enumerate(similarities) if sim >= self.sim_threshold]
        
        if self.verbose:
            logger.info(f"Filtered {len(embeddings) - len(valid_indices)} texts below semantic similarity threshold")
        
        return valid_indices
    
    def cluster(self, texts: List[str]) -> Tuple[List[int], AgglomerativeClustering]:
        """Cluster texts based on cosine similarity.
        
        Args:
            texts: List of texts to cluster
            
        Returns:
            Tuple of (cluster_labels, clustering_model)
        """
        if not texts:
            logger.warning("No texts provided for clustering")
            return [], None
        
        if not self.model:
            logger.error("No embedding model available for clustering")
            return [0] * len(texts), None
        
        # Filter texts by word count
        valid_indices = self._filter_by_word_count(texts)
        if not valid_indices:
            logger.warning("No texts meet the word count threshold")
            return [0] * len(texts), None
        
        filtered_texts = [texts[i] for i in valid_indices]
        
        # Generate embeddings
        embeddings = self.model.encode(filtered_texts)
        
        # Filter by semantic relevance
        semantic_indices = self._filter_by_semantic_relevance(embeddings)
        if not semantic_indices:
            logger.warning("No texts meet the semantic relevance threshold")
            return [0] * len(texts), None
        
        # Map back to original indices
        final_indices = [valid_indices[i] for i in semantic_indices]
        final_embeddings = embeddings[semantic_indices]
        
        # Determine number of clusters
        n_clusters = min(self.top_k, len(final_embeddings))
        if n_clusters <= 1:
            # If only one cluster, assign all to cluster 0
            cluster_labels = [0] * len(texts)
            for i, idx in enumerate(final_indices):
                cluster_labels[idx] = 0
            return cluster_labels, None
        
        # Perform clustering
        clustering_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='cosine',
            linkage=self.linkage_method,
            distance_threshold=self.max_dist if n_clusters is None else None
        )
        
        # Fit the model
        sub_labels = clustering_model.fit_predict(final_embeddings)
        
        # Map labels back to original texts
        cluster_labels = [-1] * len(texts)  # -1 for filtered out texts
        for i, idx in enumerate(final_indices):
            cluster_labels[idx] = sub_labels[i]
        
        if self.verbose:
            logger.info(f"Created {n_clusters} clusters from {len(final_indices)} texts")
        
        return cluster_labels, clustering_model
    
    def get_cluster_keywords(self, model: Any, cluster_id: int, top_n: int = 5) -> List[str]:
        """Get top keywords for a cluster using CountVectorizer.
        
        Args:
            model: Not used for this implementation
            cluster_id: ID of the cluster
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        # This implementation gets keywords based on the most recent clustering
        # It requires that get_cluster_summary has been called to organize texts by cluster
        if not hasattr(self, '_cluster_texts'):
            logger.warning("No cluster texts available. Call get_cluster_summary first.")
            return []
        
        if cluster_id not in self._cluster_texts:
            logger.warning(f"Cluster ID {cluster_id} not found")
            return []
        
        texts = self._cluster_texts.get(cluster_id, [])
        if not texts:
            return []
        
        # Join all texts in the cluster
        combined_text = " ".join(texts)
        
        # Extract keywords using CountVectorizer
        vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
        try:
            X = vectorizer.fit_transform([combined_text])
            feature_names = vectorizer.get_feature_names_out()
            return list(feature_names)
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def get_cluster_summary(self, texts: List[str], labels: List[int]) -> Dict[int, List[str]]:
        """Get a summary of clusters.
        
        Args:
            texts: List of texts
            labels: Cluster labels for each text
            
        Returns:
            Dictionary mapping cluster IDs to lists of texts
        """
        # Call parent method
        clusters = super().get_cluster_summary(texts, labels)
        
        # Store for keyword extraction
        self._cluster_texts = clusters
        
        return clusters


def get_clustering_strategy(strategy_name: str = "kmeans", **kwargs) -> BaseClusteringStrategy:
    """Get a clustering strategy by name.
    
    Args:
        strategy_name: Name of the strategy ('cosine', 'kmeans', 'hierarchical')
        **kwargs: Additional parameters for the strategy
        
    Returns:
        A clustering strategy instance
    """
    strategy_name = strategy_name.lower()
    
    if strategy_name in ['cosine', 'cosine_similarity']:
        return CosineStrategy(**kwargs)
    elif strategy_name in ['kmeans', 'k-means']:
        # Currently mapped to CosineStrategy since it's the most versatile
        logger.info("KMeans clustering mapped to CosineStrategy")
        return CosineStrategy(**kwargs)
    elif strategy_name in ['hierarchical', 'agglomerative']:
        # Currently mapped to CosineStrategy with ward linkage
        logger.info("Hierarchical clustering mapped to CosineStrategy with ward linkage")
        kwargs['linkage_method'] = 'ward'
        return CosineStrategy(**kwargs)
    else:
        logger.warning(f"Unknown clustering strategy: {strategy_name}, using CosineStrategy")
        return CosineStrategy(**kwargs) 