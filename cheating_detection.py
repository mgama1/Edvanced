import numpy as np
from typing import List, Dict, Tuple
from scipy.special import comb
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist 
from sklearn.decomposition import PCA
class CheatingDetection:
    def __init__(self):
        pass
    

    def prob_matching_wrong_answers(k, m, n, c, accuracy,alpha=.01):
        '''
        calculates probability that at least there is one pair with k identical wrong answers 
        '''
        p_0= round(((1-accuracy)**2)/(c-1),5)
        # Calculate the binomial probability
        binomial_prob = comb(n, k) * (p_0**k) * ((1-p_0)**(n-k))
        # Calculate the final probability
        p_value = 1 - (1 - binomial_prob)**comb(m, 2)
        if p_value <alpha:
            significance='statistically significant ✅'
        else:
            significance='statistically insignificant ❌'
        return p_value,significance


    def count_same_wrong_answers_vectorized(
        model: np.ndarray, 
        students: List[np.ndarray], 
        top_n: int = 5
    ) -> List[Tuple[Tuple[int, int], int]]:
        """
        Vectorized implementation to count same wrong answers between all student pairs.
        
        Args:
            model: Correct answer matrix (num_questions × num_choices)
            students: List of student answer matrices
            top_n: Number of top pairs to return
            
        Returns:
            List of tuples containing ((student_i, student_j), count) sorted by count in descending order
        """
        # Convert model and student answers to choice indices (num_students × num_questions)
        correct_choices = np.argmax(model, axis=1)  # (num_questions,)
        student_choices = np.array([np.argmax(s, axis=1) for s in students])  # (num_students × num_questions)
        
        # Create mask of wrong answers for each student (num_students × num_questions)
        wrong_mask = student_choices != correct_choices
        
        # Prepare for pairwise comparisons
        n_students = len(students)
        indices = np.triu_indices(n_students, k=1)  # Upper triangle indices excluding diagonal
        
        # Vectorized computation of same wrong answers
        s1, s2 = indices
        both_wrong = wrong_mask[s1] & wrong_mask[s2]  # Where both got it wrong
        same_wrong = (student_choices[s1] == student_choices[s2]) & both_wrong
        
        # Count same wrong answers per pair
        counts = np.sum(same_wrong, axis=1)
        
        # Create result pairs with counts
        pairs = list(zip(zip(s1, s2), counts))
        
        # Sort by count descending and return top_n
        top_pairs = sorted(pairs, key=lambda x: -x[1])[:top_n]
        
        return top_pairs

    def get_clusters(students):
        
        X = np.array([student.flatten() for student in students])
        clustering = DBSCAN(eps=.1, min_samples=2,metric='hamming').fit(X)
        #clustering = HDBSCAN(min_cluster_size=2,metric='hamming').fit(X)
        labels = clustering.labels_
        
        # Apply PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        
        centroid = np.median(X_pca, axis=0).reshape(1, -1)
        distances = cdist(X_pca, centroid, metric='euclidean').flatten()
        normalized_distances = (distances - np.mean(distances)) / np.std(distances)
        
        std_dev = np.std(normalized_distances)
        distance_threshold = np.mean(normalized_distances) + 2 * std_dev
        outliers_mask = distances > distance_threshold
        labels=np.where(outliers_mask==True,labels,-1)
        return labels