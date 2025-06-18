import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity


def get_nearest_neighbours(inp_feature, ref_model="MAE", num_neighbor=5):
    """
    Get nearest neighbours of a feature vector from a reference model.

    Args:
        inp_feature (numpy.ndarray): Input feature vector of shape (feature_dim,).
        ref_model (str): Reference model name.
        num_neighbor (int): Number of nearest neighbours to retrieve.

    Returns:
        list: List of nearest neighbour names.
    """
    if ref_model == "MAE":
        ref_df = pd.read_csv("subcell_features/MAE_mean_feat.csv", index_col=0)
    elif ref_model == "ViT":
        ref_df = pd.read_csv("subcell_features/ViT_mean_feat.csv", index_col=0)
    else:
        raise ValueError(f"Unknown reference model: {ref_model}")
    # Assuming `ref_model` is a dictionary containing the reference features
    ref_features = ref_df.values
    similarities = cosine_similarity(inp_feature.reshape(1, -1), ref_features)
    nearest_indices = np.argsort(similarities[0])[::-1][:num_neighbor]
    nearest_neighbours = ref_df.iloc[nearest_indices].index.tolist()
    return nearest_neighbours
