import numpy as np


class DepthFeaturesNovelty1Table:
    def __init__(self, n_feats, n_values):
        self.n_feats = n_feats
        self.n_values = n_values
        self.features_indices = np.arange(n_feats)
        self.features_depth = np.full(shape=(n_feats, n_values), fill_value=np.inf)

    def check(self, feature_values, depth):
        return np.any(depth <= self.features_depth[self.features_indices, feature_values])

    def check_and_update(self, feature_values, depth):
        assert len(feature_values) == self.n_feats
        mask = depth < self.features_depth[self.features_indices, feature_values]
        if np.any(mask):
            self.features_depth[self.features_indices[mask], np.asarray(feature_values)[mask]] = depth  # TODO: mask only once, keep features_depth[idx, values] as a variable and reuse it
            return True
        return False
