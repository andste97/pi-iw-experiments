from torch.utils.data import IterableDataset, Dataset
from data.experience_replay import ExperienceReplay


# Original Source for this: https://towardsdatascience.com/en-lightning-reinforcement-learning-a155c217c3de
class DummyDataset(Dataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, item):
        return {}
