from torch.utils.data import IterableDataset
from piiw.data.experience_replay import ExperienceReplay


# Original Source for this: https://towardsdatascience.com/en-lightning-reinforcement-learning-a155c217c3de
class ExperienceDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ExperienceReplay, sample_size, episode_length) -> None:
        self.buffer = buffer
        self.sample_size = sample_size
        self.episode_length = episode_length

    def __iter__(self):
        # this is to fix the number of epochs pytorch has
        for i in range(int(self.episode_length/self.sample_size)):
            _, batch = self.buffer.sample(self.sample_size)
            for j in range(len(batch["observations"])):
                yield batch["observations"][i], batch["target_policy"][i]