from piiw.data.experience_replay import ExperienceReplay
import torch


def generate_random_experience():
    return torch.rand(84,84)*255, torch.rand(16)

def init_experience_replay() -> ExperienceReplay:
    return ExperienceReplay(
        keys=["observations", "target_policy"],
        capacity=10
    )

def test_store_one_and_sample_storage():
    experience_replay = init_experience_replay()
    experience = generate_random_experience()

    experience_replay.append({"observations": experience[0],
                    "target_policy": experience[1]})

    sampled = experience_replay.sample_one()

    assert torch.all(torch.eq(sampled[0], experience[0]))
    assert torch.all(torch.eq(sampled[1], experience[1]))

test_store_one_and_sample_storage()