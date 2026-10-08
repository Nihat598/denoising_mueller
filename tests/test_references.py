import numpy as np

from denoising_mueller.data.references import reference_excluding, split_inputs_reference


def test_reference_excludes_input():
    frames = np.arange(4, dtype=float)[:, None, None, None] * np.ones((4, 16, 2, 2))
    assert np.allclose(reference_excluding(frames, 0), (1 + 2 + 3) / 3)


def test_split_is_disjoint():
    frames = np.random.rand(16, 16, 4, 4)
    inputs, ref = split_inputs_reference(frames)
    assert len(inputs) == 8 and np.allclose(ref, frames[8:].mean(0))
