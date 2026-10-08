import numpy as np

from denoising_mueller.polarimetry.lu_chipman import from_matrix, polarimetric_maps


def _linear_retarder(delta_deg, theta_deg):
    d, t = np.radians(delta_deg), np.radians(theta_deg)
    c2, s2, cd, sd = np.cos(2 * t), np.sin(2 * t), np.cos(d), np.sin(d)
    return np.array([[1, 0, 0, 0],
                     [0, c2**2 + s2**2 * cd, c2 * s2 * (1 - cd), -s2 * sd],
                     [0, c2 * s2 * (1 - cd), s2**2 + c2**2 * cd, c2 * sd],
                     [0, s2 * sd, -c2 * sd, cd]])


def test_pure_retarder_and_depolarizer():
    R = _linear_retarder(60, 30)
    D = np.diag([1, 0.5, 0.5, 0.5]) @ R            # partial depolarizer after the retarder
    nM = from_matrix(np.stack([R, D])[None])        # (1, 2, 16), lab storage layout
    m = polarimetric_maps(nM)
    assert np.allclose(m["linR"], 60, atol=1e-6)
    assert np.allclose(m["totD"], 0, atol=1e-9)
    assert np.allclose(m["totP"], [0, 0.5], atol=1e-9)
    assert m["valid"].all()
    # azimuth convention follows the lab's compiled code; it must be identical for both pixels
    assert np.isclose(m["azimuth"][0, 0], m["azimuth"][0, 1])


def test_nonphysical_detected():
    M = np.eye(4)
    M[3, 2] = 1.3                                    # |m| > 1, as in the 0000_098 MMs
    m = polarimetric_maps(from_matrix(M[None, None]))
    assert not m["valid"].any()
