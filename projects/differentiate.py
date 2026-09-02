import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    r = np.zeros(len(u)-1)
    for i in range(0, len(u)-1):
        r[i] = (u[i + 1] - u[i]) / dt
    return r
    raise NotImplementedError

def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    return [x for x in np.diff(u) / dt]
    # np.gradient(u, dt) would also work, but it behaves differently at the boundaries.
    # For example, if u = [0, 1, 2], then np.gradient(u, dt) returns [1/dt, 1/dt, 1/dt], while the above implementation returns [1/dt, 1/dt].
    # return np.gradient(u, dt)
    raise NotImplementedError

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()
    