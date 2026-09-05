from __future__ import annotations

import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, fixed
import numpy as np


def solver(
    I: float, a: float, T: float, dt: float, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt."""
    Nt = int(T / dt)      # no of time intervals
    T = Nt * dt           # adjust T to fit time step dt
    u = np.zeros(Nt + 1)  # array of u[n] values
    t = np.linspace(0, T, Nt + 1)  # time mesh

    u[0] = I  # assign initial condition
    for n in range(Nt):
        u[n + 1] = (1 - (1 - theta) * a * dt) / (1 + theta * dt * a) * u[n]
    return u, t


def u_exact(t: float | np.ndarray, I: float, a: float) -> float | np.ndarray:
    return I * np.exp(-a * t)


def plot_numerical_and_exact(theta: float, I: float, a: float, T: float, dt: float):
    """Compare the numerical and exact solution in a plot."""
    u, t = solver(I, a, T, dt, theta)

    t_e = np.linspace(0, T, 1001)  # fine mesh for u_e
    u_e = u_exact(t_e, I, a)

    plt.plot(
        t,
        u,
        "r--o",  # red dashes w/circles
        t_e,
        u_e,
        "b-",
    )  # blue line for exact sol.
    plt.legend(["numerical", "exact"])
    plt.xlabel("t")
    plt.ylabel("u")
    plt.title(f"theta={theta:g}, dt={dt:g}")
    plt.show()

    # plt.savefig(f"plot_{theta}_{dt:g}.png")

def interact_plot_numerical_and_exact(I: float = 0.1, a: float = 0.8, dt: float = 0.8):
    """Interactively compare the numerical and exact solution in a plot."""
    interact(
        plot_numerical_and_exact,
        theta=FloatSlider(min=0, max=1, step=0.1, value=0.5),
        I=fixed(I),
        a=fixed(a),
        T=FloatSlider(min=0, max=50, step=1, value=5),
        dt=fixed(dt)
    )



def test_solver_three_steps():
    """Compare three steps with known manual computations."""
    theta, a, I, dt = 0.8, 2, 0.1, 0.8
    u_by_hand = np.array([I, 0.0298245614035, 0.00889504462912, 0.00265290804728])

    if Nt is None:
        Nt = 3  # number of time steps
    u, _ = solver(I, a, Nt * dt, dt, theta)

    tol = 1e-12  # tolerance for comparing floats
    diff = np.abs(u - u_by_hand).max()
    success = diff < tol
    assert success

if __name__ == "__main__":
    interact_plot_numerical_and_exact()