import numpy as np
from cmaes import CMAwM
import os
from evoman.environment import Environment
from demo_controller import player_controller

# disable visuals and thus make experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"
# hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

EXPERIMENT_NAME = "nn_test"
ENEMY_IDX = [1, 2, 3, 4, 5, 6, 7, 8]

env = Environment(
    experiment_name=EXPERIMENT_NAME,
    enemies=ENEMY_IDX,
    multiplemode="yes",
    speed="fastest",
    logs="off",
    savelogs="no",
    player_controller=player_controller(10),
    visuals=False,
)


def ellipsoid_int(env, ind):
    # return (env.play(ind)[0],)
    f, p, e, t = env.play(ind)
    return (p - e,)


def main():
    dim = 265
    bounds = np.array([[-1, 1] for _ in range(265)])
    steps = np.zeros(265)
    optimizer = CMAwM(mean=np.random.normal(0, 1) * np.ones(dim), sigma=2.0, bounds=bounds, steps=steps)
    print(" evals    f(x)")
    print("======  ==========")

    evals = 0
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x_for_eval, x_for_tell = optimizer.ask()
            value = -ellipsoid_int(env, x_for_eval)[0]
            evals += 1
            solutions.append((x_for_tell, value))
            if evals % 20 == 0:
                print(f"{evals:5d}  {value:10.5f}")
            if value <= - 68:
                print("Solution found! ======================================================================")
                print(f"{evals:5d}  {value:10.5f}")
                print("======================================================================================")
                np.savetxt('cmaes_wm/weights_tell_' + str(value) + '.txt', x_for_tell)
                np.savetxt('cmaes_wm/weights_eval_' + str(value) + '.txt', x_for_eval)
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    main()
