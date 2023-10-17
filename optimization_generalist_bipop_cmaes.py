import math
import numpy as np
import os
from cmaes import CMA
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


def evaluate(env, ind):
    # https://www.sfu.ca/~ssurjano/ackley.html
    _, p, e, _ = env.play(ind)
    return (p - e,)


def main():
    bounds = np.array([[-1, 1] for _ in range(265)])

    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    mean = lower_bounds + (np.random.rand(265) * (upper_bounds - lower_bounds))
    sigma = 1 * 2 / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    n_restarts = 0  # A small restart doesn't count in the n_restarts
    small_n_eval, large_n_eval = 0, 0
    popsize0 = optimizer.population_size
    inc_popsize = 2

    # Initial run is with "normal" population size; it is
    # the large population before first doubling, but its
    # budget accounting is the same as in case of small
    # population.
    poptype = "normal"

    for generation in range(100000000):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = - evaluate(env=env, ind=x)[0]
            solutions.append((x, value))
            print(f"#{generation} {value}")
            if value <= - 68:
                np.savetxt('cmaes_bipop/weights_' + str(generation) + '_' + str(value) + '.txt', x)
        optimizer.tell(solutions)

        if optimizer.should_stop():
            n_eval = optimizer.population_size * optimizer.generation
            if poptype == "small":
                small_n_eval += n_eval
            else:  # poptype == "large"
                large_n_eval += n_eval

            if small_n_eval < large_n_eval:
                poptype = "small"
                popsize_multiplier = inc_popsize ** n_restarts
                popsize = math.floor(
                    popsize0 * popsize_multiplier ** (np.random.uniform() ** 2)
                )
            else:
                poptype = "large"
                n_restarts += 1
                popsize = popsize0 * (inc_popsize ** n_restarts)

            mean = lower_bounds + (np.random.rand(265) * (upper_bounds - lower_bounds))
            optimizer = CMA(
                mean=mean,
                sigma=sigma,
                bounds=bounds,
                population_size=popsize,
            )
            print("Restart CMA-ES with popsize={} ({})".format(popsize, poptype))


if __name__ == "__main__":

    main()
