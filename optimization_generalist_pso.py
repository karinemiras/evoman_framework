import random
import os

import hydra
import numpy as np
import math

from deap import base
from deap import creator
from deap import tools

from demo_controller import player_controller
from evolve.logging import DataVisualizer
from evoman.environment import Environment

# disable visuals and thus make experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"
# hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

EXPERIMENT_NAME = "optimization_generalist_pso"
enemies = [2, 4, 7, 8]
#enemies = [1, 5, 6]

env = Environment(
    experiment_name=EXPERIMENT_NAME,
    enemies=enemies,
    multiplemode="yes",
    speed="fastest",
    logs="off",
    savelogs="no",
    player_controller=player_controller(10),
    visuals=False,
)

env_gain = Environment(
    experiment_name=EXPERIMENT_NAME,
    enemies=[1, 2, 3, 4, 5, 6, 7, 8],
    multiplemode="yes",
    speed="fastest",
    logs="off",
    savelogs="no",
    player_controller=player_controller(10),
    visuals=False,
)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    if not os.path.exists(EXPERIMENT_NAME):
        os.makedirs(EXPERIMENT_NAME)

    # create a toolbox
    toolbox = prepare_toolbox(config)

    # create a data gatherer object
    logger = DataVisualizer(EXPERIMENT_NAME)

    for i in range(config.train.num_runs):
        print(f"=====RUN {i + 1}/{config.train.num_runs}=====")
        new_seed = 2137 + i * 10
        best_ind = train_loop_pso(toolbox, config, logger, new_seed, enemies)
        eval_gain(best_ind, logger, i, enemies)
        np.savetxt(EXPERIMENT_NAME + '/best_' + str(i) + '_' + str(enemies) + '.txt', best_ind)

    # Result by which optuna will choose which solution performs best
    eval_result = best_ind.fitness.values[0]
    return eval_result


def prepare_toolbox(config):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=list,
                   smin=None, smax=None, best=None)

    # create the toolbox
    toolbox = base.Toolbox()
    # attribute generator
    toolbox.register("particle", generate, size=265, pmin=-1, pmax=1, smin=config.pso.smin, smax=config.pso.smax)
    # register the population as a list of particles
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    # register the update rules
    toolbox.register("update", updateParticle, phi1=config.pso.phi1, phi2=config.pso.phi2)
    # register the evaluation function
    toolbox.register("evaluate", eval_fitness)
    # ----------
    return toolbox


def eval_fitness(individual):
    return (env.play(pcont=individual)[0],)


def eval_gain(individual, logger, winner_num, enemies):
    num_runs_for_gain = 5
    for _ in range(num_runs_for_gain):
        _, p, e, _ = env_gain.play(pcont=individual)
        gain = p - e
        logger.gather_box(winner_num, gain, enemies)


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(np.random.uniform(pmin, pmax, size))
    part.speed = np.random.uniform(smin, smax, size)
    part.smin = smin
    part.smax = smax
    return part


def train_loop_pso(toolbox, config, logger, seed, enemies):
    pop = toolbox.population(n=config.pso.pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("std", np.std)

    random.seed(seed)
    np.random.seed(seed)

    logbook = tools.Logbook()
    logbook.header = ["gen"] + stats.fields

    GEN = config.pso.num_gens
    best = None

    for part in pop:
        part.fitness.values = toolbox.evaluate(individual=np.array(part))

    # Inertia weight starts at 1 and linearly decreases to 0.1
    w = 1
    w_dec = 1/ GEN
    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(np.array(part))
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best, w)
        w -= w_dec

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, **stats.compile(pop))
        print(logbook.stream)
        logger.gather_line([ind.fitness.values[0] for ind in pop], g, enemies)

    return best


def updateParticle(part, best, w, phi1, phi2):
    u1 = np.random.uniform(0, phi1, len(part))
    u2 = np.random.uniform(0, phi2, len(part))
    v_u1 = u1 * (part.best - part)
    v_u2 = u2 * (best - part)
    part.speed = w * part.speed + (v_u1 + v_u2)
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part += part.speed


if __name__ == "__main__":
    main()
    os.system('afplay /System/Library/Sounds/Sosumi.aiff')

