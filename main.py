
from random import choice, random

from numpy import asarray, clip
from numpy.random import rand


# define objective function
def obj(x):
    return 0


# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]

    return mutated_bound


# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial


def differential_evolution():
    # define lower and upper bounds
    bounds = asarray([-5.0, 5.0])

    pop_size = 10
    iter = 100
    F = 0.5

    # initialise population of candidate solutions randomly within the specified bounds
    pop = bounds[:, 0] + (random(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))

    obj_all = [obj(ind) for ind in pop]

    # run iterations of the algorithm
    for i in range(iter):
    # iterate over all candidate solutions
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]

            # perform mutation
            mutated = mutation([a, b, c], F)
            mutated_bound = check_bounds([mutated], bounds)

if __name__ == "__main__":
    pass
