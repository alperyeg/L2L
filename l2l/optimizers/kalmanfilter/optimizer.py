import logging
import numpy as np

from collections import namedtuple
from l2l import get_grouped_dict
from l2l.utils.tools import cartesian_product
from .updateEnKF import update_enknf
from l2l import dict_to_list
from l2l.optimizers.optimizer import Optimizer

logger = logging.getLogger("optimizers.kalmanfilter")

EnsembleKalmanFilterParameters = namedtuple(
    'EnsembleKalmanFilter', ['noise', 'gamma', 'tol',
                             'maxit', 'stopping_crit', 'n_iteration',
                             'pop_size', 'shuffle', 'n_batches', 'online',
                             'epsilon', 'decay_rate', 'seed']
)


EnsembleKalmanFilterParameters.__doc__ = """
:param noise: float, Noise level
:param gamma
:param tol: float, Tolerance for convergence to stop the iteration inside the 
            Kalman Filter
:param maxit: int, Epochs to run inside the Kalman Filter
:param stopping_crit: string, Name of the stopping criterion
:param n_iteration: int, Number of iterations to perform
:param pop_size: int, Minimal number of individuals per simulation.
:param shuffle: bool, **True** if the data set should be shuffled. 
                Default: True
:param n_batches: int, Number of mini-batches to use in the Kalman Filter
:param online: bool, Indicates if only one data point will used, 
               Default: False
:param epsilon: float, A value which is used when sampling from the best individual. 
                The value is multiplied to the covariance matrix as follows:
                :math:`\\epsilon * I` where I is the identity matrix with the 
                same size as the covariance matrix. The value is 
                exponentially decaying and should be in [0,1] and used in 
                combination with `decay_rate`. 
:param decay_rate: float, Decay rate for the sampling. 
                For the exponential decay as follows:
                .. math::
                    \\epsilon = \\epsilon_0 e^{-decay_rate * epoch}
                    
                Where :math:`\\epsilon` is the value from `epsilon`. The
:param seed: The random seed used to sample and fit the distribution. 
             Uses a random generator seeded with this seed.
"""


class EnsembleKalmanFilter(Optimizer):
    """
    Class for an Ensemble Kalman Filter optimizer
    """
    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 optimizee_create_new_individuals,
                 parameters,
                 optimizee_bounding_func=None):
        super().__init__(traj,
                         optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights,
                         parameters=parameters,
                         optimizee_bounding_func=optimizee_bounding_func)

        self.optimizee_bounding_func = optimizee_bounding_func
        self.optimizee_create_individual = optimizee_create_individual
        self.optimizee_fitness_weights = optimizee_fitness_weights
        self.optimizee_create_new_individuals = optimizee_create_new_individuals
        self.parameters = parameters

        traj.f_add_parameter('gamma', parameters.gamma, comment='Noise level')
        # traj.f_add_parameter('p', parameters.p, comment='Exact solution')
        traj.f_add_parameter('noise', parameters.noise,
                             comment='Multivariate noise distribution')
        traj.f_add_parameter('tol', parameters.tol,
                             comment='Convergence threshold')
        traj.f_add_parameter('maxit', parameters.maxit,
                             comment='Maximum iterations')
        traj.f_add_parameter('n_iteration', parameters.n_iteration,
                             comment='Number of iterations to run')
        traj.f_add_parameter('stopping_crit', parameters.stopping_crit,
                             comment='Name of stopping criterion')
        traj.f_add_parameter('shuffle', parameters.shuffle)
        traj.f_add_parameter('n_batches', parameters.n_batches)
        traj.f_add_parameter('online', parameters.online)
        traj.f_add_parameter('epsilon', parameters.epsilon)
        traj.f_add_parameter('decay_rate', parameters.decay_rate)
        traj.f_add_parameter('seed', np.uint32(parameters.seed),
                             comment='Seed used for random number generation '
                                     'in optimizer')
        traj.f_add_parameter('pop_size', parameters.pop_size)

        _, self.optimizee_individual_dict_spec = dict_to_list(
            self.optimizee_create_individual(), get_dict_spec=True)

        traj.results.f_add_result_group('generation_params')

        # Set the random state seed for distribution
        self.random_state = np.random.RandomState(traj.parameters.seed)

        #: The current generation number
        self.g = 0
        # for the sampling procedure
        # `epsilon` value given by the user
        self.epsilon = parameters.epsilon
        # decay rate
        self.decay_rate = parameters.decay_rate

        #: The population (i.e. list of individuals) to be evaluated at the
        # next iteration
        current_eval_pop = [self.optimizee_create_individual() for _ in range(parameters.pop_size)]

        if optimizee_bounding_func is not None:
            current_eval_pop = [self.optimizee_bounding_func(ind) for ind in current_eval_pop]

        self.eval_pop = current_eval_pop
        # self.eval_pop_asarray = np.array([dict_to_list(x) for x in self.eval_pop])
        self.best_fitness = 0.
        self.best_individual = None

        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        This is the key function of this class. Given a set of :obj:`fitnesses_results`,
        and the :obj:`traj`, it uses the fitness to decide on the next set of
        parameters to be evaluated. Then it fills the :attr:`.Optimizer.eval_pop`
        with the list of parameters it wants evaluated at the next simulation
        cycle, increments :attr:`.Optimizer.g` and calls :meth:`._expand_trajectory`

        :param  ~l2l.utils.trajectory.Trajectory traj: The trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :param list fitnesses_results: This is a list of fitness results that contain tuples run index and the fitness.
            It is of the form `[(run_idx, run), ...]`

        """

        # old_eval_pop = self.eval_pop.copy()
        self.eval_pop.clear()

        n_iter = traj.generation
        individuals = traj.individuals[n_iter]
        gamma = traj.gamma
        all_results = []
        fitnesses = []
        data_input = None
        data_targets = None
        # go over all individuals
        for i in individuals:
            # shifts are the ensembles
            # weights = i.weights
            ens = np.array(i.shift)
            ensemble_size = ens.shape[0]
            data_input = i.input
            data_targets = i.targets
            # get the score/fitness of the individual
            fitness_per_individual = traj.current_results[i.ind_idx][1][0]
            fitnesses.append(fitness_per_individual)
            model_output = traj.current_results[i.ind_idx][1][1]
            new_ensembles = update_enknf(data=data_input,
                                         ensemble=ens,
                                         ensemble_size=ensemble_size,
                                         moments1=np.mean(ens, axis=0),
                                         u_exact=None,
                                         observations=data_targets,
                                         model_output=model_output,
                                         noise=traj.noise,
                                         p=None, gamma=gamma, tol=traj.tol,
                                         maxit=traj.maxit,
                                         stopping_crit=traj.stopping_crit,
                                         online=traj.online,
                                         shuffle=traj.shuffle)
            all_results.append(new_ensembles[0])

        generation_name = 'generation_{}'.format(self.g)
        traj.results.generation_params.f_add_result_group(generation_name)
        fitnesses = np.array(fitnesses)

        if self.g > 1 and self.g % 1000 == 0:
            ranking_idx = list(reversed(np.argsort(fitnesses)))
            best_fitness = fitnesses[ranking_idx][0]
            self.best_fitness = best_fitness
            best_ranking_idx = ranking_idx[0]
            self.best_individual = individuals[best_ranking_idx]
            # do the decay
            eps = self.epsilon * np.exp(-self.decay_rate * self.g)
            # now do the sampling
            shifts = [
                self.optimizee_create_new_individuals(self.random_state,
                                                      individuals[
                                                          best_ranking_idx].shift,
                                                      eps)
                for _ in range(traj.pop_size)]

            self.eval_pop = [dict(shift=shifts[i],
                                  targets=data_targets,
                                  input=data_input)
                             for i in range(traj.pop_size)]
        else:
            self.eval_pop = [dict(shift=all_results[i],
                                  targets=data_targets,
                                  input=data_input)
                             for i in range(traj.pop_size)]
        self.g += 1
        self._expand_trajectory(traj)

    def end(self, traj):
        """
        Run any code required to clean-up, print final individuals etc.

        :param  ~l2l.utils.trajectory.Trajectory traj: The  trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        """
        best_last_indiv = self.best_individual

        traj.f_add_result('final_individual', best_last_indiv)

        logger.info("The last individual was %s with fitness %s",
                    self.best_individual, self.best_fitness)
        logger.info("-- End of (successful) gradient descent --")

    def _expand_trajectory(self, traj):
        """
        Add as many explored runs as individuals that need to be evaluated. Furthermore, add the individuals as explored
        parameters.

        :param  ~l2l.utils.trajectory.Trajectory traj: The  trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :return:
        """

        grouped_params_dict = get_grouped_dict(self.eval_pop)
        grouped_params_dict = {'individual.' + key: val for key, val in
                               grouped_params_dict.items()}

        final_params_dict = {'generation': [self.g],
                             'ind_idx': range(len(self.eval_pop))}
        final_params_dict.update(grouped_params_dict)

        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`: This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``, so that every individual has just one
        # unique index within a generation.
        traj.f_expand(cartesian_product(final_params_dict,
                                        [('ind_idx',) + tuple(
                                            grouped_params_dict.keys()),
                                         'generation']))
