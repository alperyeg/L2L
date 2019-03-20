import logging
from collections import namedtuple
from l2l import get_grouped_dict
from l2l.utils.tools import cartesian_product
from .updateEnKF import update_enknf

import numpy as np

from sklearn.datasets import load_digits

from l2l import dict_to_list
from l2l import list_to_dict
from l2l.optimizers.optimizer import Optimizer

logger = logging.getLogger("optimizers.kalmanfilter")

EnsembleKalmanFilterParameters = namedtuple(
    'EnsembleKalmanFilter', ['noise', 'gamma', 'tol',
                             'maxit', 'stopping_crit', 'n_iteration',
                             'pop_size', 'shuffle', 'n_batches', 'online']
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
"""


class EnsembleKalmanFilter(Optimizer):
    """
    Class for an Ensemble Kalman Filter optimizer
    """
    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
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

        _, self.optimizee_individual_dict_spec = dict_to_list(
            self.optimizee_create_individual(), get_dict_spec=True)

        #: The current generation number
        self.g = 0
        #: The population (i.e. list of individuals) to be evaluated at the
        # next iteration
        current_eval_pop = [self.optimizee_create_individual() for _ in range(parameters.pop_size)]

        if optimizee_bounding_func is not None:
            current_eval_pop = [self.optimizee_bounding_func(ind) for ind in current_eval_pop]

        self.eval_pop = current_eval_pop
        self.eval_pop_asarray = np.array([dict_to_list(x) for x in self.eval_pop])

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
        # NOTE: Always remember to keep the following two lines.
        self.g += 1
        n_iter = traj.generation
        individuals = traj.individuals[n_iter]
        # works now for pop_size = 1
        shifts_per_individual = []
        gamma = traj.gamma
        for i in individuals:
            # shifts are the ensembles
            # weights = i.weights
            # TODO change to row view
            ens = np.array(i.shift)
            shifts_per_individual.append(ens)
            ensemble_size = ens.shape[0]
            # if weights.ndim == 1:
            #     weights = weights[np.newaxis]
            data_input = i.input
            data_targets = i.targets
            model = i.model
            results = update_enknf(data=data_input[0:100],
                                   ensemble=ens,
                                   ensemble_size=ensemble_size,
                                   moments1=np.mean(ens, axis=0),
                                   u_exact=None,
                                   observations=data_targets[0:100],
                                   model=model, noise=traj.noise, p=None,
                                   gamma=gamma,  tol=traj.tol,
                                   maxit=traj.maxit,
                                   stopping_crit=traj.stopping_crit,
                                   online=True)
        generation_name = 'generation_{}'.format(self.g)
        traj.results.generation_params.f_add_result_group(generation_name)
        generation_result_dict = {
            'success': []
        }

        traj.results.generation_params.f_add_result(
            generation_name + '.algorithm_params', generation_result_dict)
        # TODO: Set eval_pop to the values of parameters you want to evaluate
        #  in the next cycle
        # self.eval_pop = ...
        self._expand_trajectory(traj)

    def end(self, traj):
        """
        Run any code required to clean-up, print final individuals etc.

        :param  ~l2l.utils.trajectory.Trajectory traj: The  trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        """
        pass

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
