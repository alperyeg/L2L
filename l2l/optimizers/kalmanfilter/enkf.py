import numpy as np
import abc
from numpy import sqrt
from numpy.linalg import norm, inv, solve
from abc import ABCMeta


# TODO: revise the documentation


class KalmanFilter(metaclass=ABCMeta):

    @abc.abstractmethod
    def fit(self, data, ensemble, ensemble_size, moments1,
            observations, model_output, gamma, noise, p, u_exact):
        pass


class EnsembleKalmanFilter(KalmanFilter):
    def __init__(self, tol, maxit, stopping_crit, n_batches=32,
                 shuffle=True, online=False, converge=False,
                 loss_function='norm'):
        """
        Ensemble Kalman Filter (EnKF)

        EnKF following the formulation found in Iglesias et al. (2013), 
        The Ensemble Kalman Filter for Inverse Problems.

        :param tol: float, tolerance value for convergence
        :param maxit: int, maximum number of iterations
        :param stopping_crit: str, stopping criterion,
            `discrepancy`: checks if the actual misfit is smaller or equal
            to the noise
            `relative`: checks if the absolute value between actual and
            previous misfit is smaller than given tolerance `tol`
            otherwise calculates the loss between iteration `n` and `n-1` and
            stops if the difference is `< tol`
        :param shuffle: bool, True if the dataset should be shuffled,
            Default is `True`.
        :param n_batches, int,  number of batches to used in mini-batch, 
            Default is `32`.
        :param online, bool, True if one random data point is requested,
            between [0, dims], otherwise do mini-batch, Default is False
        :param converge, bool, Checks and stops the iteration and updadting step
            if convergence is reached, Default is `False`.
        :param loss_function, string, name of the loss function
           `MAE` is the Mean Absolute Error or l1 - loss
           `MSE` is the Mean Squared Error or l2 - loss
           `CE` cross-entropy loss
           `norm` norm-2 or Forbenius norm,
           Default is `norm`

        """
        self.M = []
        self.E = []
        self.R = []
        self.AE = []
        self.AR = []
        self.total_cost = []

        self.Cpp = None
        self.Cup = None
        self.ensemble = None
        self.observations = None
        self.data = None
        self.e = None
        self.r = None
        self.misfit = None
        self.u_exact = None
        self.norm_uexact2 = None
        self.norm_p2 = None
        self.m1 = None

        self.maxit = maxit
        self.shuffle = shuffle
        self.stopping_crit = stopping_crit
        self.tol = tol
        self.converge = converge
        self.online = online
        self.n_batches = n_batches
        self.loss_function = loss_function
        self.gamma = 0.
        self.gamma_s = 0
        self.dims = 0

    def fit(self, data, ensemble, ensemble_size, moments1,
            observations, model_output, gamma, noise=0., p=None, u_exact=None):
        """
        Prediction and update step of the EnKF
        
        Calculates new ensembles.

        :param ensemble: nd numpy array, contains ensembles `u`
        :param ensemble_size: int, number of ensembles
        :param moments1: nd numpy array, first moment (mean)
        :param u_exact: nd numpy array, exact control, e.g. if weight sof the model
                   are known. Default is `None`
        :param observations: nd numpy array, observation or targets
        :param model_output: nd numpy array, output of the model
            In terms of the Kalman Filter the model maps the ensembles (dim n)
            into the observed data `y` (dim k). E.g. network output activity
        :param noise: nd numpy array, Noise can be added to the model (for `gamma`) 
            and is used in the misfit calculation for convergence.
            E.g. multivariate normal distribution. Default is `0.0`
        :param  gamma: nd numpy array, Normalizes the model-data distance in the
            update step, :`noise * I` (I is identity matrix) or
            :math:`\gamma=I` if `noise` is zero
        :param p: nd numpy array
            Exact solution given by :math:`G * u_exact`, where `G` is inverse
            of a linear elliptic function `L`, it maps the control into the
            observed data, Default is `None`
        :return self, Possible outputs are, if calculated:
            ensembles: nd numpy array, optimized `ensembles`
            Cpp: nd numpy array, covariance matrix of the model output
            Cup: nd numpy array, covariance matrix of the model output and the 
                ensembles 
            M: nd numpy array, Misfit
                Measures the quality of the solution at each iteration
            E: nd numpy array, Deviation of :math:`v^j` from the mean,
                where the :math:`v^j` is the j-th sample of the approximated
                distribution of samples
            R: nd numpy array, Deviation of :math:`v^j` from the "true" solution
                :math:`u^{\dagger}`, see (Herty2018 eq.29)
            AE: nd numpy array, Deviation of :math:`v^j` under the application of
                the model `G`, see (Herty2018 eq.28)
            AR: nd numpy array, Deviation of from the "true" solution under
                the application of model `G`, see (Herty2018 eq.28)
           total_cost: list, contains the costs as defined in `loss_function`
        """

        if isinstance(gamma, (int, float)):
            if float(gamma) == 0.:
                self.gamma = np.eye(self.gamma_s)
        sqrt_inv_gamma = sqrt(inv(self.gamma))
        self.e = np.zeros_like(ensemble)
        self.r = np.zeros_like(ensemble)
        self.misfit = np.zeros_like(ensemble)
        self.m1 = moments1
        # get shapes
        self.gamma_s, self.dims = _get_shapes(observations, model_output)

        self.total_cost = []

        if self.u_exact is not None:
            self.norm_uexact2 = norm(u_exact) ** 2
            if p is not None:
                self.norm_p2 = norm(p) ** 2

        # copy the data so we do not overwrite the original arguments
        self.ensemble = ensemble.copy()
        self.observations = observations.copy()
        self.observations = _encode_targets(observations, self.gamma_s)
        self.data = data.copy()

        for i in range(self.maxit):
            if (i % 100) == 0:
                print('Iteration {}/{}'.format(i, self.maxit))
            if self.shuffle:
                data, observations = _shuffle(self.data, self.observations)

            # check for early stopping
            if self.converge:
                _convergence(norm_uexact2=self.norm_uexact2,
                             norm_p2=self.norm_p2,
                             sqrt_inv_gamma=sqrt_inv_gamma,
                             ensemble_size=ensemble_size,
                             G=model_output, m1=self.m1, M=self.M,
                             E=self.E,
                             R=self.R, AE=self.AE, AR=self.AR, e=self.e,
                                  r=self.r, misfit=self.misfit)
            if self.stopping_crit == 'discrepancy' and noise > 0:
                if self.M[i] <= np.linalg.norm(noise, 2) ** 2:
                    break
            elif self.stopping_crit == 'relative' and self.converge:
                if i >= 1:
                    if np.abs(self.M[i] - self.M[i - 1]) < self.tol:
                        break
            elif self.stopping_crit == 'cost' and self.converge:
                # calculate loss
                cost = _calculate_cost(y=observations, y_hat=model_output.T,
                                       loss_function=self.loss_function)
                self.total_cost.append(cost)
                # check if early stopping is possible
                if i >= 1:
                    if np.abs(self.total_cost[-1] - cost) <= self.tol:
                        break
            # now get mini_batches
            if self.n_batches > self.dims:
                num_batches = 1
            else:
                num_batches = self.n_batches
            mini_batches = _get_batches(
                num_batches, shape=self.dims, online=self.online)
            for idx in mini_batches:
                for l in range(ensemble_size):
                    self.e[l] = ensemble[l] - self.m1
                if u_exact is not None:
                    self.misfit[:, :, idx], r = _calculate_misfit(ensemble,
                                                                  ensemble_size,
                                                                  self.misfit[:, :, idx],
                                                                  self.r,
                                                                  model_output[:, :, idx],
                                                                  u_exact,
                                                                  noise)
                # in case of online learning idx should be an int
                # put it into a list to loop over it
                if not isinstance(idx, np.ndarray):
                    idx = [idx]

                for d in idx:
                    # now get only the individuals output according to idx
                    g_tmp = model_output[:, :, d]
                    self.ensemble = _update_step(ensemble, observations[d],
                                                 g_tmp, gamma, ensemble_size)

            self.m1 = np.mean(ensemble, axis=0)
        return self


def _update_step(ensemble, observations, g,  gamma, ensemble_size):
    """
    Update step of the kalman filter
    Calculates the covariances and returns new ensembles
    """
    # Calculate the covariances
    Cup = _cov_mat(ensemble, g, ensemble_size)
    Cpp = _cov_mat(g, g, ensemble_size)
    for j in range(ensemble_size):
        # create one hot vector
        target = observations
        tmp = solve(Cpp + gamma, target - g[j])
        ensemble[j] = ensemble[j] + (Cup @ tmp)
    return ensemble


def _convergence(m1, sqrt_inv_gamma,
                 ensemble_size, G,
                 M, E, R, AE, AR,
                 e, r, misfit,
                 norm_uexact2=None, norm_p2=None):
    E.append(norm(e) ** 2 / norm(m1) ** 2 / ensemble_size)
    # AE.append(norm(sqrt_inv_gamma @ G) ** 2 / norm(G @ m1) ** 2 / ensemble_size)
    if norm_uexact2 is not None:
        R.append((norm(r) ** 2 / norm_uexact2) / ensemble_size)
        M.append((norm(misfit) ** 2) / ensemble_size)
    if norm_p2 is not None:
        AR.append(norm(sqrt_inv_gamma @ G @ r) ** 2 / norm_p2 / ensemble_size)
    return


def _cov_mat(x, y, ensemble_size):
    """
    Covariance matrix
    """
    x_bar = _get_mean(x)
    y_bar = _get_mean(y)

    cov = 0.0
    for j in range(ensemble_size):
        try:
            cov = cov + np.tensordot((x[j] - x_bar), (y[j] - y_bar).T, 0)
        except IndexError:
            cov = cov + np.tensordot((x - x_bar), (y - y_bar).T, 0)
    cov /= ensemble_size
    return cov


def _get_mean(x):
    """
    Depending on the shape returns the correct mean
    """
    if x.shape[1] == 1:
        return np.mean(x)
    return np.mean(x, axis=0)


def _get_shapes(observations, model_output):
    """
    Returns individual shapes

    :returns gamma_shape, length of the observation (here: size of last layer
                          of network)
    :returns dimensions, number of observations (and data)
    """
    if model_output.ndim > 2:
        gamma_shape = model_output.shape[1]
    else:
        gamma_shape = model_output.shape[0]
    dimensions = observations.shape[0]
    return gamma_shape, dimensions


def _convergence_non_vec(ensemble_size, u_exact, r, e, m1, gamma, g, p,
                         misfit):
    """
    Non vectorized convergence function
    """
    tmp_e = 0
    tmp_r = 0
    tmp_ae = 0
    tmp_ar = 0
    tmp_m = 0
    for l in range(ensemble_size):
        tmp_r += norm(r[:, l], 2) ** 2 / norm(u_exact, 2) ** 2
        tmp_e += norm(e[:, l], 2) ** 2 / norm(m1, 2) ** 2
        tmp_ae += sqrt(inv(gamma)) @ g @ e[:, l].T @ \
            sqrt(inv(gamma)) @ g @ e[:, l] / norm(g @ m1) ** 2
        tmp_ar += sqrt(inv(gamma)) @ g @ r[:, l].conj().T @ \
            sqrt(inv(gamma)) @ g @ r[:, l] / norm(p) ** 2
        tmp_m += norm(misfit[:, l], 2) ** 2
    return tmp_e, tmp_r, tmp_ae, tmp_ar, tmp_m


def _calculate_cost(y, y_hat, loss_function='BCE'):
    """
    Loss functions
    :param y: target
    :param y_hat: calculated output (here G(u) or feed-forword output a)
    :param loss_function: name of the loss function
           `MAE` is the Mean Absolute Error or l1 - loss
           `MSE` is the Mean Squared Error or l2 - loss
           `CE` cross-entropy loss, requires `y_hat` to be in [0, 1]
           `norm` norm-2 or Forbenius norm of `y - y_hat`
    :return: cost calculated according to `loss_function`
    """
    if loss_function == 'CE':
        term1 = -y * np.log(y_hat)
        term2 = (1 - y) * np.log(1 - y_hat)
        return np.sum(term1 - term2)
    elif loss_function == 'MAE':
        return np.sum(np.absolute(y_hat - y)) / len(y)
    elif loss_function == 'MSE':
        return np.sum((y_hat - y) ** 2) / len(y)
    elif loss_function == 'norm':
        return norm(y - y_hat)
    else:
        raise KeyError(
            'Loss Function \'{}\' not understood.'.format(loss_function))


def _l1_regularization(weights, lambda_=0.1):
    """
    Compute L1-regularization cost.
    """
    return (lambda_ / 2.0) * np.sum(np.abs(weights))


def _l2_regularization(weights, lambda_=0.1):
    """
    Compute L2-regularization cost.
    """
    return lambda_ * np.sum(weights ** 2)


def _flatten_to_net_weights(model, flattened_weights):
    weight_shapes = model.get_weights_shapes()

    cumulative_num_weights_per_layer = np.cumsum(
        [np.prod(weight_shape) for weight_shape in weight_shapes])

    weights = []
    for i, weight_shape in enumerate(weight_shapes):
        if i == 0:
            w = flattened_weights[
                :cumulative_num_weights_per_layer[i]].reshape(weight_shape)
        else:
            w = flattened_weights[
                cumulative_num_weights_per_layer[i - 1]:
                cumulative_num_weights_per_layer[i]].reshape(weight_shape)
        weights.append(w)
    return weights


def _calculate_misfit(ensemble, ensemble_size, misfit, r, g_all, u_exact,
                      noise):
    """
    Calculates and returns the misfit and the deviation from the true solution
    """
    for d in range(ensemble_size):
        r[d] = ensemble[d] - u_exact
        misfit[d] = g_all[d] * r[d, 0] - noise
    return misfit, r


def _one_hot_vector(index, shape):
    """
    Encode targets into one-hot representation
    """
    target = np.zeros(shape)
    target[index] = 1.0
    return target


def _encode_targets(targets, shape):
    return np.array(
        [_one_hot_vector(targets[i], shape) for i in range(targets.shape[0])])


def _shuffle(data, targets):
    """
    Shuffles the data and targets by permuting them
    """
    indices = np.random.permutation(targets.shape[0])
    return data[indices], targets[indices]


def _get_batches(n_batches, shape, online):
    """
    :param n_batches, int, number of batches
    :param shape, int, shape of the data
    :param online, bool, True if one random data point is requested,
                         between [0, dims], otherwise do mini-batch
    """
    if online:
        return [np.random.randint(0, shape)]
    else:
        num_batches = n_batches
        mini_batches = _mini_batches(shape=shape,
                                     n_batches=num_batches)
        return mini_batches


def _mini_batches(shape, n_batches):
    """
    Splits the data set into `n_batches` of shape `shape`
    """
    return np.array_split(range(shape), n_batches)
