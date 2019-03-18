import numpy as np
from numpy import sqrt
from numpy.linalg import norm, inv, solve
# from moments import moments


def update_enknf(data, ensemble, ensemble_size, moments1, u_exact,
                 observations, model, gamma, p, noise, tol, maxit, stopping_crit):
    """
    Ensemble Kalman Filter

    :param ensemble: nd numpy array, contains the calculated ensembles u
    :param ensemble_size: int, number of ensembles
    :param moments1: nd numpy array, first moment (mean)
    :param u_exact: nd numpy array, exact control
    :param observations: nd numpy array, noisy observation
    :param G: nd numpy array, Model
            `G` maps the control (dim n) into the observed data `y` (dim k),
            inverse of `A` using `numpy.inv`
    :param  gamma: nd numpy array
            `noise_level * I` (I is identity matrix)
    :param p: nd numpy array
            Exact solution given by :math:`G * u_exact`, where `G` is inverse
            of a linear elliptic function `L`, it maps the control into the
            observed data, see section 5.1 of Herty2018
    :param noise: nd numpy array, Multivariate normal distribution
    :param tol: float, tolerance for convergence
    :param maxit: int, maximum number of iteration
        if the update step does not stop
    :param stopping_crit: str, stopping criterion,
            `discrepancy`: checks if the actual misfit is smaller or equal
            to the noise
            `relative`: checks if the absolute value between actual and
            previous misfit is smaller than given tolerance `tol`
    :return:
        M: nd numpy array, Misfit
            Measures the quality of the solution at each iteration,
            see (Herty2018 eq.30 and eq.31)
        E: nd numpy array, Deviation of :math:`v^j` from the mean,
            where the :math:`v^j` is the j-th sample of the approximated
            distribution of samples, see (Herty2018 eq.28)
        R: nd numpy array, Deviation of :math:`v^j` from the "true" solution
            :math:`u^{\dagger}`, see (Herty2018 eq.29)
        AE: nd numpy array, Deviation of :math:`v^j` under the application of
            the model `G`, see (Herty2018 eq.28)
        AR: nd numpy array, Deviation of from the "true" solution under
            the application of model `G`, see (Herty2018 eq.28)
    """
    # Initializations
    M = []
    E = []
    R = []
    AE = []
    AR = []

    m1 = moments1

    e = np.zeros_like(ensemble)
    r = np.zeros_like(ensemble)
    misfit = np.zeros_like(ensemble)

    norm_uexact2 = None
    norm_p2 = None

    if u_exact is not None:
        norm_uexact2 = norm(u_exact) ** 2
    if p is not None:
        norm_p2 = norm(p) ** 2

    # get shapes
    gamma_s, dims = _get_shapes(observations, ensemble)

    if isinstance(gamma, (int, float)):
        if float(gamma) == 0.:
            gamma = np.eye(10)
    sqrt_inv_gamma = sqrt(inv(gamma))

    Cpp = None
    Cup = None

    ensemble = ensemble

    total_cost = []
    for i in range(maxit):
        # calculate loss
        # model_output = model.get_output_activation(data,
        #                                            *_flatten_to_net_weights(
        #                                                model, m1))
        # tc = _stopping_criterion(observations, model_output.T,
        #                          dims=dims,
        #                          loss_function='norm')
        # total_cost.append(tc)
        idx = np.random.randint(0, dims)
        # g_all are all model evaluations for one pass
        g_all = []
        for l in range(ensemble_size):
            g_all.append(model.get_output_activation(data[idx],
                                                     *_flatten_to_net_weights(
                                                         model,
                                                         ensemble[:, l])))
            e[:, l] = ensemble[:, l] - m1
        g_all = np.array(g_all).T
        if u_exact is not None:
            misfit, r = _calculate_misfit(ensemble, ensemble_size, dims,
                                          misfit, r, g_all, u_exact, noise)
        _convergence(norm_uexact2=norm_uexact2,
                     norm_p2=norm_p2,
                     sqrt_inv_gamma=sqrt_inv_gamma,
                     ensemble_size=ensemble_size, G=g_all, m1=m1, M=M, E=E,
                     R=R, AE=AE, AR=AR, e=e, r=r, misfit=misfit)
        if stopping_crit == 'discrepancy' and noise > 0:
            if M[i] <= np.linalg.norm(noise, 2) ** 2:
                break
        elif stopping_crit == 'relative':
            if i >= 1:
                if np.abs(M[i] - M[i - 1]) < tol:
                    break
        else:
            pass
            # sc = _stopping_criterion(y=observations, y_hat=g_all)
            # if sc <= tol:
            #     break
        if g_all.ndim > 2:
            for d in range(dims):
                g = g_all[d]
                ensemble = _update_step(ensemble, observations, g, gamma,
                                        ensemble_size, d)
        else:
            g = g_all
            ensemble = _update_step(ensemble, observations, g, gamma,
                                    ensemble_size, idx)
        m1 = np.mean(ensemble, axis=1)
    # return M, E, R, AE, AR, Cpp, Cup, m1
    return ensemble, Cpp, Cup, m1


def _update_step(ensemble, observations, g,  gamma, ensemble_size, idx):
    """
    Update step of the kalman filter
    Calculates the covariances and returns new ensembles
    """
    # Calculate the covariances
    Cup = _cov_mat(ensemble, g, ensemble_size)
    Cpp = _cov_mat(g, g, ensemble_size)
    for j in range(ensemble_size):
        # create one hot vector
        target = _one_hot_vector(observations[idx], g.shape[0])
        tmp = solve(Cpp + gamma, target - g[:, [j]])
        ensemble[:, [j]] = ensemble[:, [j]] + (Cup @ tmp)
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

    cov = np.zeros((x.shape[0], y.shape[0]))
    for j in range(ensemble_size):
        try:
            cov = cov + np.tensordot((x[:, j] - x_bar), (y[:, j] - y_bar).T, 0)
        except IndexError:
            cov = cov + np.tensordot((x[:, j] - x_bar), (y - y_bar).T, 0)
    cov /= ensemble_size
    return cov


def _get_mean(x):
    """
    Depending on the shape returns the correct mean
    """
    if x.shape[1] == 1:
        return np.mean(x)
    return np.mean(x, axis=1)


def _get_shapes(observations, ensemble):
    """
    Returns individual shapes
    """
    gamma_shape = observations.shape[0]
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


def _stopping_criterion(y, y_hat, dims, loss_function='BCE'):
    """
    Loss functions
    :param y: target
    :param y_hat: calculated output (here G(u) or feed-forword output a)
    :param loss_function: name of the loss function
    :return: cost calculated according to `loss_function`
    """
    cost = 0.0
    for d in range(dims):
        if loss_function == 'BCE':
            term1 = -y * np.log(y_hat[d])
            term2 = (1 - y) * np.log(1 - y_hat[d])
            cost += np.sum(term1 - term2)
        elif loss_function == 'MAE' or 'L1':
            cost += np.sum(np.absolute(y_hat[d] - y))
        elif loss_function == 'MSE' or 'L2':
            cost += np.sum((y_hat[d] - y) ** 2 / y.size)
        elif loss_function == 'norm':
            cost += norm(y - y_hat[d])
        else:
            raise KeyError(
                'Loss Function \'{}\' not understood.'.format(loss_function))
    return cost / dims


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


def _calculate_misfit(ensemble, ensemble_size, dims, misfit, r, g_all, u_exact,
                      noise):
    """
    Calculates and returns the misfit and the deviation from the true solution
    """
    for d in range(dims):
        if dims >= 2:
            g = g_all[d]
        else:
            g = g_all
        for l in range(ensemble_size):
            r[:, l] = ensemble[:, l] - u_exact
            misfit[:, l] = g[:, l] * r[l, 0] - noise
    return misfit, r


def _one_hot_vector(index, shape):
    """
    Encode targets into one-hot representation
    """
    target = np.zeros(shape)
    target[index] = 1.0
    target = target[np.newaxis].T
    return target


def _encode_targets(targets, shape):
    return [_one_hot_vector(targets[i], shape) for i in range(targets.shape[0])]


def _mini_batches(shape, n_batches):
    """
    Splits the data set into `n_batches` of shape `shape`
    """
    return np.array_split(range(shape), n_batches)


def _shuffle(data, targets):
    """
    Shuffles the data and targets by permuting them
    """
    indices = np.random.permutation(targets.shape[0])
    return data[indices], targets[indices]
