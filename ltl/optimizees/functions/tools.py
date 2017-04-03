
def plot(function):
    """
    Implements plotting of 2D functions generated by FunctionGenerator
    :param function: Instance of FunctionGenerator
    """
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)

    # Make data.
    X = np.arange(function.bound[0], function.bound[1], 0.05)
    Y = np.arange(function.bound[0], function.bound[1], 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = [function.cost_function([x, y]) for x, y in zip(X.ravel(), Y.ravel())]
    Z = np.array(Z).reshape(X.shape)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
