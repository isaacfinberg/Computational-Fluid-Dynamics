"""User defined module for simulation."""

import numpy


def get_analytical(grid, asol, user_bc):
    """Compute and set the analytical solution.

    Arguments
    ---------
    grid : flowx.Grid object
        Grid containing data.
    asol : string
        Name of the variable on the grid.

    """
    X, Y = numpy.meshgrid(grid.x, grid.y)

    if(user_bc == 'dirichlet'):
        values = numpy.sin(2 * numpy.pi * X) * numpy.sin(2 * numpy.pi * Y)
    else:
        values = numpy.cos(2 * numpy.pi * X) * numpy.cos(2 * numpy.pi * Y)

    grid.set_values(asol, values.transpose())


def get_rhs(grid, rvar, user_bc):
    """Compute and set the right-hand side of the Poisson system.

    Arguments
    ---------
    grid : flowx.Grid object
        Grid containing data.
    rvar : string
        Name of the variable on the grid.

    """
    X, Y = numpy.meshgrid(grid.x, grid.y)

    if(user_bc == 'dirichlet'):
        values = (-8 * numpy.pi**2 *
                  numpy.sin(2 * numpy.pi * X) * numpy.sin(2 * numpy.pi * Y))
    else:
        values = (-8 * numpy.pi**2 *
                  numpy.cos(2 * numpy.pi * X) * numpy.cos(2 * numpy.pi * Y))

    grid.set_values(rvar, values.transpose())
