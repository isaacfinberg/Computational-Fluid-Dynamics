"""Routine to solve the Poisson system with SOR."""

import numpy

def solve_gsor(grid, ivar, rvar, omega, maxiter=3000, tol=1e-9, verbose=False):
    """Solve the Poisson system using a Jacobi method.

    Arguments
    ---------
    grid : Grid object
        Grid containing data.
    ivar : string
        Name of the grid variable of the numerical solution.
    rvar : string
        Name of the grid variable of the right-hand side.
    omega: integer
        Extrapolation factor.
    maxiter : integer, optional
        Maximum number of iterations;
        default: 3000
    tol : float, optional
        Exit-criterion tolerance;
        default: 1e-9

    Returns
    -------
    ites: integer
        Number of iterations computed.
    residual: float
        Final residual.
    verbose : bool, optional
        Set True to display convergence information;
        default: False.

    """

    phi = grid.get_values( ivar )
    nx, ny = grid.nx, grid.ny
    
    ites = 0
    residual = tol + 1.0
    while ites < maxiter and residual > tol:
        phi_old = phi.copy()
        for j in range( 1, ny - 1 ):
            for i in range( 1, nx - 1 ):
                phi[ j, i ] = ( 1 - omega ) * phi_old[ j, i ] +  omega / 4 * ( phi_old[ j, i - 1 ] + phi_old[ j, i + 1 ] + phi_old[ j - 1, i ] + phi_old[ j + 1, i ] )

        grid.fill_guard_cells( ivar )
        residual = ( numpy.sqrt( numpy.sum( ( phi - phi_old )**2 ) / ( ( grid.nx + 2 ) * ( grid.ny + 2 ) ) ) )
        ites += 1
    
    if verbose:
        print( 'SOR method:' )
        if ites == maxiter:
            print( 'Warning: maximum number of iterations reached!' )
        print( '- Number of iterations: {}'.format( ites ) )
        print( '- Final residual: {}'.format( residual ) )
    
    return ites, residual