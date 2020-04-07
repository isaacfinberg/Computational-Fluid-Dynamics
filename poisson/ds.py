import numpy
from scipy.sparse.linalg.dsolve import linsolve
from spsmtx import mtx

def solve_direct( grid, ivar, rvar, verbose = False ):
    
    """Solve the Poisson system using a direct solver.

    Arguments
    ---------
    grid : Grid object
        Grid containing data.
    ivar : string
        Name of the grid variable of the numerical solution.
    rvar : string
        Name of the grid variable of the right-hand side.

    Returns
    -------
    residual: float
        Final residual.
    verbose : bool, optional
        Set True to display convergence information;
        default: False.

    """
    
    phi = grid.get_values( ivar )
    b = grid.get_values( rvar )
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    A = mtx( grid, ivar )
    x = linsolve.spsolve( A, b[ 1:-1, 1:-1 ].flatten() )
    residual = numpy.linalg.norm( A * x - b[ 1:-1, 1:-1 ].flatten() )

    phi[ 1:-1,1:-1 ] = numpy.reshape( x, ( nx, ny ) )
    grid.fill_guard_cells( ivar )

    if verbose:
        print( 'Direct Solver:' )
        print( '- Final residual: {}'.format( residual ) )

    return residual