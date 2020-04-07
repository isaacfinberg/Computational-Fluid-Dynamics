import numpy
import scipy
from spsmtx import mtx

def solve_cg( grid, ivar, rvar, verbose = False ):
    
    """Solve the Poisson system using a the Conjugate Gradient method.

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

    def solve( A, b ):
        num_iters = 0
        def callback( xk ):
            num_iters+=1
        x, status = scipy.sparse.linalg.cg( A, b, tol = 1e-10, callback = callback )
        return x, num_iters
    
    x, num_iters = solve( A, b[ 1:-1, 1:-1 ].flatten() )
#     x = scipy.sparse.linalg.cg( A, b[ 1:-1, 1:-1 ].flatten() )[ 0 ]
    residual = numpy.linalg.norm( A * x - b[ 1:-1, 1:-1 ].flatten() )

    phi[ 1:-1,1:-1 ] = numpy.reshape( x, ( nx, ny ) )
    grid.fill_guard_cells( ivar )

    if verbose:
        print('Conjugate Gradient method:')
        print('- Final residual: {}'.format(residual))

    return residual, num_iters