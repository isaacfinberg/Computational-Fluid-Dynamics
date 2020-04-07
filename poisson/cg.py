"From scratch. Error in the solution with Dirichlet bc"

import numpy


def solve_conjugate_gradient( grid, ivar, rvar, maxiter = 3000, tol = 1e-9, verbose = False ):
    
    """Solve the Poisson system using a Conjugate Gradient method.

    Arguments
    ---------
    grid : Grid object
        Grid containing data.
    ivar : string
        Name of the grid variable of the numerical solution.
    rvar : string
        Name of the grid variable of the right-hand side.
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
    
    def Laplace( phi ):
        return ( -4.0 * phi[ 1:-1, 1:-1 ] + phi[ 1:-1, :-2 ] + phi[ 1:-1, 2: ] + phi[ :-2, 1:-1 ] + phi[ 2:, 1:-1 ] ) / dx**2
    
    def fill_guard_cells_neumann(x, bc_val, dx, dy):
        x[0, :] = bc_val * dx + x[1, :]
        x[-1, :] = bc_val * dx + x[-2, :]
        x[:, 0] = bc_val * dy + x[:, 1]
        x[:, -1] = bc_val * dy + x[:, -2]
    
    phi = grid.get_values( ivar )
    dx, dy = grid.dx, grid.dy
    b = grid.get_values( rvar )
    
    r = b[ 1:-1, 1:-1 ] - Laplace( phi )
    rk_norm = numpy.sum(r * r)  
    d = numpy.zeros_like( phi )  
    d[ 1:-1, 1:-1 ] = r  
   
    bc_type, bc_val = grid.bc_type[ivar][0], grid.bc_val[ivar][0]
    if bc_type == 'neumann':
        # Apply Neumann boundary conditions to search direction.
        fill_guard_cells_neumann( d, bc_val, dx, dy )
    
    residual = rk_norm
    ites = 0
    
    while residual > tol and ites < maxiter:
        Ad = Laplace( phi )
        Ad[ 1:-1, 1:-1 ] = Laplace( d )
        rk = r.copy()
        alpha = rk_norm / numpy.sum( d[ 1:-1, 1:-1 ] * Ad )
        phi[ 1:-1, 1:-1 ] = phi[ 1:-1, 1:-1 ] + alpha * d[ 1:-1, 1:-1 ]
        r = r - alpha * Ad
        beta = numpy.sum( r * r ) / numpy.sum( rk * rk )
        d[ 1:-1, 1:-1 ] = r + beta * d[ 1:-1, 1:-1 ]
        if bc_type == 'neumann':
            fill_guard_cells_neumann(d, bc_val, dx, dy)
        
        residual = numpy.sum( r * r )
        ites += 1
    grid.fill_guard_cells( ivar )
    
    if verbose:
        print('Conjugate Gradient method:')
        if ites == maxiter:
            print('Warning: maximum number of iterations reached!')
        print('- Number of iterations: {}'.format(ites))
        print('- Final residual: {}'.format(residual))
        
    return ites, residual