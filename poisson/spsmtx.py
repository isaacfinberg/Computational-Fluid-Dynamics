import numpy
import scipy.sparse as sps

def mtx( grid, ivar ):
    
    """Build sparse matrix for linear solver.

    Arguments
    ---------
    grid : Grid object
        Grid containing data.
    ivar : string
        Name of the grid variable of the numerical solution.
        
    Returns
    -------
    A: CSR format matrix

    """
    
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    bc_type = grid.bc_type[ ivar ]
    N = nx * ny
    
    U = numpy.diag( numpy.ones( N - 1 ), k = 1 )
    L = numpy.diag( numpy.ones( N - 1 ), k = -1 )
    UU = numpy.diag( numpy.ones( N - nx ), k = nx )
    LL = numpy.diag( numpy.ones( N - nx ), k = -nx )  
    A = LL + L + numpy.diag( -4 * numpy.ones( N ) ) + U + UU
    
    if bc_type[ 0 ] == 'neumann':
        wall = -3
        corner = -2
    else:
        wall = -5
        corner = -6
      
    diag = 0
    for j in range( ny ):
        for i in range( nx ):

            if j == 0 or j == ny - 1:
                A[ diag, diag ] = wall

            if i % ( nx - 1 ) == 0:
                A[ diag, diag ] = wall
                if i == 0 and j > 0 and j < ny - 1:
                    A[ diag, diag - 1 ] = 0
                if i == nx - 1 and j > 0 and j < ny - 1:
                    A[ diag, diag + 1 ] = 0

            if ( j == 0 or j == ny - 1 ) and i % ( nx - 1) == 0:
                A[ diag, diag ] = corner
                if i == nx - 1 and j == 0:
                    A[ diag, diag + 1 ] = 0
                if i == 0 and j == ny - 1:
                    A[ diag, diag - 1 ] = 0

            diag += 1
    
    A = A / dx**2
    A = sps.csr_matrix( A )
    
    return A
    