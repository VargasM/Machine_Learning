## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
import matplotlib.pyplot as plot

if len( sys.argv ) < 2:
  print( 'Usage: python ' + sys.argv[ 0 ] + ' csv' )
  sys.exit( 1 )
# end if

## -- Read data
D = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )
numpy.random.shuffle( D )

## -- Separate X and y
X = D[ : , 0 : 2 ]
y = D[ : , 2 : ]

## -- Plot data
Z = X[ y[ : , 0 ] == 0 , : ]
O = X[ y[ : , 0 ] == 1 , : ]
figure, axis = plot.subplots( )
axis.scatter( Z[ : , 0 ], Z[ : , 1 ], marker = '+', label = 'Class 0' )
axis.scatter( O[ : , 0 ], O[ : , 1 ], marker = 'x', label = 'Class 1' )
axis.legend( )
axis.grid( True )
axis.set_aspect( 1 )

plot.show( )

## eof - $RCSfile$
