## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from scipy.special import softmax
from .Base import *

'''
'''
class SoftMax( Base ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __call__( self, z, derivative = False ):
    if derivative:
      return None
    else:
      e = numpy.exp(z - numpy.max(z))
      s = e.sum()
      return e / s
    # end if
  # end def

# end class

## eof - $RCSfile$
