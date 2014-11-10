import cython
from cython cimport view
import numpy as np
cimport numpy as np

from cpython cimport PyCapsule_GetPointer # PyCObject_AsVoidPtr
from scipy.linalg.blas import fblas

REAL = np.float32
ctypedef np.float32_t REAL_t

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
 
ctypedef void (*sger_ptr) (const int *M, const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY, float *A, const int * LDA) nogil
cdef sger_ptr sger=<sger_ptr>PyCapsule_GetPointer(fblas.sger._cpointer , NULL)  # A := alpha*x*y.T + A

cdef void outer_prod(REAL_t*  x, REAL_t* y, REAL_t * out, int x_len, int y_len):
    sger(&y_len, &x_len, &ONEF, y, &ONE, x, &ONE, out, &y_len)

ctypedef void (*dgemv_ptr) (char *trans, int *m, int *n,\
    float *alpha, float *a, int *lda, float *x, int *incx,\
    float *beta,  float *y, int *incy)

ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil


cdef sdot_ptr sdot=<sdot_ptr>PyCapsule_GetPointer(fblas.sdot._cpointer, NULL) # float = dot(x, y)
cdef dgemv_ptr dgemv=<dgemv_ptr>PyCapsule_GetPointer(fblas.dgemv._cpointer, NULL)

cpdef np.ndarray[REAL_t, ndim=3] vector_outer_product(np.ndarray[REAL_t, ndim=2] _x, np.ndarray[REAL_t, ndim=2] _y):
    
    cdef int i, length = _x.shape[0], x_len = _x.shape[1], y_len = _y.shape[1]
    cdef int box_size = x_len * y_len
    cdef np.ndarray[REAL_t, ndim=3] result
    result = np.zeros([length, x_len, y_len], dtype = REAL)
                                        
    cdef REAL_t*  x = <REAL_t *>(np.PyArray_DATA(_x))
    cdef REAL_t* y = <REAL_t *>(np.PyArray_DATA(_y))
    
    cdef REAL_t[:,:] x_view = _x
    cdef REAL_t[:,:] y_view = _y
    
    for i in range(length):
        outer_prod(&x_view[i,0], &y_view[i,0], &result[i,0,0], x_len, y_len)
    
    return result.transpose((1,2,0))

def tensor_delta_down_with_output(
    np.ndarray[REAL_t, ndim=3] tensor,
    np.ndarray[REAL_t, ndim=2] dEdz,
    np.ndarray[REAL_t, ndim=2] input,
    np.ndarray[REAL_t, ndim=2] out):
    
    cdef:
        int size = dEdz.shape[1]
        np.ndarray[REAL_t, ndim=3] outer_dotted   = vector_outer_product(dEdz, input)
    
    for i in range(size):
        out += np.dot(tensor[i,:,:], outer_dotted[i,:,:]).T
        out += np.dot(tensor[i,:,:].T, outer_dotted[i,:,:]).T

def tensor_delta_down(
    np.ndarray[REAL_t, ndim=3] tensor,
    np.ndarray[REAL_t, ndim=2] dEdz,
    np.ndarray[REAL_t, ndim=2] input):
    
    cdef:
        int size = dEdz.shape[1]
        np.ndarray[REAL_t, ndim=2] delta_unbiased = np.zeros_like(input, dtype=REAL)
        np.ndarray[REAL_t, ndim=3] outer_dotted   = vector_outer_product(dEdz, input)
    
    for i in range(size):
        delta_unbiased += np.dot(tensor[i,:,:], outer_dotted[i,:,:]).T
        delta_unbiased += np.dot(tensor[i,:,:].T, outer_dotted[i,:,:]).T
    
    return delta_unbiased