# Deep Learning for Robust Normal Estimation in Unstructured Point Clouds
# Copyright (c) 2016 Alexande Boulch and Renaud Marlet
#
# This program is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street,
# Fifth Floor, Boston, MA 02110-1301  USA
#
# PLEASE ACKNOWLEDGE THE ORIGINAL AUTHORS AND PUBLICATION:
# "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
# by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016,
# Computer Graphics Forum

# distutils: language = c++
# distutils: sources = houghCNN.cxx

import numpy as np
cimport numpy as np
import cython
from libcpp.string cimport string



cdef extern from "houghCNN.h":
    cdef cppclass NormEst:

        NormEst()

        void loadXYZ(string)
        void saveXYZ(string)

        void getPoints(double*, int,int)
        void setPoints(double*, int, int)
        void getNormals(double*, int,int)
        void setNormals(double*, int, int)

        int getPCSize()
        int getPCNormalsSize()

        int getT()
        void setT(int)

        int getA()
        void setA(int)

        int getDensitySensitive()
        void setDensitySensitive(bool)

        int getKaniso()
        void setKaniso(int)

        void initialize()
        void getBatch(int,int, double*)
        void setBatch(int,int, double*)

        void getKs(int*,int)
        void setKs(int*, int)
        int getKsSize()

        int generateTrainAccRandomCorner(int,int, double*, double*)



cdef class NormalEstimatorHoughCNN:
    cdef NormEst *thisptr

    def __cinit__(self):
        self.thisptr = new NormEst()

    def __dealloc__(self):
        del self.thisptr

    cpdef loadXYZ(self, filename):
        self.thisptr.loadXYZ(str.encode(filename))

    cpdef saveXYZ(self,filename):
        self.thisptr.saveXYZ(str.encode(filename))

    cpdef getPCSize(self):
        return self.thisptr.getPCSize()

    cpdef getPCNormalsSize(self):
        return self.thisptr.getPCNormalsSize()

    cpdef getT(self):
        return self.thisptr.getT()
    cpdef setT(self, T):
        self.thisptr.setT(T)

    cpdef getA(self):
        return self.thisptr.getA()
    cpdef setA(self, A):
        self.thisptr.setA(A)

    cpdef getDensitySensitive(self):
        return self.thisptr.getDensitySensitive()
    cpdef setDensitySensitive(self, d_s):
        self.thisptr.setDensitySensitive(d_s)

    cpdef get_K_density(self):
        return self.thisptr.getKaniso()
    cpdef set_K_density(self, K_d):
        self.thisptr.setKaniso(K_d)

    def getKs(self):
        cdef m
        m = self.getKsSize()
        d = np.zeros(m, dtype = np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] d2 = d
        self.thisptr.getKs(<int *> d2.data, m)
        return d

    def getKsSize(self):
        return self.thisptr.getKsSize()

    def setKs(self, Ks):
        cdef np.ndarray[np.int32_t, ndim = 1] d2 = Ks.astype(np.int32)
        self.thisptr.setKs(<int *> d2.data, Ks.shape[0])

    def getPoints(self):
        cdef int m, n
        m = self.getPCSize()
        n = 3
        d = np.zeros((m,n),dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = d
        self.thisptr.getPoints(<double *> d2.data, m,n)
        return d

    def getNormals(self):
        cdef int m, n
        m = self.getPCSize()
        n = 3
        d = np.zeros((m,n),dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = d
        self.thisptr.getNormals(<double *> d2.data, m,n)
        return d

    def initialize(self):
        self.thisptr.initialize()

    def getBatch(self, pt_id, batch_size):
        cdef int ptid, bs, ks, A
        ptid = pt_id
        bs=batch_size
        ks= self.getKsSize()
        A = self.getA()
        d = np.zeros((bs,ks,A,A), dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 4] d2 = d
        self.thisptr.getBatch(ptid, bs, <double *> d2.data)
        return d

    def setBatch(self, pt_id, batch_size, batch):
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = batch
        self.thisptr.setBatch(pt_id, batch_size, <double *> d2.data)


    def setPoints(self, points):
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = points
        self.thisptr.setPoints(<double *> d2.data, points.shape[0], points.shape[1])

    def setNormals(self, normals):
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = normals
        self.thisptr.setNormals(<double *> d2.data, normals.shape[0], normals.shape[1])

    def generateTrainAccRandomCorner(self, n_points, noise_val=-1):
        cdef int npt, ks, A, nv
        npt = n_points
        nv = noise_val
        ks= self.getKsSize()
        A = self.getA()
        d = np.zeros((npt,ks,A,A), dtype=np.double)
        t = np.zeros((npt,2), dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 4] d2 = d
        cdef np.ndarray[np.float64_t, ndim = 2] t2 = t
        nbr = self.thisptr.generateTrainAccRandomCorner(nv,npt, <double *> d2.data, <double *> t2.data)
        return nbr, d,t
