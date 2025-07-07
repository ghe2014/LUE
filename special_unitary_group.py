#  Copyright (c) 2025.  Guangliang He
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  haha If not, see <http://www.gnu.org/licenses/>.
import warnings
from collections import namedtuple
from itertools import product
import numpy as np
from bidict import bidict
from scipy.linalg import schur, block_diag, expm

GroupStructure = namedtuple("GroupStructure", ["T", "F", "f", "d"])


def from_adjoint(R, atol=1e-8):
    """
    Take an input R in Ad(SU(N) subset of SO(N^2-1), compute U.
    None is returned if R is not in Ad(SU(N))
    :param R: numpy real square matrix in SO(N^2-1)
    :param atol: absolute tolerance for comparing to zero
    :return: numpy array in SU(N) or None

    Note: The returned U is one of N possible values
          Since for all z in Z(SU(N)), U*z have the same Adjoint rep.
    """

    M = R.shape[0]
    N = round(np.sqrt(M+1))
    if N*N-1 != M:  # R doesn't even have the right dimension
        return None

    if not np.allclose(R.T@R-np.eye(M), 0, atol=atol, rtol=0):  # Not in SO(M)
        return None

    T, Z = schur(R, output="real")  # get the canonical form of SO(M)
    T_list, Z1 = sort_matrix_list(extract_diagonal_blocks(T, [1, 2]))  # get the sorted 2x2 and 1x1 blocks
    Z = Z@Z1

    # check if we have at least N-1 +1's at the end of the list
    if len(T_list) < N-1:
        return None  # doesn't have N-1 blocks of 1.

    for block in T_list[-(N-1):]:
        if block.shape != (1, 1) or not np.isclose(block[0,0]-1, 0, atol):
            return None  # not enough 1's

    if N > 2:  # might need to take care of degenerate 2x2 blocks.
        warnings.warn("Might need to take care of degeneracy")

    phi_list = []
    i = 0
    while i < len(T_list)-(N-1):
        m = T_list[i]
        if m.shape == (2, 2):
            phi_list.append(R2phi(m))
            i += 1
        else: # the shape is (1, 1)
            m1 = T_list[i+1]
            if np.isclose(m-1, 0, atol=atol, rtol=0) and np.isclose(m1-1, 0, atol=atol, rtol=0):
                phi_list.append(0)
            elif np.isclose(m+1, 0,  atol=atol, rtol=0) and np.isclose(m1+1, 0, atol=atol, rtol=0):
                phi_list.append(np.pi)
            else:
                return None # we shouldn't be here
            i += 2

    phi_vec = np.array(phi_list)
    su_N = get_group_structure(N)

    # tildeF_a = Z.T@F_a@Z
    tildeF = np.einsum("ij,ajk,kl->ail", Z.T, su_N.F, Z)

    Y = np.array([[0, -1j], [1j, 0]])
    ZeroN1N1 = np.zeros((N-1,N-1))
    for k in product([0, 1, -1], repeat=phi_vec.shape[0]):
        phi_vec_k = phi_vec + 2*np.pi*np.array(k)
        if np.max(np.abs(phi_vec_k)) >= np.pi*2:
            continue  # we don't need to process this k
        logT_list = [phi_vec_k[i]*Y for i in range(phi_vec.shape[0])]
        logT_list.append(ZeroN1N1)
        logT = block_diag(*logT_list)

        # project logT to tildeF
        theta = np.trace(np.einsum("ij,ajk->ika", logT, tildeF))/N
        if np.allclose(np.einsum("i,ijk->jk", theta, tildeF)-logT, 0, atol=atol, rtol=0):
            return expm(1j*np.einsum("i,ijk->jk", theta, su_N.T))

    return None


def to_adjoint(U):
    """
    Take input U in SU(N), computer Ad_U
    Ad_U[i,j] = 2*Tr(T[i]@U@T[j]@U^dagger)
    :param U:
    :return: Ad_U
    """
    N = U.shape[0]
    R = np.empty((N*N-1, N*N-1))  # R is real
    su_N = get_group_structure(N)
    for j in range(N*N-1):
        temp = U@su_N.T[j]@U.conj().T
        for i in range(N*N-1):
            R[i, j] = 2*np.trace(su_N.T[i]@temp).real

    return R

def basis_projection_to_matrix(p, group_dims, adjoint=False):
    """
    construct matrix from its basis projection
    :param p: input projection tensor - mode K
    :param group_dims: dimensions of subspaces.
                       little ending convention, group_dims[0] is the dimension for the
                       right most subspace
    :param adjoint: boolean, if true, construct in the adjoint rep
    :return: a matrix M = sum_{i1,...iK}P[i1,...iK](T_i1 otimes ... otimes T_iK)
    """
    basis_list = []
    m_size = 1
    for n in group_dims:
        if adjoint:
            d = n*n-1
            scale = np.sqrt(n/(n*n-1))
        else:
            d = n
            scale = 1/np.sqrt(2*n)

        m_size *= d

        su_n = get_group_structure(n)
        T_array = np.empty((n*n, d, d), dtype=complex)
        T_array[0] = scale*np.eye(d)
        if adjoint:
            T_array[1:] = su_n.F
        else:
            T_array[1:] = su_n.T
        basis_list.append(T_array)

    ranges = [range(n*n) for n in group_dims]
    M = np.zeros((m_size, m_size), dtype=complex)
    for indices in product(*ranges):
        G = 1
        for k in range(len(group_dims)):
            G = np.kron(basis_list[k][indices[k]], G)
        M += p[indices]*G

    return M

def matrix_to_basis_projection(m, group_dims, adjoint=False):
    """
    project matrix m
    :param m: a square matrix
    :param group_dims: dimensions of subspaces.
                 Little ending convention, group_dims[0] is the dimension for the
                 right most subspace.
    :param adjoint: boolean, if true, project on to the basis of the adjoint rep
                             otherwise, project on to the basis of the defining rep
    :return: a projection tensor of dimensions (d[0], d[1], ..., d[K-1])
             if adjoint, project_tensor[i_0, ..., i_{K-1}] =
                 Tr(m@(T^{(K-1)}_{i_{K-1}} otimes ... otimes T^{(0)}_{i_0}))
             else
                 Tr(m@(F^{(K-1)}_{i_{K-1}} otimes ... otimes F^{(0)}_{i_0}))

             Note, for SU(N), T_0 = I/sqrt(2N), T_1... are the regular generators
                              F_0 = sqrt(N/(N*N-1))I, F_1... are the regular generators
    """

    # SU(n) has n*n-1 generators, plus the identity, total n*n basis operators
    basis_list = []
    for n in group_dims:
        if adjoint:
            d = n*n-1
            scale = np.sqrt(n/(n*n-1))
        else:
            d = n
            scale = 1/np.sqrt(2*n)
        T_array = np.empty((n*n, d, d), dtype=complex)
        T_array[0] = scale*np.eye(d)
        su_n = get_group_structure(n)
        if adjoint:
            T_array[1:] = su_n.F
        else:
            T_array[1:] = su_n.T
        basis_list.append(T_array)

    K = len(group_dims)
    # proj_tensor contains the projection coefficients of the matrix
    # onto the generators
    proj_tensor = np.empty([n*n for n in group_dims], dtype=complex)
    ranges = [range(n*n) for n in group_dims]
    for indices in product(*ranges):
        G = 1
        for i in range(K):
            G = np.kron(basis_list[i][indices[i]], G)
        proj_tensor[indices] = np.trace(m@G)

    if adjoint:
        proj_tensor /= np.prod(group_dims)
    else:
        proj_tensor *= 2**K

    return proj_tensor

def get_index_bidict(N):
    """
    The generalized Gell-Mann matrices can be indexed either sequentially (1 - N^2-1)
    or by matrix properties, for diagonal ones, index by l, the number of positive
    diagonal elements.  for non-diagonal ones, by (a, i, j)
    with a = 0 (symmetrical), 1 (antisymmetrical), 1 <= i < j <= N
    :param N:
    :return: a bidict takes keys in the form of (a, i, j) or l and values between 1 - N^2-1.

    Note: indices are 1 based.  Adjustment is needed to index python arrays.
    """
    bd = bidict()
    seq = 1
    for i in range(N):
        for j in range(i+1, N):
            bd[(0, i+1, j+1)] = seq  # symmetrical
            bd[(1, i+1, j+1)] = seq+1  # anti-symmetrical

            seq += 2

    for l in range(1, N):
        bd[l] = seq  # diagonal

        seq += 1

    return bd

def get_diagonal_ggm_indices(N, base1=True):
    """
    return the (base-0) indices of diagonal generalized Gell-Mann matrices
    :param N: matrix dimension
    :param base1: boolean, if true, return indices is 1 based, otherwise 0 based
    :return: a list of 1-based indices for diagonal generalized gell-mann matrices
    """
    idx_bd = get_index_bidict(N)
    base = 0
    if base1:
        base = 1
    return [idx_bd[l]-(1-base) for l in range(1, N)]

def get_index_mapping(N):
    """
    The basis {T_a} of su(N) is Lambda_a/2
    {Lambda_a} can be grouped into (0, i, j), (1, i, j), (l)
    where 1 <= i < j <= N, 1 <= l < N
    so the set {a} is mapped into a list of triplets or singletons
    :param N:
    :return: a dict, with keys of (0, i, j), (1, i, j), (l) and values of a

    Note:  i, j, l are 1 based and a is 0 based.
    """
    su_n = get_group_structure(N)
    index_map = dict()
    for a in range(N*N-1):
        t = su_n.T[a]
        indices = np.nonzero(t)  # tuple of arrays of nonzero indices
        i = indices[0][0]
        j = indices[1][0]
        if i == j:
            # diagonal
            l = len(indices[0])-1
            index_map[l] = a
        else:
            sa = int(np.iscomplex(t[i, j]))
            index_map[(sa, i+1, j+1)] = a  # i, j are 0 based, adjust to 1 based

    return index_map

def get_group_structure(N):
    """
    get SU(N) group structure
    :param N: int, dimension of the special unitary group
    :return: named tuple of
       T: the generators in the defining representation,
          (N*N-1)xNxN numpy array
       F: the generators in the adjoint representation,
          (N*N-1)x(N*N-1)x(N*N-1) numpy array
       f: the anti-symmetrical structure constants
          NxNxN numpy array
       d: the symmetrical structure constants
          NxNxN numpy array

    Ref:
    [1] Useful relations among the generators in the defining
        and adjoint representation of SU(N) (2021)
        Haber, Howard E.
        https://arxiv.org/pdf/1912.13302
    """

    # the generators in the defining representation
    T = 0.5*generalized_gell_mann(N)
    T.setflags(write=False)  # no one should change it

    # the structure constant, real array
    f = np.empty((N*N-1, N*N-1, N*N-1))
    d = np.empty((N*N-1, N*N-1, N*N-1))
    for a in range(N*N-1):
        for b in range(N*N-1):
            for c in range(N*N-1):
                f[a, b, c] = (-2j*np.trace(((T[a, :, :] @ T[b, :, :] - T[b, :, :] @ T[a, :, :]) @ T[c, :, :]))).real
                d[a, b, c] = (2*np.trace(((T[a, :, :] @ T[b, :, :] + T[b, :, :] @ T[a, :, :]) @ T[c, :, :]))).real

    f.setflags(write=False)
    d.setflags(write=False)

    # the generators in the adjoint representation
    F = -1j*f
    F.setflags(write=False)

    return GroupStructure(T=T, F=F, f=f, d=d)

def generalized_gell_mann(N):
    """
    Returns Generalized Gell-mann matrices
    :param N: int, the dimension N of the SU(N)
    :return: NxNxN array.
      The ggm[i-1, :, :] is the i-th generalized Gell-Mann matrix

    Note: the generalized Gell-Mann matrices have a different order
          from the Gell-Mann matrix in the case of SU(3).

    Ref:
    [1] Bloch vectors for qudits (2028)
        Bertlmann, Reinhold A. and Krammer, Philipp
        https://arxiv.org/pdf/0806.1174
    """

    ggm = np.zeros((N*N-1, N, N), dtype=complex)

    index_bidict = get_index_bidict(N)  # 1 based index bidict

    for key, seq in index_bidict.items():
        if isinstance(key, tuple):  # either symmetrical or antisymmetrical
            sa, i, j = key
            if sa == 0:  # symmetrical
                ggm[seq-1, i-1, j-1] = 1
                ggm[seq-1, j-1, i-1] = 1
            else:  # antisymmetrical
                ggm[seq-1, i-1, j-1] = -1j
                ggm[seq-1, j-1, i-1] = 1j
        else:  # diagonal
            l = key
            scale = np.sqrt(2/(l*(l+1)))
            ggm[seq-1, :l, :l] = scale*np.eye(l)
            ggm[seq-1, l, l] = -l*scale

    ggm.setflags(write=False)
    return ggm

def extract_diagonal_blocks(T, block_sizes, atol=1e-10, copy_blocks=True):
    """
    Take a block diagonal matrix of possible block sizes,
    extract the diagonal blocks into a list
    :param T: a numpy array for the input square matrix
    :param block_sizes: list of positive integers, possible block sizes
    :param atol: positive float, any number with absolute less than atol is considered as 0
    :param copy_blocks: logical, if True, the diagonal blocks are copied to the output list
                                 otherwise, referenced
    :return: a list of diagonal blocks
    """

    # input validation
    if not isinstance(T, np.ndarray) or T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be a square 2D numpy array")
    if not block_sizes or not all(isinstance(s, int) and s > 0 for s in block_sizes):
        raise ValueError("block_sizes must be a list of positive integer")

    sorted_block_sizes = sorted(set(block_sizes))  # remove duplicates and sort
    n = T.shape[0]
    diagonal_blocks = []
    abs_T = np.abs(T)

    start_idx = 0
    while start_idx < n:
        block_found = False
        for s in sorted_block_sizes:
            end_idx = start_idx + s
            if end_idx> n:
                break  # break out the look and raise error

            is_zero_below = (end_idx == n) or np.max(abs_T[end_idx:, start_idx:end_idx]) < atol
            is_zero_right = (end_idx == n) or np.max(abs_T[start_idx:end_idx, end_idx:]) < atol
            if is_zero_below and is_zero_right:
                block_found = True
                block = T[start_idx:end_idx, start_idx:end_idx]
                diagonal_blocks.append(block.copy() if copy_blocks else block)
                start_idx += s
                break
        if not block_found:
            remaining = n - start_idx
            valid_sizes = [s for s in sorted_block_sizes if s <= remaining]
            raise ValueError(f"No suitable blocks found after ({start_idx},{start_idx}). "
                             f"Remaining matrix size: {remaining}x{remaining}. "
                             f"Valid block sizes: {valid_sizes}. ")

    return diagonal_blocks

def sort_matrix_list(m_list):
    """
    Given a list of square matrices, sort them to the sizes of the matrices (descending order),
    and for 1x1 matrices, sort in the ascending order of the real part.
    :param m_list: list of square matrices
    :return: m_list_new, sorted list
             Z, permutation matrix, such that Z@block_diag(*m_list_new)@Z.T = block_diag(*m_list)
    """

    # tag the matrices with locations
    s_idx = 0
    matrix_info = []
    for i, m in enumerate(m_list):
        matrix_info.append((m, i, s_idx))
        s_idx += m.shape[0]

    Z = np.zeros((s_idx, s_idx))

    # first, sort the dimension of the block descending,
    # then sort the real parts of [0,0]-th elements, ascending,
    # finally, sort the real parts of [0,-1]-th elements, ascending.
    matrix_info_sorted = sorted(matrix_info,
                                key=lambda x: (-x[0].shape[0], x[0][0,0].real, x[0][0,-1].real))

    m_list_new = []
    s_idx = 0
    for info in matrix_info_sorted:
        m = info[0]
        i = info[1]
        m_list_new.append(m)
        Z[matrix_info[i][2]:(matrix_info[i][2]+m.shape[0]), s_idx:(s_idx+m.shape[0])] = np.eye(m.shape[0])
        s_idx += m.shape[0]

    return m_list_new, Z

def R2phi(R):
    """
    Take a SO(2) matrix R and compute the rotation angle phi
    :param R:
    :return: phi in (-pi, pi]

    Note, for any integer k, phi+2*k*pi is also a solution
    """
    # R = [[cos(phi), sin(phi)], [-sin(phi), cos(phi)]]
    #
    return np.arctan2(R[0, 1], R[0, 0])
