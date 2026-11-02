"""Haar matrix and global/local t-SVD bases (paper Sec. II-D, III-B)."""
import numpy as np


def haar_matrix(n):
    """Orthonormal n x n Haar matrix built recursively with Eq. (8). n must be a power of 2."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"Haar matrix needs a power of two, got {n}")
    H = np.ones((1, 1))
    while H.shape[0] < n:
        m = H.shape[0]
        H = np.vstack([np.kron(H, [1.0, 1.0]), np.kron(np.eye(m), [1.0, -1.0])]) / np.sqrt(2.0)
    return H


def channel_fft(P):
    """Orthonormal FFT along the channel axis (axis -3) of (..., c, ps, ps) patches.

    Only the non-redundant slices are kept (rfft); for c = 3 these are slice 0 (real)
    and slice 1 (complex, its conjugate is slice 2).
    """
    return np.fft.rfft(P, axis=-3, norm="ortho")


def channel_ifft(F, c):
    return np.fft.irfft(F, n=c, axis=-3, norm="ortho")


def _bases_from_correlations(C_row, C_col):
    # eigh returns ascending eigenvalues; order columns by descending energy.
    _, U = np.linalg.eigh(C_row)
    _, V = np.linalg.eigh(C_col)
    return U[..., ::-1], V[..., ::-1]


def learn_tsvd_bases(P):
    """Globally learned t-SVD bases from a set of patches (Eq. (11)).

    P: (N, c, ps, ps) real patches.
    Returns U, V of shape (n_slices, ps, ps) (complex), one pair per Fourier slice.
    """
    F = channel_fft(P.astype(np.float64))  # (N, s, ps, ps)
    C_row = np.einsum("nsab,nscb->sac", F, F.conj())  # sum_i P P^H
    C_col = np.einsum("nsba,nsbc->sac", F.conj(), F)  # sum_i P^H P
    U, V = _bases_from_correlations(C_row, C_col)
    # Self-conjugate slices (DC, and Nyquist for even c) are real: keep their bases real so the
    # whole projection is a real orthogonal map (needed by the GPU path).
    c = P.shape[-3]
    for j in {0, c // 2} if c % 2 == 0 else {0}:
        Ur, Vr = _bases_from_correlations(C_row[j].real, C_col[j].real)
        U[j], V[j] = Ur, Vr
    return U, V


def learn_local_bases(F):
    """Per-group (locally learned) t-SVD bases.

    F: (B, K, s, ps, ps) Fourier-domain groups. Returns U, V of shape (B, s, ps, ps).
    """
    C_row = np.einsum("bksxy,bkszy->bsxz", F, F.conj())
    C_col = np.einsum("bksyx,bksyz->bsxz", F.conj(), F)
    return _bases_from_correlations(C_row, C_col)


def project(F, U, V):
    """S = U^H P V for each Fourier slice. U, V broadcast against F's leading axes."""
    return np.conj(np.swapaxes(U, -1, -2)) @ F @ V


def unproject(S, U, V):
    """P = U S V^H."""
    return U @ S @ np.conj(np.swapaxes(V, -1, -2))
