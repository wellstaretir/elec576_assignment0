# %% Task 2: NumPy linear algebra equivalents
import numpy as np
import scipy.linalg
from scipy import linalg, signal
from scipy.sparse.linalg import cg, eigs

np.set_printoptions(precision=6, suppress=True, linewidth=88, threshold=np.inf)


# %% Task 2 - 01. Array dimensions
a = np.arange(1, 55).reshape(6, 9)
print('np.ndim(a) =')
print(np.ndim(a))
print('a.ndim =')
print(a.ndim)


# %% Task 2 - 02. Element count
a = np.arange(1, 55).reshape(6, 9)
print('np.size(a) =')
print(np.size(a))
print('a.size =')
print(a.size)


# %% Task 2 - 03. Array shape
a = np.arange(1, 55).reshape(6, 9)
print('np.shape(a) =')
print(np.shape(a))
print('a.shape =')
print(a.shape)


# %% Task 2 - 04. Length of a selected axis
a = np.arange(1, 55).reshape(6, 9)
n = 2
print('a.shape[n-1] =')
print(a.shape[n-1])


# %% Task 2 - 05. Array construction
print('np.array([[1., 2., 3.], [4., 5., 6.]]) =')
print(np.array([[1., 2., 3.], [4., 5., 6.]]))


# %% Task 2 - 06. Block construction
a = np.array([[1, 2]])
b = np.array([[3]])
c = np.array([[4, 5]])
d = np.array([[6]])
print('np.block([[a, b], [c, d]]) =')
print(np.block([[a, b], [c, d]]))


# %% Task 2 - 07. Last vector entry
a = np.array([10, 20, 30, 40])
print('a[-1] =')
print(a[-1])


# %% Task 2 - 08. Entry at row 2, column 5
a = np.arange(1, 55).reshape(6, 9)
print('a[1, 4] =')
print(a[1, 4])


# %% Task 2 - 09. Second row
a = np.arange(1, 55).reshape(6, 9)
print('a[1] =')
print(a[1])
print('a[1, :] =')
print(a[1, :])


# %% Task 2 - 10. First five rows
a = np.arange(1, 55).reshape(6, 9)
print('a[0:5] =')
print(a[0:5])
print('a[:5] =')
print(a[:5])
print('a[0:5, :] =')
print(a[0:5, :])


# %% Task 2 - 11. Last five rows
a = np.arange(1, 55).reshape(6, 9)
print('a[-5:] =')
print(a[-5:])


# %% Task 2 - 12. Rectangular slice
a = np.arange(1, 55).reshape(6, 9)
print('a[0:3, 4:9] =')
print(a[0:3, 4:9])


# %% Task 2 - 13. Nonconsecutive rows and columns
a = np.arange(1, 55).reshape(6, 9)
print('a[np.ix_([1, 3, 4], [0, 2])] =')
print(a[np.ix_([1, 3, 4], [0, 2])])


# %% Task 2 - 14. Rows 3, 5, ..., 21
a = np.arange(1, 64).reshape(21, 3)
print('a[2:21:2, :] =')
print(a[2:21:2, :])


# %% Task 2 - 15. Every other row
a = np.arange(1, 55).reshape(6, 9)
print('a[::2, :] =')
print(a[::2, :])


# %% Task 2 - 16. Reverse row order
a = np.arange(1, 55).reshape(6, 9)
print('a[::-1, :] =')
print(a[::-1, :])


# %% Task 2 - 17. Append the first row
a = np.arange(1, 55).reshape(6, 9)
print('a[np.r_[:len(a), 0]] =')
print(a[np.r_[:len(a), 0]])


# %% Task 2 - 18. Transpose
a = np.array([[1+2j, 3-1j], [4j, 5+0j]])
print('a.transpose() =')
print(a.transpose())
print('a.T =')
print(a.T)


# %% Task 2 - 19. Conjugate transpose
a = np.array([[1+2j, 3-1j], [4j, 5+0j]])
print('a.conj().transpose() =')
print(a.conj().transpose())
print('a.conj().T =')
print(a.conj().T)


# %% Task 2 - 20. Matrix multiplication
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
b = np.array([[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]])
print('a @ b =')
print(a @ b)


# %% Task 2 - 21. Elementwise multiplication
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
b = np.array([[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]])
print('a * b =')
print(a * b)


# %% Task 2 - 22. Elementwise division
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
b = np.array([[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]])
print('a / b =')
print(a / b)


# %% Task 2 - 23. Elementwise cube
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
print('a**3 =')
print(a**3)


# %% Task 2 - 24. Threshold comparison
a = np.array([[0.2, 0.5, 0.8], [0.9, 0.1, 0.6]])
print('a > 0.5 =')
print(a > 0.5)


# %% Task 2 - 25. Indices satisfying a condition
a = np.array([[0.2, 0.5, 0.8], [0.9, 0.1, 0.6]])
print('np.nonzero(a > 0.5) =')
print(np.nonzero(a > 0.5))


# %% Task 2 - 26. Select columns using nonzero
a = np.array([[0.2, 0.5, 0.8], [0.9, 0.1, 0.6]])
v = np.array([0.2, 0.7, 0.9])
print('a[:, np.nonzero(v > 0.5)[0]] =')
print(a[:, np.nonzero(v > 0.5)[0]])


# %% Task 2 - 27. Select columns using a Boolean mask
a = np.array([[0.2, 0.5, 0.8], [0.9, 0.1, 0.6]])
v = np.array([0.2, 0.7, 0.9])
print('a[:, v.T > 0.5] =')
print(a[:, v.T > 0.5])
v_col = v[:, None]
print("Explicit (3,1) column-vector version:")
print(a[:, v_col.ravel() > 0.5])


# %% Task 2 - 28. Set entries below the threshold to zero
a = np.array([[0.2, 0.5, 0.8], [0.9, 0.1, 0.6]])
a[a < 0.5] = 0
print(a)


# %% Task 2 - 29. Mask by multiplication
a = np.array([[0.2, 0.5, 0.8], [0.9, 0.1, 0.6]])
print('a * (a > 0.5) =')
print(a * (a > 0.5))


# %% Task 2 - 30. Fill an existing array with a scalar
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
a[:] = 3
print(a)


# %% Task 2 - 31. Copy a complete array
x = np.array([[1, 2, 3], [4, 5, 6]])
y = x.copy()
print(y)
print("shares_memory:", np.shares_memory(x, y))


# %% Task 2 - 32. Copy a selected row
x = np.array([[1, 2, 3], [4, 5, 6]])
y = x[1, :].copy()
print(y)
print("shares_memory:", np.shares_memory(x, y))


# %% Task 2 - 33. Flatten to a copied vector
x = np.array([[1, 2, 3], [4, 5, 6]])
y = x.flatten()
print("C order:", y)
print("F order:", x.flatten("F"))
print("shares_memory:", np.shares_memory(x, y))


# %% Task 2 - 34. Range from 1 through 10
print('np.arange(1., 11.) =')
print(np.arange(1., 11.))
print('np.r_[1.:11.] =')
print(np.r_[1.:11.])
print('np.r_[1:10:10j] =')
print(np.r_[1:10:10j])


# %% Task 2 - 35. Range from 0 through 9
print('np.arange(10.) =')
print(np.arange(10.))
print('np.r_[:10.] =')
print(np.r_[:10.])
print('np.r_[:9:10j] =')
print(np.r_[:9:10j])


# %% Task 2 - 36. Column vector from a range
print('np.arange(1., 11.)[:, np.newaxis] =')
print(np.arange(1., 11.)[:, np.newaxis])


# %% Task 2 - 37. Two-dimensional zeros
print('np.zeros((3, 4)) =')
print(np.zeros((3, 4)))


# %% Task 2 - 38. Three-dimensional zeros
print('np.zeros((3, 4, 5)) =')
print(np.zeros((3, 4, 5)))


# %% Task 2 - 39. Two-dimensional ones
print('np.ones((3, 4)) =')
print(np.ones((3, 4)))


# %% Task 2 - 40. Identity matrix
print('np.eye(3) =')
print(np.eye(3))


# %% Task 2 - 41. Extract a diagonal
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
print('np.diag(a) =')
print(np.diag(a))


# %% Task 2 - 42. Construct a diagonal matrix
v = np.array([2, 4, 6])
print('np.diag(v, 0) =')
print(np.diag(v, 0))


# %% Task 2 - 43. Reproducible random array
from numpy.random import default_rng
rng = default_rng(42)
print('rng.random((3, 4)) =')
print(rng.random((3, 4)))
np.random.seed(42)
print("Legacy API, corrected argument form:")
print(np.random.rand(3, 4))


# %% Task 2 - 44. Equally spaced samples
print('np.linspace(1, 3, 4) =')
print(np.linspace(1, 3, 4))


# %% Task 2 - 45. Dense grids
print('np.mgrid[0:9., 0:6.] =')
print(np.mgrid[0:9., 0:6.])
print('np.meshgrid(np.r_[0:9.], np.r_[0:6.]) =')
print(np.meshgrid(np.r_[0:9.], np.r_[0:6.]))


# %% Task 2 - 46. Open grids
print('np.ogrid[0:9., 0:6.] =')
print(np.ogrid[0:9., 0:6.])
print('np.ix_(np.r_[0:9.], np.r_[0:6.]) =')
print(np.ix_(np.r_[0:9.], np.r_[0:6.]))


# %% Task 2 - 47. Dense grids with nonuniform coordinates
print('np.meshgrid([1, 2, 4], [2, 4, 5]) =')
print(np.meshgrid([1, 2, 4], [2, 4, 5]))


# %% Task 2 - 48. Open grids with nonuniform coordinates
print('np.ix_([1, 2, 4], [2, 4, 5]) =')
print(np.ix_([1, 2, 4], [2, 4, 5]))


# %% Task 2 - 49. Tile an array
a = np.array([[1, 2], [3, 4]])
m, n = 2, 3
print('np.tile(a, (m, n)) =')
print(np.tile(a, (m, n)))


# %% Task 2 - 50. Horizontal concatenation
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print('np.concatenate((a, b), 1) =')
print(np.concatenate((a, b), 1))
print('np.hstack((a, b)) =')
print(np.hstack((a, b)))
print('np.column_stack((a, b)) =')
print(np.column_stack((a, b)))
print('np.c_[a, b] =')
print(np.c_[a, b])


# %% Task 2 - 51. Vertical concatenation
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print('np.concatenate((a, b)) =')
print(np.concatenate((a, b)))
print('np.vstack((a, b)) =')
print(np.vstack((a, b)))
print('np.r_[a, b] =')
print(np.r_[a, b])


# %% Task 2 - 52. Overall maximum
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
print('a.max() =')
print(a.max())
print('np.nanmax(a) =')
print(np.nanmax(a))
a_nan = np.array([1., np.nan, 4.])
print("NaN example:", np.nanmax(a_nan))


# %% Task 2 - 53. Column maxima
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
print('a.max(0) =')
print(a.max(0))


# %% Task 2 - 54. Row maxima
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
print('a.max(1) =')
print(a.max(1))


# %% Task 2 - 55. Pairwise maxima
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
b = np.array([[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]])
print('np.maximum(a, b) =')
print(np.maximum(a, b))


# %% Task 2 - 56. Euclidean norm
v = np.array([3., 4.])
print('np.sqrt(v @ v) =')
print(np.sqrt(v @ v))
print('np.linalg.norm(v) =')
print(np.linalg.norm(v))


# %% Task 2 - 57. Logical AND
a = np.array([0, 1, 3, 4])
b = np.array([1, 0, 4, 3])
print('np.logical_and(a, b) =')
print(np.logical_and(a, b))


# %% Task 2 - 58. Logical OR
a = np.array([0, 1, 3, 4])
b = np.array([1, 0, 4, 3])
print('np.logical_or(a, b) =')
print(np.logical_or(a, b))


# %% Task 2 - 59. Bitwise AND
a = np.array([0, 1, 3, 4])
b = np.array([1, 0, 4, 3])
print('a & b =')
print(a & b)


# %% Task 2 - 60. Bitwise OR
a = np.array([0, 1, 3, 4])
b = np.array([1, 0, 4, 3])
print('a | b =')
print(a | b)


# %% Task 2 - 61. Matrix inverse
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
print('linalg.inv(a) =')
print(linalg.inv(a))


# %% Task 2 - 62. Moore-Penrose pseudoinverse
a = np.array([[1., 2.], [2., 4.], [3., 6.]])
print('linalg.pinv(a) =')
print(linalg.pinv(a))


# %% Task 2 - 63. Matrix rank
a = np.array([[1., 2.], [2., 4.], [3., 6.]])
print('np.linalg.matrix_rank(a) =')
print(np.linalg.matrix_rank(a))


# %% Task 2 - 64. Left division: square and rectangular systems
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
b = np.array([1., 2., 3.])
print('linalg.solve(a, b) =')
print(linalg.solve(a, b))
a = np.array([[1., 0.], [1., 1.], [1., 2.], [1., 3.]])
b = np.array([1., 2., 2., 4.])
print("Rectangular lstsq: solution, residual sum, rank, singular values")
print(linalg.lstsq(a, b))


# %% Task 2 - 65. Right division through a transposed solve
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
b = np.array([[1., 2., 3.], [4., 5., 6.]])
x = linalg.solve(a.T, b.T).T
print(x)
print("x @ a =")
print(x @ a)


# %% Task 2 - 66. Singular value decomposition
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
U, S, Vh = linalg.svd(a)
V = Vh.T
print("U ="); print(U)
print("S ="); print(S)
print("Vh ="); print(Vh)
print("V ="); print(V)


# %% Task 2 - 67. Cholesky decomposition
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
print('linalg.cholesky(a) =')
print(linalg.cholesky(a))


# %% Task 2 - 68. Ordinary eigenproblem
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
D, V = linalg.eig(a)
print("D ="); print(D)
print("V ="); print(V)


# %% Task 2 - 69. Generalized eigenproblem
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
b = np.diag([2., 3., 4.])
D, V = linalg.eig(a, b)
print("D ="); print(D)
print("V ="); print(V)


# %% Task 2 - 70. Three selected eigenpairs
a = np.diag(np.arange(1., 7.))
D, V = eigs(a, k=3, v0=np.ones(6))
print("D ="); print(D)
print("V ="); print(V)


# %% Task 2 - 71. QR decomposition
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
Q, R = linalg.qr(a)
print("Q ="); print(Q)
print("R ="); print(R)
a_rect = np.arange(1., 9.).reshape(4, 2)
Qe, Re = linalg.qr(a_rect, mode="economic")
print("Rectangular economy shapes:", Qe.shape, Re.shape)


# %% Task 2 - 72. LU decomposition
a = np.array([[0., 2., 1.], [1., 1., 0.], [2., 1., 1.]])
P, L, U = linalg.lu(a)
print("P ="); print(P)
print("L ="); print(L)
print("U ="); print(U)
print("P @ L @ U ="); print(P @ L @ U)


# %% Task 2 - 73. Conjugate-gradient solution
a = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., 2.]])
b = np.array([1., 2., 3.])
x, info = cg(a, b, rtol=1e-12, atol=0.0)
print("x =", x)
print("info =", info)


# %% Task 2 - 74. Forward discrete Fourier transform
a = np.array([1., 2., 3., 4.])
print('np.fft.fft(a) =')
print(np.fft.fft(a))
a2 = np.array([[1., 2.], [3., 4.], [5., 6.]])
print("2D example, transform along columns:")
print(np.fft.fft(a2, axis=0))


# %% Task 2 - 75. Inverse discrete Fourier transform
a = np.array([10.+0j, -2.+2j, -2.+0j, -2.-2j])
print('np.fft.ifft(a) =')
print(np.fft.ifft(a))


# %% Task 2 - 76. Sorting: table default and column sort
a = np.array([[3, 1, 2], [0, 5, 4], [6, -1, 8]])
print('np.sort(a) =')
print(np.sort(a))
a.sort(axis=0)
print("a.sort(axis=0), modified a =")
print(a)


# %% Task 2 - 77. Sort each row
a = np.array([[3, 1, 2], [0, 5, 4], [6, -1, 8]])
print('np.sort(a, axis=1) =')
print(np.sort(a, axis=1))
a.sort(axis=1)
print("In-place row sort:")
print(a)


# %% Task 2 - 78. Sort rows by the first column
a = np.array([[3, 30], [1, 10], [2, 20]])
I = np.argsort(a[:, 0])
b = a[I, :]
print("I =", I)
print("b ="); print(b)


# %% Task 2 - 79. Least-squares linear regression
Z = np.column_stack((np.ones(4), np.arange(4.)))
y = np.array([1., 2., 2., 4.])
x = linalg.lstsq(Z, y)
print("Full return tuple:"); print(x)
coef, residuals, rank, singular_values = x
print("Coefficients:", coef)
print("Predictions:", Z @ coef)


# %% Task 2 - 80. Fourier resampling
x = np.sin(2 * np.pi * np.arange(12) / 12)
q = 3
print('signal.resample(x, int(np.ceil(len(x) / q))) =')
print(signal.resample(x, int(np.ceil(len(x) / q))))


# %% Task 2 - 81. Unique values
a = np.array([3, 1, 3, 2, 1])
print('np.unique(a) =')
print(np.unique(a))


# %% Task 2 - 82. Remove singleton axes
a = np.arange(6).reshape(1, 2, 3, 1)
print('a.shape =')
print(a.shape)
print('a.squeeze() =')
print(a.squeeze())
print('a.squeeze().shape =')
print(a.squeeze().shape)


# %% Task 3: Required plot
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,2,7,14])
plt.axis([0, 6, 0, 20])
plt.show()


# %% Task 4: Matplotlib figure
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 5, 1001)
envelope = np.exp(-0.6 * t)
y = envelope * np.cos(2 * np.pi * 1.5 * t)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t, y, color="navy", linewidth=1.6, label="Damped oscillation")
ax.plot(t, envelope, "--", color="darkorange", label="Positive envelope")
ax.plot(t, -envelope, "--", color="darkorange", label="Negative envelope")
ax.set(title="Exponentially damped oscillation",
       xlabel="Time (s)", ylabel="Amplitude (normalized)",
       xlim=(0, 5), ylim=(-1.1, 1.1))
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
plt.show()


# %% Task 5: Version control system (GitHub)



# %% Task 6: Integrated development environment (Spyder)


