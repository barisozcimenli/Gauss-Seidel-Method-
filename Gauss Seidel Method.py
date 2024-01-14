import numpy as np

def gauss_seidel_recursive(A, b, x, iteration=0, tolerance=1e-4, max_iterations=50):
    if iteration >= max_iterations:
        print("Maximum iteration limit reached.")
        return x

    x_old = x.copy()

    for i in range(3):
        sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x_old[i + 1:])
        x[i] = round((b[i] - sigma) / A[i, i], 4)

    print(f"Iteration {iteration + 1}:")
    print(f"x1 = {x[0]:.4f}  x2 = {x[1]:.4f}  x3 = {x[2]:.4f}")

    if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
        print(f"Converged in {iteration + 1} iterations.")
        return x

    return gauss_seidel_recursive(A, b, x, iteration + 1, tolerance, max_iterations)

def question_2_recursive():
    A = np.array([[-5, -1, 2], [4, 12, -6], [1, 0.5, 3.5]])
    x = np.array([0, 0.0006, 0])
    b = ([1, 4, 16])

    final = gauss_seidel_recursive(A, b, x)
    print("Final solution:", f"x1 = {final[0]:.4f}  x2 = {final[1]:.4f}  x3 = {final[2]:.4f}")

question_2_recursive()