# Import packages.
import cvxpy as cp
import numpy as np

def main():
    x = cp.Variable(3)
    P = np.array([[19.,12.,-2.],
                  [12.,17., 6.],
                  [-2., 6.,12.]])
    q = np.array([-22.0,
                  -14.5,
                  13.0])
    r = 1
    G = np.array([[1, 0, 0], # x_1 <= 1
                  [0, 1, 0], # x_2 <= 1
                  [0, 0, 1], # x_3 <= 1
                  [-1, 0 ,0], # -x_1 <= 1 ; -1 <= x_1
                  [0, -1, 0], # -x_2 <= 1 ; -1 <= x_2
                  [0, 0, -1]]) # -x_3 <= 1 ; -1 <= x_3
    h = np.array([1, 1, 1, 1, 1, 1])
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x +r),
                     [G @ x <= h])
    # Print result.
    prob.solve()
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)

    '''
    np.random.seed(1)
    P = np.array([[19.,12.,-2.],
                  [12.,17., 6.],
                  [-2., 6.,12.]])
    r = 1
    q = np.array([-22.0,
                  -14.5,
                  13.0])
    G = np.array([[1, 0, 0], # x_1 <= 1
                  [0, 1, 0], # x_2 <= 1
                  [0, 0, 1], # x_3 <= 1
                  [-1, 0 ,0], # -x_1 <= 1 ; -1 <= x_1
                  [0, -1, 0], # -x_2 <= 1 ; -1 <= x_2
                  [0, 0, -1]]) # -x_3 <= 1 ; -1 <= x_3
    #h = G @ np.random.randn(n)
    h = np.array([1, 1, 1, 1, 1, 1])

    x = cp.Variable(3)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x +r),
                     [G @ x <= h])
    prob.solve()
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)
    '''

if __name__ == "__main__":
    main()