from math import log, sqrt

def f(x, W):
    w1, w2, w3 = W
    x1, x2, x3 = x
    return -w1*log(x1)-w2*log(x2)-w3*log(x3)+1/((1-x1-x2-x3)**2)

def derivatives(x, W):
    w1, w2, w3 = W
    x1, x2, x3 = x

    # compute each gradient of objective function
    dx1 = -w1/x1 + 2/(1-x1-x2-x3)**(3)
    dx2 = -w2/x2 + 2/(1-x1-x2-x3)**(3)
    dx3 = -w3/x3 + 2/(1-x1-x2-x3)**(3)

    return dx1, dx2, dx3


def line_search_method(x0, W, a0, c, tau):
    # W : parameters of optimization problem
    # a0 : initial state of step size in backtracking line search
    # c : controll parameter in backtracking line search
    # tau ; stepsize decay parameter in backtracking line search

    k = 0 #starting index
    x1, x2, x3 = x =x0 #initial guess
    print("initial value at x0 : {}".format(f(x0, W)))
    print("initial point is at {}".format(x0))

    while True:
        # compute gradient at x[k]
        x1, x2, x3 = x
        dx1, dx2, dx3 = derivatives([x1,x2,x3], W)
        
        #Descent direction is normalized to unit vector for Backtracking line search.
        #"It is assumed that p is a unit vector in a direction in which some local decrease is possible" 
        #ref: https://en.wikipedia.org/wiki/Backtracking_line_search        
        norm = sqrt(dx1**2+dx2**2+dx3**2)
        px1, px2, px3 = -dx1/norm , -dx2/norm, -dx3/norm

        # run Backtracking line search.
        a = a0
        i = 0
        while f([x1+a*px1,x2+a*px2, x3+a*px3], W) > f([x1,x2,x3], W) + a*c*(px1*dx1+px2*dx2+px3*dx3): 
            #until Armijo-Goldstein condition fulfilled: f(x+ap) <= f(x)+acm
            a = a*tau
            i+=1
            if i > 20:
                break
        x = x1+a*px1, x2+a*px2, x3+a*px3
        k = k+1
        print("{}-step's value at : {}".format(k,f(x, W)))
        print("point is at {}".format(x))
        if norm < 1E-4:
            break
        if x[0]+x[1]+x[2] > 1:
            break


if __name__ == "__main__":
    k = 0 # starting index

    x0 = [0.01, 0.01, 0.01] # initial guess
    W = [1., 1., 1.]
    a0 = 0.1
    c = 0.005
    tau = 0.5
    line_search_method(x0, W, a0, c, tau)

    # x0 = [0.08, 0.08, 0.08] # initial guess
    # print(f(x0, W))

    # x0 = [0.1, 0.1, 0.1] # initial guess
    # print(f(x0, W))

    # print(derivatives(x0, W))

    # x0 = [0.11, 0.11, 0.11] # initial guess
    # print(f(x0, W))

    # print(derivatives(x0, W))


    # x0 = [0.12, 0.12, 0.12] # initial guess
    # print(f(x0, W))

    # print(derivatives(x0, W))

    # x0 = [0.13, 0.13, 0.13] # initial guess
    # print(f(x0, W))

    # print(derivatives(x0, W))





    # x0 = [0.15, 0.15, 0.15] # initial guess
    # print(f(x0, W))

    # x0 = [0.2, 0.2, 0.2] # initial guess
    # print(f(x0, W))


    # x0 = [0.3, 0.3, 0.3] # initial guess
    # print(f(x0, W))

    # x0 = [2.4, 2.4, 2.4] # initial guess
    # print(f(x0, W))