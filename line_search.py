from math import log, sqrt

# def f(x, W):
#     w1, w2, w3 = W
#     x1, x2, x3 = x
#     return w1*log(x1)+w2*log(x2)+w3*log(x3)+1/(1-x1-x2-x3)**2

# def derivatives(x, W):
#     w1, w2, w3 = W
#     x1, x2, x3 = x

#     # compute each gradient of objective function
#     dx1 = -w1/x1 + 2/(1-x1-x2-x3)**(3)
#     dx2 = -w2/x2 + 2/(1-x1-x2-x3)**(3)
#     dx3 = -w3/x3 + 2/(1-x1-x2-x3)**(3)

#     return dx1, dx2, dx3

def f(x, W):
    w1, w2, w3 = W
    x1, x2, x3 = x
    return w1*x1**2+w2*x2**2+w3*x3**2

def derivatives(x, W):
    w1, w2, w3 = W
    x1, x2, x3 = x

    # compute each gradient of objective function
    dx1 = 2*w1*x1
    dx2 = 2*w2*x2
    dx3 = 2*w3*x3

    return dx1, dx2, dx3


def line_search_method(x0, W, a0, c, tau):
    # W : parameters of optimization problem
    # a0 : initial state of step size in backtracking line search
    # c : controll parameter in backtracking line search
    # tau ; stepsize decay parameter in backtracking line search

    k = 0 #starting index
    x1, x2, x3 = x0 #initial guess
    print("initial value at x0 : {}".format(f(x0, W)))

    while True:
        # compute gradient at x[k]
        dx1, dx2, dx3 = derivatives([x1,x2,x3], W)
        
        #Descent direction is normalized to unit vector for Backtracking line search.
        #"It is assumed that p is a unit vector in a direction in which some local decrease is possible" 
        #ref: https://en.wikipedia.org/wiki/Backtracking_line_search        
        norm = sqrt(dx1**2+dx2**2+dx3**2)
        px1, px2, px3 = -dx1/norm , -dx2/norm, -dx3/norm

        # run Backtracking line search.
        a = a0
        while f([x1+a*px1,x2+a*px2, x3+a*px3], W) > f([x1,x2,x3], W) + a*c*(px1*dx1+px2*dx2+px3*dx3): 
            #until Armijo-Goldstein condition fulfilled: f(x+ap) <= f(x)+acm
            a = a*tau
        print("a : ", a)
        x = x1+a*px1, x2+a*px2, x3+a*px3
        k = k+1
        print("{}-step's value at : {}".format(k,f(x, W)))

        if k > 100:
            print(x)
            break


if __name__ == "__main__":
    k = 0 # starting index
    x0 = [0.1, 0.1, 0.1] # initial guess
    W = [1., 1., 1.]
    a0 = 4.0
    c = 0.05
    tau = 0.5
    line_search_method(x0, W, a0, c, tau)