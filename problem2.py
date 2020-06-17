from math import log, sqrt

def f(x, W):
    w1, w2, w3 = W
    x1, x2, x3 = x
    part1 = -w1*log(x1+1E-5)-w2*log(x2+1E-5)-w3*log(x3+1E-5)
    part2 = +1/((1-x1-x2+1E-5)**2)+1/((1-x1-x3+1E-5)**2)

    return part1+part2

def derivatives(x, W):
    w1, w2, w3 = W
    x1, x2, x3 = x

    # compute each gradient of objective function
    dx1 = -w1/x1 + 2/(1-x1-x2)**(3) + 2/(1-x1-x3)**(3)
    dx2 = -w2/x2 + 2/(1-x1-x2)**(3)
    dx3 = -w3/x3 + 2/(1-x1-x3)**(3)

    return dx1, dx2, dx3


def line_search_method(x0, W, a0, c, tau):
    # W : parameters of optimization problem
    # a0 : initial state of step size in backtracking line search
    # c : controll parameter in backtracking line search
    # tau ; stepsize decay parameter in backtracking line search

    k = 0 #starting index
    x1, x2, x3 = x = x0 #initial guess
    print("initial value at x0 : {}".format(f(x0, W)))

    while True:
        # compute gradient at x[k]
        x1, x2, x3 = x
        dx1, dx2, dx3 = derivatives([x1,x2,x3], W)
        
        #Descent direction is normalized to unit vector for Backtracking line search.
        #"It is assumed that p is a unit vector in a direction in which some local decrease is possible" 
        #ref: https://en.wikipedia.org/wiki/Backtracking_line_search        
        norm = sqrt(dx1**2+dx2**2+dx3**2)
        px1, px2, px3 = -dx1/norm , -dx2/norm, -dx3/norm
        if norm < 1E-3:
            break        
        # run Backtracking line search.
        a = a0
        i = 0  
        while f([x1+a*px1,x2+a*px2, x3+a*px3], W) > f([x1,x2,x3], W) + a*c*(px1*dx1+px2*dx2+px3*dx3): 
            #until Armijo-Goldstein condition fulfilled: f(x+ap) <= f(x)+acm
            a = a*tau
            i+=1
            if i > 30:
                break
        x = x1+a*px1, x2+a*px2, x3+a*px3
        k = k+1
        print("{}-step's value is : {}".format(k,f(x, W)))
        print("{}-step's point is at {}".format(k, x))


if __name__ == "__main__":
    k = 0 # starting index

    x0 = [0.01, 0.01, 0.01] # initial guess
    W = [1., 1., 1.]
    a0 = 0.01
    c = 0.005
    tau = 0.5
    line_search_method(x0, W, a0, c, tau)

    # x0 = [0.08, 0.08, 0.08] # initial guess
    # print(x0)
    # print(f(x0, W))
    # print(derivatives(x0, W))

    # x0 = [0.1, 0.1, 0.1] # initial guess
    # print(x0)
    # print(f(x0, W))
    # print(derivatives(x0, W))

    # x0 = [0.12, 0.12, 0.12] # initial guess
    # print(x0)
    # print(f(x0, W))
    # print(derivatives(x0, W))

    # x0 = [0.10, 0.15, 0.15] # initial guess
    # print(x0)
    # print(f(x0, W))
    # print(derivatives(x0, W))


    # x0 = [0.11, 0.17, 0.17] # initial guess
    # print(x0)
    # print(f(x0, W))
    # print(derivatives(x0, W))


    # x0 = [0.105, 0.18, 0.18] # initial guess
    # print(x0)
    # print(f(x0, W))
    # print(derivatives(x0, W))


    # x0 = [0.10, 0.15, 0.15] # initial guess
    # print(x0)
    # print(f(x0, W))
    # print(derivatives(x0, W))

    # x0 = [0.09, 0.18, 0.18] # initial guess
    # print(x0)
    # print(f(x0, W))
    # print(derivatives(x0, W))


    # x0 = [0.01, 0.185, 0.185] # initial guess
    # print(x0)
    # print(f(x0, W))
    # print(derivatives(x0, W))

