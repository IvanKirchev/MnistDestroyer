import numpy as np

def categorical_cross_entropy_cost(y_hat, y, params = None, lambd = None):
    '''
    Cost function: Categorical Cross Entropy Loss for c classes over m examples

    Params:
        y_hat: shape (c, m)
        y:     shape (c, m)
        lambd: regularization parameter (optional)
    '''

    # reg_term = 0
    m = y_hat.shape[1]
    # for key,val in params.items():
    #     if 'W' in key:
    #         reg_term += np.sum(np.square(val))

    losses = np.sum(np.multiply(y, np.log(y_hat + 1e-10)), axis = 0, keepdims = True) 
    cost = -( 1 / m) * np.sum(losses)  # + ((lambd / (m * 2)) * reg_term) L2 reg

    return cost