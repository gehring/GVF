import numpy as np

class linearValueFn(object):
    def __init__(self, weights, projector):
        self.theta= weights
        self.proj = projector
    def __call__(self, state):
        return self.projector(state).dot(self.theta)

def getGVF(policy, 
           environment,
           gamma,
           feature, 
           method='LSTDlambda', 
           projector,
           **args):
    if method == 'linearTDlambda':
        return getGVFlinearTDlambda(policy, 
                                    environment,
                                    gamma, 
                                    feature, 
                                    projector,
                                    **args)
    if method == 'LSTD':
        return getGVFLSTDlambda(policy, 
                                    environment,
                                    gamma, 
                                    feature, 
                                    projector,
                                    **args)
    
def getGVFlinearTDlambda(policy, 
           environment,
           gamma,
           feature, 
           projector,
           alpha = 0.01,
           lamb = 0.0,
           number_episodes,
           max_episode_length,
           **args):
    phi = projector
    theta = np.zeros(projector.size)
    e = np.zeros(projector.size)
    
    for i in xrange(number_episodes):
        x_t = environment.reset()[1]
        p_t = phi(x_t)
        z = p_t
        t=0
        while not environment.isterminal() and t<max_episode_length:
            x_tp1 = environment.step(policy(x_t))(1)
            p_tp1 = phi(x_tp1)
            delta = feature(x_t) + gamma*p_tp1.dot(theta) - p_t.dot(theta)
            theta += alpha*delta*e
            e = gamma*lamb * e + p_tp1
            x_t = x_tp1
            p_t = p_tp1
            t += 1
    return linearValueFn(theta, phi)

def getGVFLSTDlambda(policy, 
           environment,
           gamma,
           feature, 
           projector,
           lamb = 0.0,
           number_episodes,
           max_episode_length,
           **args):
    phi = projector
    A = np.zeros((projector.size, projector.size))
    b = np.zeros(projector.size)
    
    for i in xrange(number_episodes):
        x_t = environment.reset()[1]
        p_t = phi(x_t)
        z = p_t
        t=0
        while not environment.isterminal() and t<max_episode_length:
            x_tp1 = environment.step(policy(x_t))(1)
            p_tp1 = phi(x_tp1)
            A += np.outer(z, p_t - gamma*p_tp1)
            b += z * feature(x_t)
            z = gamma*lamb * z + p_tp1
            x_t = x_tp1
            p_t = p_tp1
            t += 1
            
    theta = np.linalg.solve(A, b)
    return linearValueFn(theta, phi)
        
    
    
    