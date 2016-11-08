import code.eft_internal as ekf_internal


def f_radar(x, dt):
    '''
    State transition function for a constant velocity
    aircraft with state vector [x, velocity, altitude]
    '''
    F = np.array([1, dt, 0],
		 [0, 1, 0],
		 [0, 0, 1]], dtype=float)
    return np.dot(F, x)


