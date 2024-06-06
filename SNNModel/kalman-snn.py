import numpy as np
import nengo
import matplotlib.pyplot as plt
from nengo.dists import Uniform
from dataloader import Dataloader
from kalman import Kalman
from nengo.processes import Piecewise

dt = 0.008  # simulation time step
t_rc = 0.001  # membrane RC time constant
t_ref = 0.0001  # refractory period
tau = 0.001  # synapse time constant for standard first-order lowpass filter synapse
N_A = 10000  # number of neurons in first population
rate_A = 200, 400  # range of maximum firing rates for population A
sampling_rate = 250  # Hz: sampling rate of the data
pool = 0

dataloader = Dataloader()
kalman = Kalman()

trainX, trainY, testX, testY = dataloader.getData()
ChN = np.where(np.sum(trainX, axis=1) != 0) #filter out empty channels
trainX = np.squeeze(trainX[ChN, :])
testX = np.squeeze(testX[ChN, :])

num_channels = trainX.shape[0]
num_states = trainY.shape[0]

A_0, B_0 = kalman.calculate(trainX, trainY, pool=pool, dt=dt, tau=tau)
print("calculated filter matrices")

# A_1, B_1 = kalman.calculate(trainX, trainY, pool=1, dt=dt, tau=tau)

# kalman.Kalman_Filter(testX, testY)

# kalman.standard_Kalman_Filter(testX, testY)


#

def data(t):
    """
    Neuron records, Y_k, and calculate B * Y_k
    """
    if t == 0.0:
        return np.zeros(num_states).tolist()  # Initial condition
    yt = np.mat(testX[:, int(sampling_rate * t)])
    out = B_0 @ yt.T
    return np.squeeze(np.asarray(out)).tolist()


lifRate_neuron = nengo.LIFRate(tau_rc=t_rc, tau_ref=t_ref)
model = nengo.Network(label="NEF")


def update(x):
    """
    Kalman Filter: X_k = A * X_k_1 + B * Y_k
    """
    Externalmat = np.mat(x[num_states:]).T  # External inputs
    Inputmat = np.mat(x[0:num_states]).T  # state matrix

    # Perform the Kalman Filter state update
    next_state = np.squeeze(np.asarray(A_0 @ Inputmat + Externalmat))
    return next_state.tolist()

def state_func(t):
    """
    Provide continuous state updates from testY.
    """
    if t == 0.0:
        return np.zeros(num_states).tolist()  # Initial condition
    index = int(sampling_rate * t)
    if index >= testY.shape[1]:
        return np.zeros(num_states).tolist()  # Return zeros if t exceeds testY length
    return np.squeeze(np.asarray(testY[:, index])).tolist()

with model:
    # Direct neurons do not model spiking dynamics
    Dir_Nurons = nengo.Ensemble(
        1,
        dimensions=num_states * 2,  # 6 for state + 6 for external input
        neuron_type=nengo.Direct()
    )

    # Biological neurons
    LIF_Neurons = nengo.Ensemble(
        N_A,
        dimensions=num_states,
        intercepts=Uniform(-1, 1),
        max_rates=Uniform(rate_A[0], rate_A[1]),
        neuron_type=lifRate_neuron
    )

    # The origin outputs the kinematic data at the corresponding time
    origin = nengo.Node(lambda t: testY[:, int(sampling_rate * t)])  # Ensure indexing is within bounds
    origin_probe = nengo.Probe(origin)  # Used to collect data from the origin node

    # state_func = Piecewise({
    #     0.0: [0.0] * num_states,  # Initial condition: zero for all 6 dimensions
    #     dt: np.squeeze(np.asarray(testY[: , 0])),  # First state update from testY
    #     2 * dt: [0.0] * num_states  # Another state update (can be modified as needed)
    # })

    state = nengo.Node(output=state_func)
    state_probe = nengo.Probe(state)

    external_input = nengo.Node(output=lambda t: data(t))
    external_input_probe = nengo.Probe(external_input)

    # Define feedback loop and run sim
    conn0 = nengo.Connection(state, Dir_Nurons[:num_states])
    conn1 = nengo.Connection(external_input, Dir_Nurons[num_states:])
    conn2 = nengo.Connection(Dir_Nurons, LIF_Neurons, function=update, synapse=tau)
    conn3 = nengo.Connection(LIF_Neurons, Dir_Nurons[:num_states])

    neurons_out = nengo.Probe(LIF_Neurons)

    print("starting sim")
    with nengo.Simulator(model, dt=dt) as sim:
        sim.run(300)

    # Calculate correlation and RMSE
    # corrcoef_x = np.corrcoef(sim.data[neurons_out][:, 0], testY[0, 1:18001])
    # rmse_x = np.sqrt(np.mean(np.square(sim.data[neurons_out][:, 0] - testY[0, 1:18001])))
    #
    # corrcoef_y = np.corrcoef(sim.data[neurons_out][:, 1], testY[1, 1:18001])
    # rmse_y = np.sqrt(np.mean(np.square(sim.data[neurons_out][:, 1] - testY[1, 1:18001])))
    #
    # print(f"Correlation coefficient (X): {corrcoef_x[0, 1]}")
    # print(f"RMSE (X): {rmse_x}")
    # print(f"Correlation coefficient (Y): {corrcoef_y[0, 1]}")
    # print(f"RMSE (Y): {rmse_y}")

    plt.figure()
    fig, axes = plt.subplots(num_states, 1, figsize=(10, 2 * num_states), sharex=True)
    time_range = sim.trange()
    for i in range(num_states):
        axes[i].plot(time_range, sim.data[neurons_out][:, i], label="Decoded estimate", linewidth=0.5)
        axes[i].plot(time_range, sim.data[origin_probe][:, i], label="Origin", linewidth=0.5)
        axes[i].set_ylabel(f"Dimension {i+1}")
        axes[i].legend()
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()