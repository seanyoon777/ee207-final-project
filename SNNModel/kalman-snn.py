import numpy as np
import nengo
import nengo_spinnaker
import matplotlib.pyplot as plt
from nengo.dists import Uniform
from dataloader import Dataloader
from kalman import Kalman
from nengo.processes import Piecewise

dt = 0.004  # simulation time step
t_rc = 0.002  # membrane RC time constant
t_ref = 0.0003  # refractory period
tau = 0.0016  # synapse time constant for standard first-order lowpass filter synapse
N_A = 4000  # number of neurons in first population
rate_A = 200, 400  # range of maximum firing rates for population A
sampling_rate = 250  # Hz: sampling rate of the data
pool = 0
N = 10 # state update interval

mses = np.zeros((10, 7))

dataloader = Dataloader()
kalman = Kalman()

trainX, testX, trainY, testY, trainY_cursor, testY_cursor = dataloader()
trainX, testX, trainY, testY, trainY_cursor, testY_cursor = trainX.T, testX.T, trainY.T, testY.T, trainY_cursor.T, testY_cursor.T
ChN = np.where(np.sum(trainX, axis=1) != 0) #filter out empty channels
trainX = np.squeeze(trainX[ChN, :])
testX = np.squeeze(testX[ChN, :])

num_channels = trainX.shape[0]
num_states = trainY.shape[0]

A_0, B_0 = kalman.calculate(trainX, trainY, pool=pool, dt=dt, tau=tau)
# np.savez_compressed('processed_data.npz', testX=testX, testY=testY, testY_cursor=testY_cursor, A_0=A_0, B_0=B_0)
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
    Provide state updates from testY at intervals of N * dt.
    """
    if t == 0.0:
        return np.zeros(num_states).tolist()

    index = int(sampling_rate * (t - dt))
    if index >= testY.shape[1]:
        return np.zeros(num_states).tolist()  # Return zeros if t exceeds testY length

    # Only update at intervals of N milliseconds
    if ((t - dt) * 1000) % N == 0:
        return np.squeeze(np.asarray(testY[:, index])).tolist()
    else:
        return np.zeros(num_states).tolist()


def run_kalman_decoder():
    with (model):
        # Direct neurons do not model spiking dynamics
        Dir_Neurons = nengo.Ensemble(
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

        state = nengo.Node(output=state_func)
        state_probe = nengo.Probe(state)

        external_input = nengo.Node(output=lambda t: data(t))
        external_input_probe = nengo.Probe(external_input)

        # Define feedback loop and run sim
        conn0 = nengo.Connection(state, Dir_Neurons[:num_states])
        conn1 = nengo.Connection(external_input, Dir_Neurons[num_states:])
        conn2 = nengo.Connection(Dir_Neurons, LIF_Neurons, function=update, synapse=tau)
        conn3 = nengo.Connection(LIF_Neurons, Dir_Neurons[:num_states])

        neurons_out = nengo.Probe(LIF_Neurons)

        print("starting sim for N = ", N)
        with nengo.Simulator(model, dt=dt) as sim:
            sim.run(3)

        # Calculate correlation and RMSE
        time_range = sim.trange()
        for i in range(num_states):
            rmse_i = np.sqrt(np.mean(np.square(sim.data[neurons_out][:, i] - sim.data[origin_probe][:, i])))
            mses[int((N-1)/4), i] = rmse_i
            print(f"RMSE (Dimension {i+1}): {rmse_i}")
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
        plt.plot(sim.data[neurons_out][:, 0], sim.data[neurons_out][:, 1], label="Predicted trajectory", linewidth=0.5)
        plt.plot(sim.data[origin_probe][:, 0], sim.data[origin_probe][:, 1], label="True trajectory", linewidth=0.5)
        plt.legend()
        plt.show()

        plt.figure()
        fig, axes = plt.subplots(num_states, 1, figsize=(10, 2 * num_states), sharex=True)
        for i in range(num_states):
            axes[i].plot(time_range, sim.data[neurons_out][:, i], label="Decoded estimate", linewidth=0.5)
            axes[i].plot(time_range, sim.data[origin_probe][:, i], label="Origin", linewidth=0.5)
            axes[i].set_ylabel(f"Dimension {i+1}")
            axes[i].legend()
        axes[-1].set_xlabel("Time (s)")
        plt.title(f"N={N}")
        plt.tight_layout()
        plt.show()


N_list = 4 * np.arange(1, 11)
for N in N_list:
    run_kalman_decoder()

# run_kalman_decoder()

# plt.figure()
# fig, axes = plt.subplots(num_states, 10, figsize=(10, 20), sharex=True)
# for i in range(num_states):
#     axes[i].plot(N_list, mses[:, i], label=f"Dimension {i+1}")
#     axes[i].set_ylabel("RMSE")
#     axes[i].set_xlabel("N")
#     axes[i].legend()
# plt.tight_layout()
# plt.show()
