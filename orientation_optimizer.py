import numpy as np
import argparse

all_orientation_signals = np.load(
    "all_orientation_signals.npy"
)

# Normalize each precomputed profile so that its 24 bins sum to 1.
# This removes the old TOTAL_SIGNAL = 240 scaling.
all_orientation_signals = (
    all_orientation_signals
    / np.sum(all_orientation_signals, axis=1, keepdims=True)
)

# Sanity check:
print(np.sum(all_orientation_signals, axis=1))

"""
load 24 precomputed profiles
→ know how many orientations exist
→ generate every possible N-detector orientation combination
→ return those combinations"""
def compute_sigma_array(all_orientation_signals, n_detectors, total_signal, background_per_bin):

    orientation_options = np.arange( all_orientation_signals.shape[0])

    orientation_grid = np.array(
        np.meshgrid(
            *([orientation_options] * n_detectors),
            indexing="ij"
        )
    )

    orientation_combinations = (
        orientation_grid
        .reshape(n_detectors, -1)
        .T
    )

    # Use the orientation indices to retrieve the corresponding precomputed 24-bin signal profiles.
    # Retrieve the normalized profiles and scale each detector to the requested total signal count.
    orientation_signal_array = (total_signal * all_orientation_signals[orientation_combinations])

     # Read the dimensions from the signal array.
    n_configs = orientation_signal_array.shape[0]
    n_bins = orientation_signal_array.shape[2]

    n_parameters = n_detectors + 2
    common_background_index = n_parameters - 1

    # Shape:
    # configurations x detectors x parameters x time bins
    derivatives_all = np.zeros((n_configs, n_detectors, n_parameters, n_bins))

    derivatives_all[:, :, 0, :] = orientation_signal_array

    
    flat_background_derivative = np.full(
        n_bins,
        1 / n_bins
    )

    for d in range(n_detectors):
        derivatives_all[:, d, d + 1, :] = (
            flat_background_derivative
        )

    # Build the shared sinusoidal-background derivative.
    time_bin_centers = np.arange(n_bins) + 0.5
    omega = 2 * np.pi / 24.0

    common_background_shape = np.sin(
        omega * time_bin_centers
    )

    # Every detector responds to the same shared sinusoidal background.
    derivatives_all[
        :, :, common_background_index, :
    ] = common_background_shape

    # Create one flat-background array for every
    # configuration, detector, and time bin.
    background_all = np.full(
        (n_configs, n_detectors, n_bins),
        background_per_bin
    )

    # Build every Fisher matrix simultaneously.
    fisher_all = np.einsum(
        "cdik,cdjk,cdk->cij",
        derivatives_all,
        derivatives_all,
        1 / background_all
    )

    # Invert all Fisher matrices at once.
    covariance_all = np.linalg.inv(fisher_all)

    # The [0,0] entry of each covariance matrix
    # is the marginalized variance of A.
    sigma_A_all = np.sqrt(
        covariance_all[:, 0, 0]
    )

    ranking = np.argsort(sigma_A_all)

    return sigma_A_all, orientation_combinations, ranking

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-detectors",
        type=int,
        required=True,
        help="Number of detectors to optimize"
    )

    parser.add_argument(
    "--total-signal",
    type=float,
    default=240.0,
    help="Total signal counts per detector"
    )

    parser.add_argument(
    "--background-per-bin",
    type=float,
    default=100.0,
    help="Flat background counts per hourly bin"
    )
    
    args = parser.parse_args()

    sigma_A_all, orientation_combinations, ranking = compute_sigma_array(
    all_orientation_signals,
    args.n_detectors,
    args.total_signal,
    args.background_per_bin
    )

    best_index = np.argmin(sigma_A_all)

    print("Number of configurations:", len(sigma_A_all))
    print("Best sigma_A:", sigma_A_all[best_index])
    print(
        "Best orientations:",
        orientation_combinations[best_index]
    )

    background_total = (
    args.background_per_bin
    * all_orientation_signals.shape[1]
    )

    signal_to_background = (
        args.total_signal / background_total
    )

    print("S_tot / B_tot:", signal_to_background)
