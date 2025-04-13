import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os

# Helper function for plotting ellipses
def _plot_ellipse(position, covariance, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Plots an ellipse representing the covariance matrix.
    """
    pearson = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    scale_x = np.sqrt(covariance[0, 0]) * n_std
    scale_y = np.sqrt(covariance[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(position[0], position[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Main EM Algorithm Function
def run_em_algorithm(data, n_components=3, max_iter=100, tolerance=1e-4, output_dir='results/em/', seed=42):
    """
    Runs the Expectation-Maximization algorithm for a Gaussian Mixture Model.

    Args:
        data (np.ndarray): The input data (n_samples, n_features).
        n_components (int): The number of Gaussian components (K).
        max_iter (int): Maximum number of EM iterations.
        tolerance (float): Convergence threshold for log-likelihood change.
        output_dir (str): Directory to save plots.
        seed (int): Random seed for initialization reproducibility.

    Returns:
        tuple: (means, covariances, weights, responsibilities, log_likelihood_history)
               Returns None if input data is invalid.
    """
    if data is None or data.shape[0] == 0 or data.shape[1] == 0:
        print("Error: Input data is empty or invalid.")
        return None

    n_samples, n_features = data.shape
    print(f"\n--- Running EM Algorithm ---")
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Components: {n_components}, Max Iter: {max_iter}, Tolerance: {tolerance}")

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots will be saved to: {output_dir}")
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        return None # Cannot proceed without output directory

    # Task 1: Plot and save the scatter plot of the data
    print("Saving raw data scatter plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6)
    plt.title('Scatter Plot of the Synthetic Data')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    try:
        plt.savefig(os.path.join(output_dir, 'em_raw_data.png'))
    except Exception as e:
        print(f"Error saving raw data plot: {e}")
    plt.close()

    # EM Algorithm Implementation

    # 1. Initialization
    np.random.seed(seed)
    initial_means = data[np.random.choice(n_samples, n_components, replace=False), :]
    initial_covariances = [np.eye(n_features) * np.var(data, axis=0).mean() for _ in range(n_components)]
    initial_weights = np.ones(n_components) / n_components

    means = initial_means.copy()
    covariances = [cov.copy() for cov in initial_covariances]
    weights = initial_weights.copy()

    log_likelihood_history = []
    prev_log_likelihood = -np.inf
    responsibilities = np.zeros((n_samples, n_components))

    # EM Iterations
    print("Starting EM iterations...")
    iteration = 0
    for iteration in range(max_iter):
        current_responsibilities = np.zeros((n_samples, n_components))
        for k in range(n_components):
            try:
                pdf = multivariate_normal.pdf(data, mean=means[k], cov=covariances[k], allow_singular=True)
                current_responsibilities[:, k] = weights[k] * pdf
            except np.linalg.LinAlgError:
                print(f"Warning: Singular covariance matrix encountered for component {k} in E-step iter {iteration+1}. Adding small identity.")
                covariances[k] += 1e-6 * np.eye(n_features)
                try:
                   pdf = multivariate_normal.pdf(data, mean=means[k], cov=covariances[k], allow_singular=True)
                   current_responsibilities[:, k] = weights[k] * pdf
                except Exception as e_inner:
                   print(f"Error recalculating PDF after stabilization for component {k}: {e_inner}")
                   current_responsibilities[:, k] = 1e-9
            except Exception as e:
                print(f"Error in E-step for component {k}, iter {iteration+1}: {e}")
                current_responsibilities[:, k] = 1e-9

        sum_responsibilities = np.sum(current_responsibilities, axis=1)[:, np.newaxis]
        responsibilities = current_responsibilities / (sum_responsibilities + 1e-9)

        # Update parameters 
        nk = np.sum(responsibilities, axis=0)

        for k in range(n_components):
            if nk[k] < 1e-6:
                print(f"Warning: Component {k} collapsed (nk={nk[k]:.2e}) at iter {iteration+1}. Skipping update.")
                continue

            means[k] = np.sum(responsibilities[:, k][:, np.newaxis] * data, axis=0) / nk[k]
            diff = data - means[k]
            cov_k = np.dot((responsibilities[:, k][:, np.newaxis] * diff).T, diff) / nk[k]
            covariances[k] = cov_k + 1e-6 * np.eye(n_features) # regularization
            weights[k] = nk[k] / n_samples

        # Calculate Log-Likelihood
        log_likelihood_term = np.zeros((n_samples, n_components))
        for k in range(n_components):
           try:
               log_likelihood_term[:, k] = weights[k] * multivariate_normal.pdf(data, mean=means[k], cov=covariances[k], allow_singular=True)
           except Exception:
               # If PDF fails even with regularization, assign a very small probability
               log_likelihood_term[:, k] = 1e-9

        current_log_likelihood = np.sum(np.log(np.sum(log_likelihood_term, axis=1) + 1e-9))
        log_likelihood_history.append(current_log_likelihood)

        # Check for Convergence
        likelihood_change = current_log_likelihood - prev_log_likelihood
        if iteration > 0 and abs(likelihood_change) < tolerance:
            print(f"Convergence reached after {iteration + 1} iterations.")
            break
        prev_log_likelihood = current_log_likelihood

        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}/{max_iter}, Log-Likelihood: {current_log_likelihood:.4f}")

    if iteration == max_iter - 1:
        print(f"EM algorithm reached maximum iterations ({max_iter}) without converging within tolerance {tolerance}.")
    else:
        pass


    # Plot cluster assignments and report parameters
    print("\nSaving clustering results plot...")
    cluster_assignments = np.argmax(responsibilities, axis=1)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, n_components))

    for k in range(n_components):
        cluster_data = data[cluster_assignments == k]
        if cluster_data.shape[0] > 0:
             plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=10, alpha=0.6, color=colors[k], label=f'Cluster {k+1}')
        else:
             print(f"Note: Cluster {k+1} is empty.")


    plt.scatter(means[:, 0], means[:, 1], marker='X', s=150, c='red', edgecolors='black', label='Estimated Means')
    plt.title('EM Clustering Results (Colored by Assignment)')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot the estimated Gaussian distributions on the same plot
    print("Adding estimated Gaussians to clustering plot...")
    ax = plt.gca() # Get the current axes
    plot_handles = {}
    for k in range(n_components):
         if nk[k] > 1e-6:
             ellipse = _plot_ellipse(means[k], covariances[k], ax, n_std=2.0, edgecolor=colors[k], facecolor='none', lw=2, linestyle='--')
             if f'Gaussian {k+1}' not in plot_handles:
                 plot_handles[f'Gaussian {k+1}'] = ellipse
         else:
             print(f"Skipping ellipse for collapsed component {k+1}")

    # Update legend to include ellipses
    handles, labels = ax.get_legend_handles_labels()
    for label, handle in plot_handles.items():
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    ax.legend(handles, labels)
    ax.set_title('EM Clustering with Estimated Gaussian Distributions (2 std)')

    try:
        plt.savefig(os.path.join(output_dir, 'em_clustering_with_gaussians.png'))
    except Exception as e:
        print(f"Error saving clustering plot: {e}")
    plt.close() # Close the figure

    # Plot and save log-likelihood history
    print("Saving log-likelihood plot...")
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(log_likelihood_history) + 1), log_likelihood_history)
    plt.title('Log-Likelihood Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.grid(True, linestyle='--', alpha=0.5)
    try:
        plt.savefig(os.path.join(output_dir, 'em_log_likelihood.png'))
    except Exception as e:
        print(f"Error saving log-likelihood plot: {e}")
    plt.close()


    print("\nEM Algorithm Finished.")
    return means, covariances, weights, responsibilities, log_likelihood_history