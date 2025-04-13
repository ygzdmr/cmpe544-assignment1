import numpy as np
from src.classification.preprocessing import load_data, extract_features
from src.classification.knn import KNN
from src.classification.nb import NaiveBayes
from src.classification.logreg import MultinomialLogisticRegression
from src.classification.analysis import investigate_feature_space
from src.em.gmm import run_em_algorithm

QUICKDRAW_DATA_PATH = 'data/quickdraw/'
EM_DATA_PATH = 'data/dataset.npy' # Path for the EM dataset
EM_OUTPUT_DIR = 'results/em/'     # Output directory for EM plots
EM_N_COMPONENTS = 3               # Number of components for EM

def run_classification():
    # Load data
    train_images, train_labels, test_images, test_labels, class_names = load_data('data/quickdraw/')

    # Extract features
    features_train_dict, features_test_dict, feature_keys = extract_features(train_images, test_images)
    if not feature_keys:
        print("No features extracted. Exiting.")
        return

    # Use the first feature set
    feature_key = feature_keys[0]
    X_train = features_train_dict[feature_key]
    X_test = features_test_dict[feature_key]

    summary = investigate_feature_space(X_train, train_labels, class_names)
    print(summary)

    # KNN
    knn = KNN(k=3)
    knn.fit(X_train, train_labels)
    knn_preds = knn.predict(X_test)
    knn_acc = np.mean(knn_preds == test_labels)
    print(f"KNN Accuracy: {knn_acc:.4f}")

    # Naive Bayes
    nb = NaiveBayes()
    nb.fit(X_train, train_labels)
    nb_preds = nb.predict(X_test)
    nb_acc = np.mean(nb_preds == test_labels)
    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")

    # Logistic Regression
    logreg = MultinomialLogisticRegression(lr=0.05, n_iter=3000, l2=1e-4, verbose=True)
    logreg.fit(X_train, train_labels)
    logreg_preds = logreg.predict(X_test).flatten()
    logreg_acc = np.mean(logreg_preds == test_labels)
    print(f"Logistic Regression Accuracy: {logreg_acc:.4f}")


def run_em():
    print("\n--- Running Expectation Maximization Part ---")
    # Load EM data
    try:
        em_data = np.load(EM_DATA_PATH)
        print(f"Loaded EM dataset from '{EM_DATA_PATH}'")
    except FileNotFoundError:
        print(f"Error: EM Dataset file not found at '{EM_DATA_PATH}'")
        print("Skipping EM part.")
        return
    except Exception as e:
        print(f"Error loading EM dataset: {e}")
        print("Skipping EM part.")
        return

    # Run EM Algorithm
    em_results = run_em_algorithm(
        data=em_data,
        n_components=EM_N_COMPONENTS,
        output_dir=EM_OUTPUT_DIR
    )

    # Estimated parameters
    if em_results:
        means, covariances, weights, _, _ = em_results
        print("\nEM Algorithm Final Estimated Parameters:")
        for k in range(len(weights)):
            print(f"\nComponent {k+1}:")
            print(f"  Weight (pi_k): {weights[k]:.4f}")
            print(f"  Mean (mu_k): {means[k]}")
            cov_str = np.array2string(covariances[k], prefix=' ' * 4, precision=4, suppress_small=True)
            print(f"  Covariance (Sigma_k):\n{cov_str}")
    else:
        print("EM algorithm did not return results.")


if __name__ == "__main__":
    run_em()
    run_classification()