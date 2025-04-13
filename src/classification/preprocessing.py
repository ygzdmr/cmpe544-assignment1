import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
from scipy import ndimage
from skimage.feature import hog

def load_data(data_path='data/quickdraw/'):
    """
    Loads the QuickDraw dataset from given .npy files.

    Args:
        data_path (str, optional): Path to the directory containing the .npy files.
            Defaults to 'data/quickdraw/'.
    """

    print("Loading data...")
    try:
        CLASS_NAMES = {
            0: 'Rabbit', 1: 'Yoga', 2: 'Hand', 3: 'Snowman', 4: 'Motorbike'
        }

        train_images = np.load(f'{data_path}train_images.npy')
        train_labels = np.load(f'{data_path}train_labels.npy')
        test_images = np.load(f'{data_path}test_images.npy')
        test_labels = np.load(f'{data_path}test_labels.npy')

        unique_labels = np.unique(train_labels);
        print(f"  Found labels: {unique_labels}")
        if len(unique_labels) != len(CLASS_NAMES) or not all(l in CLASS_NAMES for l in unique_labels):
            print("  Warning: Labels in data don't match default CLASS_NAMES. Generating generic names.")
            CLASS_NAMES = {label: f"Class {label}" for label in unique_labels}

        return train_images, train_labels, test_images, test_labels, CLASS_NAMES

    except FileNotFoundError:
        print(f"\nError: Data files not found in directory '{data_path}'.")
        print("  Please make sure the .npy files are in the correct directory.")
        exit()
    except Exception as e:
        print(f"An error occurred loading data: {e}")
        exit()


# Geometric clean‑up
def _clean_image(img, thresh=200):
    """Return a 28×28 float32 image after binarise, crop‑resize‑pad,
       deskew and centre‑of‑mass shift."""
    # 1) binarise & invert
    bw = (img < thresh).astype(np.uint8)

    # 2) tight crop > resize 20×20 > pad to 28×28
    ys, xs = np.where(bw)
    if len(xs):
        bw = bw[ys.min():ys.max()+1, xs.min():xs.max()+1]
    bw = cv2.resize(bw, (20, 20), interpolation=cv2.INTER_NEAREST)
    bw = np.pad(bw, ((4, 4), (4, 4)), 'constant', constant_values=0)

    # 3) deskew
    m = cv2.moments(bw)
    if abs(m['mu02']) > 1e-2:
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * 28 * skew], [0, 1, 0]])
        bw = cv2.warpAffine(bw, M, (28, 28),
                            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)

    # 4) centre‑of‑mass shift
    cy, cx = ndimage.center_of_mass(bw)
    M = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
    bw = cv2.warpAffine(bw, M, (28, 28), flags=cv2.INTER_NEAREST)

    return bw.astype(np.float32)


# FEATURE BLOCKS FOR ONE IMAGE
def _features_from_image(img):
    """Return a feature vector (HOG + Zoning + Hu)."""
    # HOG
    h = hog(img,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            orientations=9,
            feature_vector=True)

    # Hu
    m = cv2.HuMoments(cv2.moments(img)).flatten()

    return np.hstack([h, m]).astype(np.float32)


# Preprocessing Dataset
def preprocess(train_images, test_images):
    print("  Preprocessing...")

    train_feat_list, test_feat_list = [], []

    t_start_preproc = time.time()
    for i, im in enumerate(train_images):
        if (i+1) % 1000 == 0:
            print(f"    Processing train image {i+1}/{len(train_images)}...")
        train_feat_list.append(_features_from_image(_clean_image(im)))

    for i, im in enumerate(test_images):
        if (i+1) % 1000 == 0:
            print(f"    Processing test image {i+1}/{len(test_images)}...")
        test_feat_list.append(_features_from_image(_clean_image(im)))

    X_train_raw = np.vstack(train_feat_list).astype(np.float32)
    X_test_raw  = np.vstack(test_feat_list).astype(np.float32)
    t_end_preproc = time.time()

    print(f"    Preprocessing finished in {t_end_preproc - t_start_preproc:.2f} s.")
    print(f"    Raw feature shapes  >  train {X_train_raw.shape}  |  "
          f"test {X_test_raw.shape}")
    return X_train_raw, X_test_raw


# FEATURE EXTRACTION
def extract_features(train_images,test_images,data_path='data/quickdraw/'):
    # 1. Preprocessed images
    X_train_raw, X_test_raw = preprocess(train_images, test_images)

    # 2. StandardScaler
    print("  Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)
    print("    Scaling complete.")

    # 3. Load training labels
    print(f"  Loading training labels for LDA from {data_path}train_labels.npy...")
    try:
        train_labels = np.load(f'{data_path}train_labels.npy')
        print(f"    Labels loaded successfully. Shape: {train_labels.shape}")
    except FileNotFoundError:
        print(f"\nError: Could not find train_labels.npy in '{data_path}'.")
        print("  LDA requires training labels to be fit.")
        print("  Please ensure 'train_labels.npy' is in the specified data_path.")
        exit()
    except Exception as e:
        print(f"An error occurred loading training labels: {e}")
        exit()


    # 4. LDA
    n_classes = len(np.unique(train_labels))
    n_features = X_train_scaled.shape[1]
    n_lda_components = min(n_features, n_classes - 1)
    print(f"  Applying LDA to {n_lda_components} components...")
    lda = LinearDiscriminantAnalysis(n_components=n_lda_components)

    t0 = time.time()
    # Fit LDA using the loaded training labels
    lda.fit(X_train_scaled, train_labels)
    print(f"    LDA fit in {time.time() - t0:.2f} s")

    # Transform both training and test data
    X_train_final = lda.transform(X_train_scaled)
    X_test_final  = lda.transform(X_test_scaled)

    print(f"    Feature shapes >  train {X_train_final.shape}  |  "
          f"test {X_test_final.shape}")

    # 5. pack into the dicts format
    features_train, features_test, feature_keys_order = {}, {}, []
    key = f'handcrafted_lda{n_lda_components}'
    features_train[key] = X_train_final
    features_test[key]  = X_test_final
    feature_keys_order.append(key)

    return features_train, features_test, feature_keys_order