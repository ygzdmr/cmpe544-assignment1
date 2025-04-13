import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances, silhouette_score

def investigate_feature_space(X, y, class_names, output_dir='results/analysis'):
    os.makedirs(output_dir, exist_ok=True)
    labels = np.array(y)
    classes = np.unique(labels)
    C = len(classes)

    # 1. LDA scatter (first two components)
    if X.shape[1] >= 2:
        plt.figure(figsize=(6, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, C))
        for idx, c in enumerate(classes):
            pts = X[labels == c]
            plt.scatter(pts[:, 0], pts[:, 1],
                        s=6, alpha=.6, color=colors[idx],
                        label=class_names.get(c, f'Class {c}'))
        plt.title('LDA feature space (first 2 dimensions)')
        plt.xlabel('LD 1')
        plt.ylabel('LD 2')
        plt.legend(markerscale=2, fontsize=8)
        plt.grid(alpha=.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lda_scatter.png'), dpi=300)
        plt.close()

    # 2. Intra‑class spread  &  Inter‑class centroid distances
    centroids = np.vstack([X[labels == c].mean(axis=0) for c in classes])
    inter_mat = pairwise_distances(centroids)          # (C, C)

    intra_dict = {}
    for c in classes:
        Xi = X[labels == c]
        mu = centroids[c]
        dists = np.linalg.norm(Xi - mu, axis=1)
        intra_dict[class_names.get(c, str(c))] = {
            'mean': dists.mean(),
            'std':  dists.std()
        }

    # 4. Heat‑map of inter‑class distances
    plt.figure(figsize=(4.5, 4))
    im = plt.imshow(inter_mat, cmap='viridis')
    plt.colorbar(im, fraction=.046, pad=.04)
    plt.xticks(classes, [class_names.get(c, c) for c in classes],
               rotation=45, ha='right')
    plt.yticks(classes, [class_names.get(c, c) for c in classes])
    plt.title('Inter‑class centroid distance matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inter_class_heatmap.png'), dpi=300)
    plt.close()

    # Save summary
    txt_path = os.path.join(output_dir, 'intra_inter_stats.txt')
    with open(txt_path, 'w') as f:
        f.write('INTRA‑CLASS average ± std distance to centroid\n')
        for k, v in intra_dict.items():
            f.write(f'  {k:<10}: {v["mean"]:.4f} ± {v["std"]:.4f}\n')
        f.write('\nINTER‑CLASS centroid distance matrix (rows/cols match class order)\n')
        np.savetxt(f, inter_mat, fmt='%.4f')
    print(f"Analysis results saved in {output_dir}")

    summary = {
        'intra': intra_dict,
        'inter': inter_mat
    }
    return summary
