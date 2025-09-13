#!/usr/bin/env python3
import os, argparse, logging, traceback
from multiprocessing import cpu_count

import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans

log = logging.getLogger("train_index")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def main():
    ap = argparse.ArgumentParser(
        description="Train FAISS IVF index from RVC features (3_featureXXX) and save into logs/<exp>."
    )
    ap.add_argument("--exp", required=True, help="Experiment name, e.g. 222")
    ap.add_argument("--feat-dir", default=None,
                    help="Path to 3_featureXXX directory. Defaults to logs/<exp>/3_feature768.")
    ap.add_argument("--feat-dim", type=int, default=768, help="Feature dimension (e.g. 768 for v2).")
    ap.add_argument("--kmeans", type=int, default=0,
                    help="If >0, downsample with MiniBatchKMeans to this many centers (e.g. 10000).")
    ap.add_argument("--batch-add", type=int, default=8192, help="Batch size for index.add().")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir for index files. Defaults to logs/<exp>.")
    args = ap.parse_args()

    exp_dir = os.path.join("logs", args.exp)
    feat_dir = args.feat_dir or os.path.join(exp_dir, f"3_feature{args.feat-dim if args.feat_dim!=768 else 768}")
    out_dir  = args.out_dir or exp_dir

    if not os.path.isdir(feat_dir):
        raise SystemExit(f"Feature dir not found: {feat_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # Load all npy feature files
    log.info(f"Loading features from: {feat_dir}")
    names = sorted(os.listdir(feat_dir))
    npys = []
    for name in names:
        if not name.endswith(".npy"):
            continue
        arr = np.load(os.path.join(feat_dir, name))
        if arr.shape[-1] != args.feat_dim:
            raise SystemExit(f"Feature dim mismatch in {name}: got {arr.shape[-1]}, expected {args.feat_dim}")
        npys.append(arr)
    if not npys:
        raise SystemExit("No .npy feature files found.")

    big = np.concatenate(npys, axis=0)
    idx = np.arange(big.shape[0])
    np.random.shuffle(idx)
    big = big[idx]
    log.info(f"Feature matrix: {big.shape}")

    # Optional downsample with MiniBatchKMeans
    if args.kmeans and big.shape[0] > args.kmeans:
        log.info(f"Downsampling with MiniBatchKMeans to {args.kmeans} centers …")
        try:
            big = MiniBatchKMeans(
                n_clusters=args.kmeans,
                verbose=True,
                batch_size=256 * cpu_count(),
                compute_labels=False,
                init="random",
            ).fit(big).cluster_centers_
            log.info(f"Downsampled to: {big.shape}")
        except Exception:
            log.warning("KMeans failed, proceeding without downsampling:\n" + traceback.format_exc())

    # Save the consolidated features (handy for reuse)
    big_path = os.path.join(out_dir, f"big_src_feature_{args.exp}.npy")
    np.save(big_path, big)
    log.info(f"Saved features: {big_path}")

    # Train IVF,Flat index
    n_ivf = min(int(16 * np.sqrt(big.shape[0])), max(1, big.shape[0] // 39))
    factory = f"IVF{n_ivf},Flat"
    log.info(f"Building index: {factory}, dim={args.feat_dim}")
    index = faiss.index_factory(args.feat_dim, factory)
    faiss.extract_index_ivf(index).nprobe = 1

    log.info("Training index …")
    index.train(big)

    trained_path = os.path.join(out_dir, f"trained_{factory.replace(',', '_')}.index")
    faiss.write_index(index, trained_path)
    log.info(f"Saved trained index: {trained_path}")

    # Add vectors
    log.info("Adding vectors …")
    bs = int(args.batch_add)
    for i in range(0, big.shape[0], bs):
        index.add(big[i:i+bs])

    added_path = os.path.join(out_dir, f"added_{factory.replace(',', '_')}.index")
    faiss.write_index(index, added_path)
    log.info(f"Saved added index:   {added_path}")

    log.info("Done.")

if __name__ == "__main__":
    main()
