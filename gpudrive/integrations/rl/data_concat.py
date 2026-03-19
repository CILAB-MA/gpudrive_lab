import os
import random
import gc

import numpy as np
import h5py
from tqdm import tqdm


def _extract_index(filename: str) -> int:
    # trajectory_{index}.h5 / trajectory_{index}.npz
    return int(filename.split("_")[1].split(".")[0])


def _compute_unit(sorted_ids: list[int]) -> int:
    if len(sorted_ids) < 2:
        return 1
    diffs = [sorted_ids[i + 1] - sorted_ids[i] for i in range(len(sorted_ids) - 1)]
    return min(diffs) if diffs else 1


def _load_h5(path: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = f[k][()]
    return out


def _load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.keys()}


def _save_h5(path: str, arrays: dict[str, np.ndarray]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in arrays.items():
            if v is None:
                continue
            arr = np.asarray(v)
            f.create_dataset(
                k,
                data=arr,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=True,
            )


def run(
    save_path: str,
    save_name: str,
    subset_path: str,
    num_scenes: int,
    seed: int = 42,
    prefer_h5: bool = True,
):
    """
    Concatenate per-batch RL storage files into a single dataset.

    Expects files created by `gpudrive/integrations/rl/storage.py`:
      - trajectory_{index}.h5 (or .npz)
      - global/global_trajectory_{index}.h5 (or .npz)
      - id/id_trajectory_{index}.h5 (or .npz)
      - label/label_trajectory_{index}.h5 (or .npz) [optional]

    Output:
      - {save_path}/{save_name}
      - {save_path}/global/global_{save_name}
      - {save_path}/id/id_{save_name}
      - {save_path}/label/label_{save_name}  [if labels exist]
    """
    if prefer_h5:
        exts = (".h5", ".npz")
    else:
        exts = (".npz", ".h5")

    def _list_traj_files(ext: str) -> list[str]:
        return [
            f
            for f in os.listdir(subset_path)
            if f.startswith("trajectory_") and f.endswith(ext)
        ]

    traj_files = []
    chosen_ext = None
    for ext in exts:
        traj_files = _list_traj_files(ext)
        if traj_files:
            chosen_ext = ext
            break

    if not traj_files or chosen_ext is None:
        raise FileNotFoundError(
            f"No trajectory files found in {subset_path} (searched {exts})."
        )

    ids = sorted([_extract_index(f) for f in traj_files])
    unit = _compute_unit(ids)
    print(f"[INFO] file_ext={chosen_ext} unit={unit} files={len(ids)}")

    need_files = (num_scenes + unit - 1) // unit
    if need_files > len(ids):
        print(
            f"[WARN] Requested {need_files} files but only {len(ids)} available. Using all."
        )
        need_files = len(ids)

    random.seed(seed)
    chosen_ids = sorted(random.sample(ids, need_files))
    chosen_files = [f"trajectory_{i}{chosen_ext}" for i in chosen_ids]

    def _load(path: str) -> dict[str, np.ndarray]:
        if path.endswith(".h5"):
            return _load_h5(path)
        return _load_npz(path)

    # Lists to concatenate
    traj_parts: dict[str, list[np.ndarray]] = {}
    global_parts: dict[str, list[np.ndarray]] = {}
    id_parts: dict[str, list[np.ndarray]] = {}
    label_parts: dict[str, list[np.ndarray]] = {}
    has_labels = os.path.isdir(os.path.join(subset_path, "label"))

    total_rows = 0

    for filename in tqdm(chosen_files, desc="Loading shards", ncols=100):
        shard_path = os.path.join(subset_path, filename)
        shard = _load(shard_path)

        # Trajectory shard
        for k, v in shard.items():
            traj_parts.setdefault(k, []).append(v)
        total_rows += int(shard["obs"].shape[0])

        del shard
        gc.collect()

        # Global shard
        global_filename = f"global_{filename}"
        global_path = os.path.join(subset_path, "global", global_filename)
        g = _load(global_path)
        for k, v in g.items():
            global_parts.setdefault(k, []).append(v)
        del g
        gc.collect()

        # ID shard
        id_filename = f"id_{filename}"
        id_path = os.path.join(subset_path, "id", id_filename)
        i = _load(id_path)
        for k, v in i.items():
            id_parts.setdefault(k, []).append(v)
        del i
        gc.collect()

        # Label shard (optional)
        if has_labels:
            label_filename = f"label_{filename}"
            label_path = os.path.join(subset_path, "label", label_filename)
            if os.path.exists(label_path):
                lbl = _load(label_path)
                for k, v in lbl.items():
                    label_parts.setdefault(k, []).append(v)
                del lbl
                gc.collect()
            else:
                print(f"[WARN] Missing {label_path}. Disabling labels for output.")
                label_parts.clear()
                has_labels = False

    # Concatenate and save
    def _concat(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for k, vs in parts.items():
            if not vs:
                continue
            out[k] = np.concatenate(vs, axis=0)
        return out

    traj_out = _concat(traj_parts)
    global_out = _concat(global_parts)
    id_out = _concat(id_parts)
    label_out = _concat(label_parts) if (has_labels and label_parts) else {}

    # Basic alignment sanity checks (first dim must match)
    n = traj_out["obs"].shape[0]
    for name, d in [
        ("global", global_out),
        ("id", id_out),
        ("label", label_out),
    ]:
        if not d:
            continue
        first_key = next(iter(d.keys()))
        if d[first_key].shape[0] != n:
            raise ValueError(
                f"Concat mismatch: traj={n} but {name}.{first_key}={d[first_key].shape[0]}"
            )

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "global"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "id"), exist_ok=True)
    if label_out:
        os.makedirs(os.path.join(save_path, "label"), exist_ok=True)

    out_main = os.path.join(save_path, save_name)
    out_global = os.path.join(save_path, "global", f"global_{save_name}")
    out_id = os.path.join(save_path, "id", f"id_{save_name}")
    out_label = os.path.join(save_path, "label", f"label_{save_name}")

    print(f"[INFO] total_rows={total_rows} -> saving to {out_main}")
    _save_h5(out_main, traj_out)
    _save_h5(out_global, global_out)
    _save_h5(out_id, id_out)
    if label_out:
        _save_h5(out_label, label_out)
        print(f"[INFO] saved labels to {out_label}")
    print("[INFO] done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, default=100)
    parser.add_argument("--num-scene", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset", type=str, default="validation", choices=["training", "validation", "testing"]
    )
    parser.add_argument(
        "--base",
        type=str,
        default="/data/after_cvpr/linear_probe_data",
        help="Base directory containing scene_{N}/...",
    )
    parser.add_argument(
        "--prefer-npz",
        action="store_true",
        help="If set, prefer reading .npz shards over .h5 when both exist.",
    )
    args = parser.parse_args()

    scene_key = f"scene_{args.scene}"
    save_path = os.path.join(args.base, scene_key)
    subset_path = os.path.join(save_path, f"{args.dataset}_rl_data")
    save_name = f"{args.dataset}_trajectory_{args.num_scene}.h5"

    run(
        save_path=save_path,
        save_name=save_name,
        subset_path=subset_path,
        num_scenes=args.num_scene,
        seed=args.seed,
        prefer_h5=not args.prefer_npz,
    )
