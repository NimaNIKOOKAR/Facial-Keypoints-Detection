import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FEATURE_COLORS = {
    "eye": "red",
    "eyebrow": "blue",
    "nose": "green",
    "mouth": "yellow"
}



def get_face_with_keypoints(df, image_index):

    row = df.loc[image_index]

    image = np.fromstring(
        row["Image"],
        sep=" "
    ).reshape(96, 96)

    keypoint_cols = [
        col for col in df.columns
        if col != "Image"
    ]

    keypoints = (
        row[keypoint_cols]
        .values
        .astype(float)
        .reshape(-1, 2)
    )

    return image, keypoints

def show_face_with_keypoints(df, image_index):
    """ give the image and the keypoints of the image at the given index in the dataframe

    Args:
        df (_type_): _dataframe containing the images and keypoints
        image_index (_type_): index of the image in the dataframe requested to be shown
    """

    image, keypoints = get_face_with_keypoints(df, image_index)

    columns = [col for col in df.columns if col != "Image"]

    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")

    for i in range(0, len(columns), 2):

        x_name = columns[i]
        y_name = columns[i + 1]

        x, y = keypoints[i // 2]

        if np.isnan(x) or np.isnan(y):
            continue

        
        if "eyebrow" in x_name:
            color = FEATURE_COLORS["eyebrow"]

        elif "eye" in x_name:
            color = FEATURE_COLORS["eye"]
        elif "nose" in x_name:
            color = FEATURE_COLORS["nose"]
        elif "mouth" in x_name:
            color = FEATURE_COLORS["mouth"]
        else:
            color = "white"

        
        plt.scatter(x, y, color=color, s=10)

        
        label = x_name.replace("_x", "")
        plt.text(
            x + 1,
            y + 1,
            label,
            color=color,
            fontsize=7
        )

    plt.title(f"Face Keypoints - Image {image_index}")
    plt.axis("off")
    plt.show()


def compare_keypoints(df_original,df_filled,image_indices,figsize_per_row=(7, 3.5),show_labels=False):
    """
    Compare original and imputed facial keypoints side by side of the images at the given indices.

    Circles (o): Original keypoints.
    Crosses (x): Newly imputed keypoints.
    """

    if len(image_indices) == 0:
        raise ValueError("Provide at least one image index.")

    # Create compact comparison grid
    fig, axes = plt.subplots(
        nrows=len(image_indices),
        ncols=2,
        figsize=(
            figsize_per_row[0],
            figsize_per_row[1] * len(image_indices)
        ),
        squeeze=False,
        gridspec_kw={
            "hspace": 0.12,
            "wspace": 0.05
        }
    )

    # Get landmark names
    landmark_names = [
        col[:-2]
        for col in df_original.columns
        if col.endswith("_x")
        and f"{col[:-2]}_y" in df_original.columns
    ]

    def plot_face(ax, df, idx, is_imputed=False):

        row = df.loc[idx]
        original_row = df_original.loc[idx]

        # Extract image
        image = np.fromstring(
            row["Image"],
            sep=" "
        ).reshape(96, 96)

        ax.imshow(image, cmap="gray")

        for name in landmark_names:

            x = row[f"{name}_x"]
            y = row[f"{name}_y"]

            if pd.isna(x) or pd.isna(y):
                continue

            
            if "eyebrow" in name:
                color = FEATURE_COLORS["eyebrow"]

            elif "eye" in name:
                color = FEATURE_COLORS["eye"]

            elif "nose" in name:
                color = FEATURE_COLORS["nose"]

            elif "mouth" in name:
                color = FEATURE_COLORS["mouth"]

            else:
                color = "white"

            was_missing = (
                pd.isna(original_row[f"{name}_x"])
                or pd.isna(original_row[f"{name}_y"])
            )

            if is_imputed and was_missing:

                ax.scatter(
                    x, y,
                    c=color,
                    marker="x",
                    s=45,
                    linewidths=1.5,
                    zorder=4
                )

            else:

                ax.scatter(
                    x, y,
                    c=color,
                    marker="o",
                    s=18,
                    zorder=3
                )

            if show_labels:
                ax.annotate(
                    name,
                    (x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color=color
                )

        ax.set_xlim(0, 96)
        ax.set_ylim(96, 0)
        ax.set_aspect("equal")
        ax.axis("off")

    for i, idx in enumerate(image_indices):

        plot_face(
            axes[i, 0],
            df_original,
            idx,
            is_imputed=False
        )

        plot_face(
            axes[i, 1],
            df_filled,
            idx,
            is_imputed=True
        )

        axes[i, 0].set_title(
            f"Image {idx} - BEFORE",
            fontsize=11,
            pad=5
        )

        axes[i, 1].set_title(
            f"Image {idx} - AFTER",
            fontsize=11,
            pad=5
        )

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.96,
        bottom=0.02
    )

    plt.show()



# ------------------------------------------
# fill with KNN-based geometric alignment
# ------------------------------------------


def get_landmark_names(df) -> list[str]:
    """
    Return landmark names in a consistent order.
    """

    names = [
        col[:-2]
        for col in df.columns
        if col.endswith("_x")
        and f"{col[:-2]}_y" in df.columns
    ]

    return names


def align_landmarks(source, target, mask) -> tuple[np.ndarray, float]:
    """
    Align source landmarks to target landmarks using
    a 2D similarity transformation.

    Only landmarks selected by mask are used to
    calculate the transformation.

    Returns
    -------
    transformed : ndarray
    rmse : float
    """

    P = source[mask]
    Q = target[mask]

    if len(P) < 3:
        return None, np.inf

    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)

    P0 = P - p_mean
    Q0 = Q - q_mean

    
    denom = np.sum(P0 ** 2)

    if denom < 1e-8:
        return None, np.inf

    # Optimal 2D rotation and scale.
    a = np.sum(
        P0[:, 0] * Q0[:, 0]
        + P0[:, 1] * Q0[:, 1]
    ) / denom

    b = np.sum(
        P0[:, 0] * Q0[:, 1]
        - P0[:, 1] * Q0[:, 0]
    ) / denom

    A = np.array([
        [a, -b],
        [b,  a]
    ])

    translation = q_mean - A @ p_mean

    transformed = source @ A.T + translation

    differences = transformed[mask] - Q

    rmse = np.sqrt(
        np.mean(np.sum(differences ** 2, axis=1))
    )

    return transformed, rmse


def fill_keypoints_pose_knn(
    df,
    reference_df=None,
    k=7,
    min_shared=4,
    max_rmse=8.0,
) -> pd.DataFrame:
    if reference_df is None:
        reference_df = df

    same_dataset = reference_df is df
    names = get_landmark_names(df)
    columns = [
        f"{name}_{axis}"
        for name in names
        for axis in ("x", "y")
    ]

    def to_points(frame):
        return frame[columns].to_numpy(dtype=float).reshape(
            len(frame), len(names), 2
        )

    queries = to_points(df)
    references = to_points(reference_df)

    reference_valid = np.isfinite(references).all(axis=2)
    safe_references = np.nan_to_num(references, nan=0.0)
    filled = queries.copy()

    for i, face in enumerate(queries):
        observed = np.isfinite(face).all(axis=1)
        missing_points = np.flatnonzero(~observed)

        if not len(missing_points):
            continue

        # Find references with enough shared observed landmarks.
        shared = reference_valid & observed
        eligible = shared.sum(axis=1) >= min_shared

        if same_dataset:
            eligible[i] = False

        candidates = []

        for j in np.flatnonzero(eligible):
            aligned, error = align_landmarks(
                safe_references[j], face, shared[j]
            )

            if (
                aligned is not None
                and np.isfinite(error)
                and error <= max_rmse
            ):
                candidates.append((error, j, aligned))

        candidates.sort(key=lambda candidate: candidate[0])

        for point in missing_points:
            neighbors = [
                candidate
                for candidate in candidates
                if reference_valid[candidate[1], point]
            ][:k]

            if not neighbors:
                continue

            errors = np.array([error for error, _, _ in neighbors])
            positions = np.array([
                aligned[point] for _, _, aligned in neighbors
            ])

            estimate = np.average(
                positions,
                axis=0,
                weights=1.0 / (errors + 0.5) ** 2,
            )

            missing_axes = ~np.isfinite(face[point])
            filled[i, point, missing_axes] = estimate[missing_axes]

    output = df.copy()
    output[columns] = filled.reshape(len(df), len(columns))
    return output


