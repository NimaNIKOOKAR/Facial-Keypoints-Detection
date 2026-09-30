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
    image, keypoints = get_face_with_keypoints(df, image_index)

   
    columns = [col for col in df.columns if col != "Image"]

    plt.figure(figsize=(8, 8))
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


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def compare_keypoints(
    df_original,
    df_filled,
    image_indices,
    figsize_per_row=(12, 5),
    show_labels=False
):
    """
    Compare original and imputed facial keypoints.

    Parameters
    ----------
    df_original : pd.DataFrame
        Original dataset containing missing keypoints.

    df_filled : pd.DataFrame
        Dataset after imputation.

    image_indices : list
        List of DataFrame indices to visualize.

    figsize_per_row : tuple
        Figure width and height per comparison row.

    show_labels : bool
        Whether to display landmark names.
    """

    if len(image_indices) == 0:
        raise ValueError("Provide at least one image index.")

    feature_colors = {
        "eye": "red",
        "eyebrow": "blue",
        "nose": "green",
        "mouth": "yellow"
    }

    fig, axes = plt.subplots(
        nrows=len(image_indices),
        ncols=2,
        figsize=(
            figsize_per_row[0],
            figsize_per_row[1] * len(image_indices)
        ),
        squeeze=False
    )

    def plot_face(ax, df, idx, is_imputed=False):

        row = df.loc[idx]

        image = np.fromstring(
            row["Image"],
            sep=" "
        ).reshape(96, 96)

        ax.imshow(image, cmap="gray")

        landmark_names = [
            col[:-2]
            for col in df.columns
            if col.endswith("_x")
            and f"{col[:-2]}_y" in df.columns
        ]

        for name in landmark_names:

            x = row[f"{name}_x"]
            y = row[f"{name}_y"]

            if pd.isna(x) or pd.isna(y):
                continue

            if "eyebrow" in name:
                color = feature_colors["eyebrow"]

            elif "eye" in name:
                color = feature_colors["eye"]

            elif "nose" in name:
                color = feature_colors["nose"]

            elif "mouth" in name:
                color = feature_colors["mouth"]

            else:
                color = "white"

            # Distinguish original and newly imputed points.
            original_row = df_original.loc[idx]

            was_missing = (
                pd.isna(original_row[f"{name}_x"])
                or pd.isna(original_row[f"{name}_y"])
            )

            if is_imputed and was_missing:

                # Newly reconstructed landmark.
                ax.scatter(
                    x, y,
                    c=color,
                    marker="x",
                    s=45,
                    linewidths=1.5,
                    zorder=4
                )

            else:

                # Originally observed landmark.
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
        ax.axis("off")

    # -------------------------------------
    # Create comparison grid
    # -------------------------------------

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
            fontsize=12
        )

        axes[i, 1].set_title(
            f"Image {idx} - AFTER",
            fontsize=12
        )

    plt.tight_layout()
    plt.show()





def fill_missing_keypoints_geometric(df, reference_df=None):
    """
    Fill missing facial keypoints using:

    1. Eye-center geometry.
    2. Facial symmetry around a vertical midline.
    3. Learned relative distances between landmarks.
    4. Median fallback.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the keypoints to impute.

    reference_df : pd.DataFrame, optional
        Original training observations used to estimate
        geometric offsets and fallback medians.
        Defaults to df.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with missing keypoints filled.
    """

    if reference_df is None:
        reference_df = df

    filled = df.copy()
    reference = reference_df.copy()

    keypoint_cols = [
        col for col in df.columns if col != "Image"
    ]

    filled[keypoint_cols] = filled[keypoint_cols].astype(float)
    reference[keypoint_cols] = (
        reference[keypoint_cols].astype(float)
    )

    # ------------------------------------------
    # Helper 1: Learn relative coordinate offsets
    # ------------------------------------------

    def learned_offset(target, anchor):
        """
        Learn the median signed distance between
        two landmarks from original observations.
        """

        if target not in reference.columns:
            return np.nan

        if anchor not in reference.columns:
            return np.nan

        valid = reference[[target, anchor]].dropna()

        if valid.empty:
            return np.nan

        return (valid[target] - valid[anchor]).median()

    # ------------------------------------------
    # Helper 2: Eye center from eye corners
    # ------------------------------------------

    def estimate_eye_center(row, side):

        for axis in ("x", "y"):

            center = f"{side}_eye_center_{axis}"
            inner = f"{side}_eye_inner_corner_{axis}"
            outer = f"{side}_eye_outer_corner_{axis}"

            if not all(
                col in row.index
                for col in (center, inner, outer)
            ):
                continue

            if pd.notna(row[center]):
                continue

            if pd.notna(row[inner]) and pd.notna(row[outer]):

                midpoint = (row[inner] + row[outer]) / 2

                # Learn the typical difference between
                # the annotated center and corner midpoint.

                valid = reference[
                    [center, inner, outer]
                ].dropna()

                if not valid.empty:

                    residual = (
                        valid[center]
                        - (valid[inner] + valid[outer]) / 2
                    ).median()

                else:
                    residual = 0.0

                row[center] = midpoint + residual

        return row

    # ------------------------------------------
    # Helper 3: Estimate facial symmetry axis
    # ------------------------------------------

    def get_face_axis(row):

        # Prefer both eye centers when available.

        left = row.get("left_eye_center_x", np.nan)
        right = row.get("right_eye_center_x", np.nan)

        if pd.notna(left) and pd.notna(right):
            return (left + right) / 2

        # Fallback to the nose position.

        nose = row.get("nose_tip_x", np.nan)

        if pd.notna(nose):
            return nose

        # Alternative: center of the mouth.

        mouth = row.get(
            "mouth_center_top_lip_x",
            np.nan
        )

        if pd.notna(mouth):
            return mouth

        # Final fallback for a 96x96 image.
        return 47.5

    # ------------------------------------------
    # Helper 4: Symmetric keypoint imputation
    # ------------------------------------------

    symmetric_pairs = [
        ("left_eye_center", "right_eye_center"),

        (
            "left_eye_inner_corner",
            "right_eye_inner_corner"
        ),

        (
            "left_eye_outer_corner",
            "right_eye_outer_corner"
        ),

        (
            "left_eyebrow_inner_end",
            "right_eyebrow_inner_end"
        ),

        (
            "left_eyebrow_outer_end",
            "right_eyebrow_outer_end"
        ),

        (
            "mouth_left_corner",
            "mouth_right_corner"
        )
    ]

    def estimate_symmetric_points(row, face_axis):

        for left, right in symmetric_pairs:

            for target, source in [
                (left, right),
                (right, left)
            ]:

                tx = f"{target}_x"
                ty = f"{target}_y"

                sx = f"{source}_x"
                sy = f"{source}_y"

                if not all(
                    col in row.index
                    for col in (tx, ty, sx, sy)
                ):
                    continue

                # X: Reflect around the facial midline.

                if pd.isna(row[tx]) and pd.notna(row[sx]):
                    row[tx] = 2 * face_axis - row[sx]

                # Y: Use the observed opposite landmark,
                # plus a typical signed vertical offset.

                if pd.isna(row[ty]) and pd.notna(row[sy]):

                    dy = learned_offset(ty, sy)

                    if pd.isna(dy):
                        dy = 0.0

                    row[ty] = row[sy] + dy

        return row

    # ------------------------------------------
    # Helper 5: Lip position from relative distance
    # ------------------------------------------

    def estimate_bottom_lip(row):

        for axis in ("x", "y"):

            bottom = f"mouth_center_bottom_lip_{axis}"
            top = f"mouth_center_top_lip_{axis}"

            if bottom not in row.index or top not in row.index:
                continue

            if pd.notna(row[bottom]):
                continue

            if pd.isna(row[top]):
                continue

            offset = learned_offset(bottom, top)

            if pd.notna(offset):
                row[bottom] = row[top] + offset

        return row

    # ------------------------------------------
    # Apply hierarchical imputation
    # ------------------------------------------

    for idx in filled.index:

        row = filled.loc[idx].copy()

        # 1. Estimate eye centers where both corners exist.
        row = estimate_eye_center(row, "left")
        row = estimate_eye_center(row, "right")

        # 2. Estimate facial midline.
        face_axis = get_face_axis(row)

        # 3. Apply proper symmetry.
        row = estimate_symmetric_points(row, face_axis)

        # 4. Recalculate eye centers if the corner
        # coordinates were recovered through symmetry.
        row = estimate_eye_center(row, "left")
        row = estimate_eye_center(row, "right")

        # 5. Estimate lower lip relative to upper lip.
        row = estimate_bottom_lip(row)

        filled.loc[idx, keypoint_cols] = row[keypoint_cols]

    # ------------------------------------------
    # Last resort: Original column medians
    # ------------------------------------------

    medians = reference[keypoint_cols].median()

    filled[keypoint_cols] = (
        filled[keypoint_cols].fillna(medians)
    )

    return filled


# ------------------------------------------
# fill with KNN-based geometric alignment
# ------------------------------------------


def get_landmark_names(df):
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


def align_landmarks(source, target, mask):
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

    # Reject degenerate configurations.
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
    max_rmse=8.0
):
    """
    Pose-aware facial keypoint imputation using
    aligned nearest neighbors.

    df:
        Dataset to impute.

    reference_df:
        Original training data with observed labels.
        Defaults to df.

    k:
        Number of nearest aligned reference faces.

    min_shared:
        Minimum number of shared landmark pairs.

    max_rmse:
        Maximum acceptable alignment error in pixels.

    Important:
        Reference landmarks must be original
        observations, not previously imputed labels.
    """

    if reference_df is None:
        reference_df = df

    output = df.copy()

    names = get_landmark_names(df)

    columns = [
        coordinate
        for name in names
        for coordinate in (f"{name}_x", f"{name}_y")
    ]

    # Convert keypoints to (N, landmarks, 2).

    query_points = (
        df[columns]
        .to_numpy(dtype=float)
        .reshape(-1, len(names), 2)
    )

    reference_points = (
        reference_df[columns]
        .to_numpy(dtype=float)
        .reshape(-1, len(names), 2)
    )

    # Use only original observed keypoint pairs.
    reference_masks = np.isfinite(reference_points).all(axis=2)

    reconstructed = query_points.copy()

    same_dataset = reference_df is df

    for i, face in enumerate(query_points):

        query_mask = np.isfinite(face).all(axis=1)

        missing_mask = ~query_mask

        if not missing_mask.any():
            continue

        candidates = []

        for j, reference in enumerate(reference_points):

            # Never use the exact same row as its own
            # reference when imputing the same dataset.
            if same_dataset and i == j:
                continue

            common_mask = (
                query_mask & reference_masks[j]
            )

            if common_mask.sum() < min_shared:
                continue

            # Alignment requires finite coordinates.
            # Temporarily replace missing reference
            # coordinates; they are never used for fitting.

            safe_reference = np.nan_to_num(
                reference,
                nan=0.0
            )

            aligned, error = align_landmarks(
                safe_reference,
                face,
                common_mask
            )

            if aligned is None:
                continue

            if not np.isfinite(error):
                continue

            if error > max_rmse:
                continue

            candidates.append(
                (error, aligned, reference_masks[j])
            )

        if not candidates:
            continue

        # Select best aligned neighbors.
        candidates.sort(key=lambda x: x[0])

        # Select neighbors separately for each
        # missing landmark. This avoids including
        # references whose target label is missing.

        for point_idx in np.where(missing_mask)[0]:

            available = [
                candidate
                for candidate in candidates
                if candidate[2][point_idx]
            ][:k]

            if not available:
                continue

            errors = np.array([
                item[0] for item in available
            ])

            predictions = np.array([
                item[1][point_idx]
                for item in available
            ])

            # Higher weight for more similar faces.
            weights = 1.0 / (errors + 0.5) ** 2
            weights /= weights.sum()

            estimate = np.average(
                predictions,
                axis=0,
                weights=weights
            )

            # Preserve every originally observed
            # coordinate, including partially
            # observed x/y pairs.

            missing_coordinates = ~np.isfinite(
                reconstructed[i, point_idx]
            )

            reconstructed[
                i, point_idx, missing_coordinates
            ] = estimate[missing_coordinates]

    # Restore original DataFrame structure.
    output[columns] = reconstructed.reshape(
        len(output),
        -1
    )

    return output


def detect_outliers(df):
    coord_cols = df.columns.drop("Image")

    outlier_mask = pd.Series(False, index=df.index)

    for col in coord_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_mask |= (
            (df[col] < lower) |
            (df[col] > upper)
        )

    return outlier_mask