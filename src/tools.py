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
    image = np.fromstring(df["Image"][image_index], sep=" ").reshape(96, 96)

    keypoints = df.iloc[image_index, :-1].values.astype("float")
    keypoints = keypoints.reshape(-1, 2)

    return image, keypoints


def show_face_with_keypoints(df, image_index):
    image, keypoints = get_face_with_keypoints(df, image_index)

   
    columns = df.columns[:-1]

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")

    for i in range(0, len(columns), 2):

        x_name = columns[i]
        y_name = columns[i + 1]

        x, y = keypoints[i // 2]

        if np.isnan(x) or np.isnan(y):
            continue

        
        if "eye" in x_name:
            color = FEATURE_COLORS["eye"]
        elif "eyebrow" in x_name:
            color = FEATURE_COLORS["eyebrow"]
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


def fill_missing_keypoints(df):
    """
    Fill missing facial keypoints.
    1. Use left/right symmetry when possible.
    2. Fill remaining NaN values with column median.
    3. Keep original dataframe unchanged.
    """

    df_filled = df.copy()

    
    keypoint_cols = [col for col in df_filled.columns if col != "Image"]

    
    symmetric_pairs = [
        ("left_eye_center_x", "right_eye_center_x"),
        ("left_eye_center_y", "right_eye_center_y"),

        ("left_eye_inner_corner_x", "right_eye_inner_corner_x"),
        ("left_eye_inner_corner_y", "right_eye_inner_corner_y"),

        ("left_eye_outer_corner_x", "right_eye_outer_corner_x"),
        ("left_eye_outer_corner_y", "right_eye_outer_corner_y"),

        ("left_eyebrow_inner_end_x", "right_eyebrow_inner_end_x"),
        ("left_eyebrow_inner_end_y", "right_eyebrow_inner_end_y"),

        ("left_eyebrow_outer_end_x", "right_eyebrow_outer_end_x"),
        ("left_eyebrow_outer_end_y", "right_eyebrow_outer_end_y"),

        ("mouth_left_corner_x", "mouth_right_corner_x"),
        ("mouth_left_corner_y", "mouth_right_corner_y"),
    ]

    
    for left_col, right_col in symmetric_pairs:
        if left_col in df_filled.columns and right_col in df_filled.columns:
            df_filled[left_col] = df_filled[left_col].fillna(df_filled[right_col])
            df_filled[right_col] = df_filled[right_col].fillna(df_filled[left_col])

    
    for col in keypoint_cols:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())

    return df_filled


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