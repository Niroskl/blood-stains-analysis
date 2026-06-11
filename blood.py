import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==================================
# Page
# ==================================
st.set_page_config(
    page_title="Blood Drop Analyzer",
    layout="wide"
)

st.title("🩸 Blood Drop Impact Angle")

st.write(
    "העלה תמונה או צלם תמונה והמערכת תזהה את טיפת הדם ותחשב זווית פגיעה."
)

# ==================================
# Upload OR Camera
# ==================================
uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

captured = st.camera_input(
    "Take Photo"
)

image_source = None

if captured is not None:
    image_source = captured

elif uploaded is not None:
    image_source = uploaded

# ==================================
# Process
# ==================================
if image_source is not None:

    image = Image.open(image_source)

    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

    elif img.shape[2] == 4:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2BGR
        )

    else:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

    original = img.copy()

    # ==========================
    # HSV Detection
    # ==========================
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    lower1 = np.array(
        [0, 50, 50]
    )

    upper1 = np.array(
        [10, 255, 255]
    )

    lower2 = np.array(
        [160, 50, 50]
    )

    upper2 = np.array(
        [180, 255, 255]
    )

    mask = (
        cv2.inRange(
            hsv,
            lower1,
            upper1
        )
        +
        cv2.inRange(
            hsv,
            lower2,
            upper2
        )
    )

    # ==========================
    # Noise cleanup
    # ==========================
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    mask = cv2.erode(
        mask,
        kernel,
        iterations=1
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=2
    )

    # ==========================
    # Contours
    # ==========================
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_score = -1

    for c in contours:

        area = cv2.contourArea(c)

        if area < 200:
            continue

        perimeter = cv2.arcLength(
            c,
            True
        )

        if perimeter == 0:
            continue

        circularity = (
            4
            *
            np.pi
            *
            area
            /
            (perimeter ** 2)
        )

        x, y, w, h = cv2.boundingRect(
            c
        )

        aspect_ratio = (
            min(w, h)
            /
            max(w, h)
        )

        score = (
            area
            *
            circularity
            *
            aspect_ratio
        )

        if score > best_score:
            best_score = score
            best = c

    # ==========================
    # Result
    # ==========================
    if best is None:

        st.error(
            "❌ No blood drop detected"
        )

    elif len(best) < 5:

        st.error(
            "❌ Not enough contour points"
        )

    else:

        ellipse = cv2.fitEllipse(
            best
        )

        (
            center,
            axes,
            angle
        ) = ellipse

        MA = max(
            axes
        )

        ma = min(
            axes
        )

        ratio = (
            ma
            /
            MA
        )

        theta = np.degrees(
            np.arcsin(
                np.clip(
                    ratio,
                    0,
                    1
                )
            )
        )

        cv2.ellipse(
            original,
            ellipse,
            (
                0,
                255,
                0
            ),
            4
        )

        output = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        c1, c2 = st.columns(
            [2, 1]
        )

        with c1:

            st.image(
                output,
                caption="Detected Blood Drop",
                use_container_width=True
            )

        with c2:

            st.metric(
                "Length",
                f"{MA:.1f}"
            )

            st.metric(
                "Width",
                f"{ma:.1f}"
            )

            st.metric(
                "Ratio",
                f"{ratio:.3f}"
            )

            st.metric(
                "Impact Angle",
                f"{theta:.1f}°"
            )
