import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==================================
# הגדרות עמוד
# ==================================
st.set_page_config(
    page_title="Blood Drop Analyzer",
    layout="wide"
)

st.title("🩸 Blood Drop Impact Angle")
st.write("העלה תמונה והמערכת תזהה את טיפת הדם ותחשב זווית פגיעה.")

# ==================================
# העלאת תמונה
# ==================================
uploaded = st.file_uploader(
    "בחר תמונה",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded)

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

    # =========================
    # HSV
    # =========================
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    lower1 = np.array([0, 50, 50])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([160, 50, 50])
    upper2 = np.array([180, 255, 255])

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

    # =========================
    # ניקוי רעש
    # =========================
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

    # =========================
    # קונטורים
    # =========================
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
            * np.pi
            * area
            / (perimeter ** 2)
        )

        x, y, w, h = cv2.boundingRect(c)

        aspect_ratio = (
            min(w, h)
            /
            max(w, h)
        )

        compactness = (
            circularity
            * aspect_ratio
        )

        score = (
            area
            * compactness
        )

        if score > best_score:
            best_score = score
            best = c

    if best is None:

        st.error("❌ לא נמצאה טיפת דם")

    elif len(best) < 5:

        st.error("❌ אין מספיק נקודות להתאמת אליפסה")

    else:

        ellipse = cv2.fitEllipse(
            best
        )

        (x, y), (MA, ma), angle = ellipse

        length = max(
            MA,
            ma
        )

        width = min(
            MA,
            ma
        )

        ratio = (
            width
            / length
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

        # ציור אליפסה
        cv2.ellipse(
            original,
            ellipse,
            (0, 255, 0),
            4
        )

        display = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        c1, c2 = st.columns(
            [2, 1]
        )

        with c1:
            st.image(
                display,
                caption="Detected Blood Drop",
                use_container_width=True
            )

        with c2:

            st.metric(
                "Length",
                f"{length:.1f}"
            )

            st.metric(
                "Width",
                f"{width:.1f}"
            )

            st.metric(
                "Ratio",
                f"{ratio:.3f}"
            )

            st.metric(
                "Impact Angle",
                f"{theta:.1f}°"
            )
