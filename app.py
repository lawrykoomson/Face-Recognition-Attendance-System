import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from face_utils import register_face, registered_names, recognize_faces, mark_attendance, load_attendance

st.set_page_config(page_title="AI Attendance System", page_icon="🎓", layout="centered")

st.title("🎓 AI Face Recognition Attendance System")
st.caption("End-of-Semester Project — built with Python, OpenCV, and face_recognition")

mode = st.sidebar.radio(
    "Mode",
    ["📸 Register Student", "✅ Take Attendance", "📋 Attendance Log"],
)

with st.sidebar:
    st.divider()
    names = registered_names()
    st.caption(f"**{len(names)} student(s) registered**")
    for n in names:
        st.write(f"- {n}")

if mode == "📸 Register Student":
    st.header("📸 Register a New Student")
    st.write(
        "Take a clear, front-facing photo (webcam or upload) and enter "
        "the student's name to add them to the system."
    )

    name = st.text_input("Student name")

    tab_camera, tab_upload = st.tabs(["📷 Use webcam", "📁 Upload photo"])

    photo = None
    with tab_camera:
        cam_photo = st.camera_input("Take a photo", key="cam_register")
        if cam_photo:
            photo = cam_photo

    with tab_upload:
        upload_photo = st.file_uploader(
            "Upload a photo", type=["jpg", "jpeg", "png"], key="upload_register"
        )
        if upload_photo:
            photo = upload_photo

    if st.button("Register", type="primary", disabled=not (name and photo)):
        image = np.array(Image.open(photo).convert("RGB"))
        with st.spinner("Detecting face..."):
            success, message = register_face(image, name)
        if success:
            st.success(message)
            st.balloons()
        else:
            st.error(message)

elif mode == "✅ Take Attendance":
    st.header("✅ Take Attendance")
    st.write("Take or upload a photo — everyone recognized gets marked present.")

    tab_camera, tab_upload = st.tabs(["📷 Use webcam", "📁 Upload photo"])

    photo = None
    with tab_camera:
        cam_photo = st.camera_input("Take a photo", key="cam_attendance")
        if cam_photo:
            photo = cam_photo

    with tab_upload:
        upload_photo = st.file_uploader(
            "Upload a photo", type=["jpg", "jpeg", "png"], key="upload_attendance"
        )
        if upload_photo:
            photo = upload_photo

    if photo:
        image = np.array(Image.open(photo).convert("RGB"))

        with st.spinner("Recognizing faces..."):
            results = recognize_faces(image)

        if not results:
            st.warning("No faces detected in this photo.")
        else:
            # Draw bounding boxes + names on a copy of the image
            annotated = Image.fromarray(image)
            draw = ImageDraw.Draw(annotated)

            for r in results:
                top, right, bottom, left = r["location"]
                color = "lime" if r["name"] != "Unknown" else "red"
                draw.rectangle([left, top, right, bottom], outline=color, width=3)
                label = r["name"] if r["name"] == "Unknown" else f"{r['name']} ({r['confidence']}%)"
                draw.rectangle([left, bottom, right, bottom + 22], fill=color)
                draw.text((left + 4, bottom + 4), label, fill="black")

            st.image(annotated, caption=f"{len(results)} face(s) detected", use_container_width=True)

            st.subheader("📝 Attendance results")
            for r in results:
                if r["name"] == "Unknown":
                    st.warning("⚠️ An unrecognized face was detected — not marked present.")
                else:
                    newly_marked = mark_attendance(r["name"])
                    if newly_marked:
                        st.success(f"✅ {r['name']} marked present ({r['confidence']}% confidence)")
                    else:
                        st.info(f"ℹ️ {r['name']} was already marked present today.")

elif mode == "📋 Attendance Log":
    st.header("📋 Attendance Log")

    records = load_attendance()

    if not records:
        st.info("No attendance recorded yet.")
    else:
        df = pd.DataFrame(records)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total records", len(df))
        with col2:
            st.metric("Unique students", df["Name"].nunique())

        st.divider()

        dates = sorted(df["Date"].unique(), reverse=True)
        selected_date = st.selectbox("Filter by date", ["All dates"] + list(dates))

        if selected_date != "All dates":
            filtered_df = df[df["Date"] == selected_date]
        else:
            filtered_df = df

        st.dataframe(
            filtered_df.sort_values(["Date", "Time"], ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download as CSV",
            data=csv_data,
            file_name=f"attendance_{selected_date if selected_date != 'All dates' else 'all'}.csv",
            mime="text/csv",
        )