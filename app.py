import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import mediapipe as mp
import os
import gdown

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Ergonomic Analysis",
    page_icon="🪑",
    layout="centered"
)

st.title("🪑 Ergonomic Posture Analysis")
st.caption("ใช้ YOLO + MediaPipe ประเมินมุมคอ หลัง เข่า จากภาพด้านข้าง")

# โหลดโมเดล YOLO (cache) พร้อมดาวน์โหลดอัตโนมัติ
@st.cache_resource
def load_yolo_model():
    model_path = "best.pt"
    
    # ถ้าไม่มีไฟล์โมเดล ให้ดาวน์โหลดจาก Google Drive
    if not os.path.exists(model_path):
        st.info("🔄 กำลังดาวน์โหลดโมเดล... (ครั้งแรกอาจใช้เวลา 1-2 นาที)")
        try:
            # แทนที่ YOUR_FILE_ID ด้วย Google Drive file ID ของคุณ
            # วิธีหา: แชร์ไฟล์ในไดร์ฟเป็น "Anyone with the link"
            # URL จะเป็น: https://drive.google.com/file/d/FILE_ID_HERE/view
            file_id = "https://drive.google.com/file/d/120x8rUd7nbJAXc0huZbQqJchkTny1pGj/view?usp=sharing"
            url = f"120x8rUd7nbJAXc0huZbQqJchkTny1pGj"
            
            # หรือใช้ลิงก์โดยตรงจาก Dropbox, OneDrive, etc.
            # url = "YOUR_DIRECT_DOWNLOAD_LINK"
            
            gdown.download(url, model_path, quiet=False)
            st.success("✅ ดาวน์โหลดโมเดลสำเร็จ")
        except Exception as e:
            st.error(f"❌ ไม่สามารถดาวน์โหลดโมเดลได้: {str(e)}")
            st.info("💡 กรุณาตรวจสอบว่า:\n- ลิงก์ Google Drive ถูกต้อง\n- ไฟล์ถูกแชร์เป็น 'Anyone with the link'\n- หรืออัปโหลดไฟล์ best.pt ลงโฟลเดอร์เดียวกับ app.py")
            st.stop()
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {str(e)}")
        st.stop()

yolo_model = load_yolo_model()

# MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ฟังก์ชันช่วยด้านยศาสตร์
def calculate_angle(a, b, c):
    """คำนวณมุมระหว่าง 3 จุด"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cos_ang = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_ang))
    return angle

def flex_from_straight(angle):
    """คำนวณการเบี่ยงเบนจากแนวตรง (180°)"""
    if angle is None:
        return None
    return abs(180.0 - angle)

def choose_side_landmarks(landmarks):
    """เลือกด้านที่มองเห็นชัดเจนกว่า (ซ้าย/ขวา)"""
    lm = mp_pose.PoseLandmark

    def get_xyz(id_):
        p = landmarks[id_]
        return p.x, p.y, p.visibility

    left_ids = [lm.LEFT_EAR, lm.LEFT_SHOULDER, lm.LEFT_HIP, lm.LEFT_KNEE, lm.LEFT_ANKLE]
    right_ids = [lm.RIGHT_EAR, lm.RIGHT_SHOULDER, lm.RIGHT_HIP, lm.RIGHT_KNEE, lm.RIGHT_ANKLE]

    left_points = [get_xyz(int(i.value)) for i in left_ids]
    right_points = [get_xyz(int(i.value)) for i in right_ids]

    left_vis = np.mean([p[2] for p in left_points])
    right_vis = np.mean([p[2] for p in right_points])

    if left_vis >= right_vis:
        side = "left"
        ear = left_points[0][:2]
        shoulder = left_points[1][:2]
        hip = left_points[2][:2]
        knee = left_points[3][:2]
        ankle = left_points[4][:2]
    else:
        side = "right"
        ear = right_points[0][:2]
        shoulder = right_points[1][:2]
        hip = right_points[2][:2]
        knee = right_points[3][:2]
        ankle = right_points[4][:2]

    return side, {
        "ear": ear,
        "shoulder": shoulder,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
    }

def classify_ergonomic(neck_flex, trunk_flex, knee_angle):
    """ประเมินท่านั่งตามหลักการยศาสตร์"""
    if neck_flex is None or trunk_flex is None or knee_angle is None:
        return "ไม่แน่ใจ", "unknown", ["ตรวจจับจุดสำคัญไม่ครบ (neck/trunk/knee เป็น None)"]

    reason = []

    # คอ
    if neck_flex <= 20:
        reason.append(f"คออยู่ในช่วงดี (เบี่ยงจากแนวตรง ~ {neck_flex:.1f}°)")
        neck_score = 2
    elif neck_flex <= 45:
        reason.append(f"คอเริ่มก้ม/เงยมาก (~ {neck_flex:.1f}°)")
        neck_score = 1
    else:
        reason.append(f"คอก้ม/เงยมากเกินไป (~ {neck_flex:.1f}°)")
        neck_score = 0

    # หลัง
    if trunk_flex <= 20:
        reason.append(f"หลังอยู่ในช่วงดี (เบี่ยงจากแนวตรง ~ {trunk_flex:.1f}°)")
        trunk_score = 2
    elif trunk_flex <= 45:
        reason.append(f"หลังเริ่มเอน/งอมาก (~ {trunk_flex:.1f}°)")
        trunk_score = 1
    else:
        reason.append(f"หลังงอมากเกินไป (~ {trunk_flex:.1f}°)")
        trunk_score = 0

    # เข่า
    if 80 <= knee_angle <= 120:
        reason.append(f"มุมเข่าอยู่ในช่วงเหมาะสม (~ {knee_angle:.1f}°)")
        knee_score = 2
    else:
        reason.append(f"มุมเข่าอาจไม่เหมาะสม (~ {knee_angle:.1f}°)")
        knee_score = 1

    total = neck_score + trunk_score + knee_score

    if total >= 5:
        status = "ท่านั่งดีตามหลักการยศาสตร์"
        level = "good"
    elif total >= 3:
        status = "ท่านั่งพอใช้ แต่ควรปรับบางจุด"
        level = "caution"
    else:
        status = "ท่านั่งเสี่ยงต่อการปวดเมื่อย/บาดเจ็บ"
        level = "poor"

    return status, level, reason

def analyze_posture_mediapipe_full(img_bgr):
    """วิเคราะห์ท่าทางด้วย MediaPipe (ใช้ทั้งภาพ)"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return out, None, None, None, None, "unknown", ["ไม่พบโครงร่างบุคคลในภาพ"]

    landmarks = results.pose_landmarks.landmark
    side, pts = choose_side_landmarks(landmarks)
    ear = pts["ear"]
    shoulder = pts["shoulder"]
    hip = pts["hip"]
    knee = pts["knee"]
    ankle = pts["ankle"]

    neck_angle = calculate_angle(ear, shoulder, hip)
    trunk_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)

    neck_flex = flex_from_straight(neck_angle)
    trunk_flex = flex_from_straight(trunk_angle)

    status, level, reason = classify_ergonomic(neck_flex, trunk_flex, knee_angle)

    annotated = img_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
    )

    out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return out_rgb, side, neck_flex, trunk_flex, knee_angle, level, reason

def analyze_posture_yolo_ergonomic(img_bgr, yolo_conf=0.3):
    """วิเคราะห์ท่าทางด้วย YOLO + MediaPipe"""
    h, w, _ = img_bgr.shape
    results = yolo_model(img_bgr, conf=yolo_conf, verbose=False)

    # ถ้า YOLO ไม่เจอเลย → ใช้ MediaPipe ทั้งภาพ
    if len(results) == 0 or len(results[0].boxes) == 0:
        return analyze_posture_mediapipe_full(img_bgr)

    r = results[0]
    boxes = r.boxes

    # หา bounding box ที่ใหญ่ที่สุด
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        areas.append((x2 - x1) * (y2 - y1))
    idx = int(np.argmax(areas))
    box = boxes[idx]

    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    cls_name = yolo_model.names.get(cls_id, str(cls_id))

    roi_bgr = img_bgr[y1:y2, x1:x2].copy()
    if roi_bgr.size == 0:
        return analyze_posture_mediapipe_full(img_bgr)

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(roi_rgb)

    if not pose_results.pose_landmarks:
        return analyze_posture_mediapipe_full(img_bgr)

    landmarks = pose_results.pose_landmarks.landmark
    side, pts = choose_side_landmarks(landmarks)
    ear = pts["ear"]
    shoulder = pts["shoulder"]
    hip = pts["hip"]
    knee = pts["knee"]
    ankle = pts["ankle"]

    neck_angle = calculate_angle(ear, shoulder, hip)
    trunk_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)

    neck_flex = flex_from_straight(neck_angle)
    trunk_flex = flex_from_straight(trunk_angle)

    status, level, reason = classify_ergonomic(neck_flex, trunk_flex, knee_angle)

    annotated_roi = roi_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated_roi,
        pose_results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
    )

    annotated_full = img_bgr.copy()
    annotated_full[y1:y2, x1:x2] = annotated_roi

    color_box = (0, 255, 0) if level == "good" else ((0, 255, 255) if level == "caution" else (0, 0, 255))
    cv2.rectangle(annotated_full, (x1, y1), (x2, y2), color_box, 2)
    cv2.putText(
        annotated_full,
        f"{cls_name} {conf:.2f}",
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_box,
        2
    )

    out_rgb = cv2.cvtColor(annotated_full, cv2.COLOR_BGR2RGB)
    return out_rgb, side, neck_flex, trunk_flex, knee_angle, level, reason


# ==================== UI หลัก ====================
st.divider()

# เพิ่มคำแนะนำ
with st.expander("💡 วิธีใช้งาน"):
    st.markdown("""
    1. **เลือกแหล่งภาพ**: อัปโหลดรูป หรือถ่ายด้วยกล้อง
    2. **ปรับ Confidence**: ค่าความมั่นใจในการตรวจจับ (0.1-0.9)
    3. **วิเคราะห์**: คลิกปุ่มวิเคราะห์
    4. **ดูผลลัพธ์**: ระบบจะแสดงมุมต่างๆ และให้คำแนะนำ
    
    ⚠️ **หมายเหตุ**: ควรถ่ายภาพจากด้านข้างเพื่อความแม่นยำสูงสุด
    """)

mode = st.radio(
    "เลือกแหล่งภาพ",
    ["อัปโหลดรูป", "ถ่ายจากกล้อง"],
    horizontal=True,
    help="เลือกวิธีการนำเข้ารูปภาพ"
)

yolo_conf = st.slider(
    "Confidence Threshold",
    0.1, 0.9, 0.3, 0.05,
    help="ใช้กำหนดความมั่นใจขั้นต่ำของ YOLO ก่อนนับว่าเจอคน"
)

# โหมดภาพนิ่ง (upload/snapshot)
img_bgr = None

if mode == "อัปโหลดรูป":
    file = st.file_uploader(
        "อัปโหลดรูปท่านั่ง (ด้านข้างจะแม่นกว่า)", 
        type=["jpg", "jpeg", "png"],
        help="รองรับไฟล์ JPG, JPEG, PNG"
    )
    if file is not None:
        try:
            pil_img = Image.open(file).convert("RGB")
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            st.image(img_rgb, caption="ภาพต้นฉบับ", use_container_width=True)
        except Exception as e:
            st.error(f"❌ ไม่สามารถโหลดรูปภาพได้: {str(e)}")

elif mode == "ถ่ายจากกล้อง":
    picture = st.camera_input("ถ่ายภาพจากกล้อง")
    if picture is not None:
        try:
            pil_img = Image.open(picture).convert("RGB")
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            st.image(img_rgb, caption="ภาพที่ถ่าย", use_container_width=True)
        except Exception as e:
            st.error(f"❌ ไม่สามารถประมวลผลภาพได้: {str(e)}")

analyze_btn = st.button("🔍 วิเคราะห์ท่านั่ง", type="primary", use_container_width=True)

if analyze_btn:
    if img_bgr is None:
        st.warning("⚠️ กรุณาเลือกหรืออัปโหลดภาพก่อน")
    else:
        with st.spinner("🔄 กำลังวิเคราะห์ท่านั่ง..."):
            try:
                result_img, side, neck_flex, trunk_flex, knee_angle, level, reason = \
                    analyze_posture_yolo_ergonomic(img_bgr, yolo_conf=yolo_conf)

                st.divider()
                st.subheader("📊 ผลการวิเคราะห์")

                # Layout: ซ้ายภาพ / ขวาข้อมูล
                col_img, col_info = st.columns([2, 1])

                with col_img:
                    st.image(result_img, caption="ภาพผลลัพธ์", use_container_width=True)

                with col_info:
                    st.markdown("### 📐 ค่าที่วัดได้")
                    st.write(f"**ด้านที่ใช้:** `{side if side else 'N/A'}`")
                    
                    # แสดงมุมด้วยสีตามค่า
                    if neck_flex is not None:
                        neck_color = "🟢" if neck_flex <= 20 else ("🟡" if neck_flex <= 45 else "🔴")
                        st.write(f"{neck_color} **Neck:** {neck_flex:.1f}°")
                    else:
                        st.write("⚪ **Neck:** N/A")
                    
                    if trunk_flex is not None:
                        trunk_color = "🟢" if trunk_flex <= 20 else ("🟡" if trunk_flex <= 45 else "🔴")
                        st.write(f"{trunk_color} **Trunk:** {trunk_flex:.1f}°")
                    else:
                        st.write("⚪ **Trunk:** N/A")
                    
                    if knee_angle is not None:
                        knee_color = "🟢" if 80 <= knee_angle <= 120 else "🟡"
                        st.write(f"{knee_color} **Knee:** {knee_angle:.1f}°")
                    else:
                        st.write("⚪ **Knee:** N/A")

                    st.markdown("---")
                    st.markdown("### 🎯 สรุปผล")
                    
                    if level == "good":
                        st.success("✅ ท่านั่งดีตามหลักการยศาสตร์")
                    elif level == "caution":
                        st.warning("⚠️ ท่านั่งพอใช้ แต่ควรปรับบางจุด")
                    elif level == "poor":
                        st.error("❌ ท่านั่งเสี่ยงต่อการปวดเมื่อย/บาดเจ็บ")
                    else:
                        st.info("ℹ️ ไม่สามารถประเมินได้แน่ชัดจากภาพนี้")

                    if reason:
                        with st.expander("📝 รายละเอียดเพิ่มเติม"):
                            for r in reason:
                                st.write("• " + r)
                
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")
                st.info("💡 ลองปรับค่า Confidence หรือใช้ภาพอื่น")

# Footer
st.divider()
st.caption("🔬 Powered by YOLO + MediaPipe | 💡 ควรถ่ายภาพด้านข้างเพื่อความแม่นยำสูงสุด")