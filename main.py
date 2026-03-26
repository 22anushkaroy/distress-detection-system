import streamlit as st
import numpy as np
import os
from datetime import datetime
from src.model import DistressModel
from src.anomaly import check_alert
from src.text_trigger import check_distress_text
from src.voice_trigger import VoiceTrigger

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="Distress Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ============= SIMPLE WHITE + PURPLE THEME =============
st.markdown("""
<style>
    /* White background */
    .stApp {
        background-color: #f8f9fc;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2d1b69;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Title */
    .main-title {
        color: #2d1b69;
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0 0.3rem 0;
    }

    .subtitle {
        color: #7c7c8a;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Section headers */
    .section-header {
        color: #2d1b69;
        font-size: 1.4rem;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #7c3aed;
        margin-bottom: 1rem;
    }

    /* Result boxes */
    .result-box {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #7c3aed;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .result-label {
        color: #7c7c8a;
        font-size: 0.8rem;
        margin-bottom: 0.2rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .result-value {
        color: #1a1a2e;
        font-size: 1.15rem;
        font-weight: 600;
    }

    /* Alert boxes */
    .alert-danger {
        background: #ef4444;
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .alert-safe {
        background: #22c55e;
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    /* Tags */
    .tag-danger {
        display: inline-block;
        background: #fef2f2;
        color: #ef4444;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        border: 1px solid #ef4444;
        font-size: 0.85rem;
    }

    .tag-safe {
        display: inline-block;
        background: #f0fdf4;
        color: #22c55e;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        border: 1px solid #22c55e;
        font-size: 0.85rem;
    }

    .tag-purple {
        display: inline-block;
        background: #f3f0ff;
        color: #7c3aed;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        border: 1px solid #7c3aed;
        font-size: 0.85rem;
    }

    /* Card */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    /* History items */
    .history-row {
        background: white;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 0.3rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Buttons */
    .stButton > button {
        background: #7c3aed;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #6d28d9;
    }

    /* Hide streamlit extras */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============= SESSION STATE =============
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'voice_trigger' not in st.session_state:
    st.session_state.voice_trigger = VoiceTrigger()
if 'history' not in st.session_state:
    st.session_state.history = []


# ============= LOAD MODEL =============
@st.cache_resource
def load_model():
    m = DistressModel()
    m.train('data/train', 'data/my_sensor_data')
    return m


# ============= SIDEBAR =============
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.5rem 0;'>
        <div style='font-size:2.5rem;'>🛡️</div>
        <h2 style='margin:0.5rem 0 0 0;'>Distress Detection</h2>
        <p style='opacity:0.7; font-size:0.85rem;'>Safety Monitoring System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🧠 Train Model", use_container_width=True):
        with st.spinner("Training..."):
            st.session_state.model = load_model()
            st.session_state.trained = True
        st.success("Model trained!")

    if st.session_state.trained:
        st.markdown("**Status:** 🟢 Active")
    else:
        st.markdown("**Status:** 🔴 Not Trained")

    st.markdown("---")
    st.markdown("### Stats")

    total = len(st.session_state.history)
    alerts = len([h for h in st.session_state.history if h['danger']])

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Alerts", alerts)
    c3.metric("Safe", total - alerts)

    st.markdown("---")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ============= MAIN =============
st.markdown("<h1 class='main-title'>🛡️ Distress Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sensor • Text • Voice Analysis</p>", unsafe_allow_html=True)

if not st.session_state.trained:
    st.warning("⚠️ Click **Train Model** in the sidebar to get started.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📱 Sensor", "💬 Text", "🎤 Voice", "📜 History"])


# ============= SENSOR TAB =============
with tab1:
    st.markdown("<div class='section-header'>📱 Sensor Detection</div>", unsafe_allow_html=True)

    sensor_mode = st.selectbox("Input Mode", ["🎲 Simulate Movement", "📂 Load CSV File"])

    if sensor_mode == "🎲 Simulate Movement":
        sim_type = st.selectbox("Movement Pattern",
                                ["Normal Walking", "Fall", "Panic Running", "Struggling", "Freeze"])

        if st.button("🔍 Simulate & Detect", use_container_width=True, key="sim"):
            n = 128

            if sim_type == "Normal Walking":
                t = np.linspace(0, 4 * np.pi, n)
                ax = 0.5 * np.sin(t * 2) + np.random.normal(0, 0.15, n)
                ay = 0.3 * np.sin(t * 2 + 1) + np.random.normal(0, 0.15, n)
                az = 9.8 + 0.4 * np.sin(t * 4) + np.random.normal(0, 0.1, n)

            elif sim_type == "Fall":
                t1 = np.linspace(0, 2 * np.pi, 40)
                ax = np.concatenate([0.3*np.sin(t1)+np.random.normal(0,0.1,40),
                                     np.random.normal(20,5,48), np.random.normal(0,0.1,40)])
                ay = np.concatenate([0.2*np.sin(t1)+np.random.normal(0,0.1,40),
                                     np.random.normal(-15,5,48), np.random.normal(0,0.1,40)])
                az = np.concatenate([9.8+np.random.normal(0,0.1,40),
                                     np.random.normal(-25,8,48), np.random.normal(0,0.1,40)])

            elif sim_type == "Panic Running":
                t = np.linspace(0, 8 * np.pi, n)
                ax = 5*np.sin(t*3) + np.random.normal(3,4,n)
                ay = 4*np.sin(t*2.5) + np.random.normal(2,4,n)
                az = 9.8 + 6*np.sin(t*5) + np.random.normal(0,3,n)

            elif sim_type == "Struggling":
                ax = np.random.uniform(-18, 18, n)
                ay = np.random.uniform(-18, 18, n)
                az = np.random.uniform(-10, 25, n)

            elif sim_type == "Freeze":
                ax = np.random.normal(0, 0.05, n)
                ay = np.random.normal(0, 0.05, n)
                az = np.random.normal(9.8, 0.05, n)

            features = st.session_state.model.extract_features_from_raw(ax, ay, az)
            pred_class, confidence = st.session_state.model.predict(features)
            class_name = st.session_state.model.get_class_name(pred_class)
            is_danger, reason = check_alert(pred_class, confidence)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class='result-box'>
                    <div class='result-label'>Input Pattern</div>
                    <div class='result-value'>{sim_type}</div>
                </div>
                <div class='result-box'>
                    <div class='result-label'>Predicted Activity</div>
                    <div class='result-value'>{class_name}</div>
                </div>
                <div class='result-box'>
                    <div class='result-label'>Confidence</div>
                    <div class='result-value'>{confidence*100:.1f}%</div>
                </div>
                <div class='result-box'>
                    <div class='result-label'>Reason</div>
                    <div class='result-value' style='font-size:0.95rem;'>{reason}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                if is_danger:
                    st.markdown("<div class='alert-danger'>🚨 ALERT — Distress Detected</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<div class='alert-safe'>✅ SAFE — No Distress</div>",
                                unsafe_allow_html=True)

            st.session_state.history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'type': '📱 Sensor',
                'details': f"{sim_type} → {class_name} ({confidence*100:.1f}%)",
                'danger': is_danger
            })

    elif sensor_mode == "📂 Load CSV File":
        folder = 'data/my_sensor_data'
        if os.path.exists(folder):
            csvs = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
            if csvs:
                sel = st.selectbox("Select File", csvs)
                if st.button("🔍 Load & Detect", use_container_width=True, key="csv"):
                    path = os.path.join(folder, sel)
                    ax, ay, az, err = st.session_state.model.read_sensor_csv(path)

                    if err or ax is None:
                        st.error(f"Error: {err}")
                    elif len(ax) < 10:
                        st.error("Too few data points")
                    else:
                        ws = 128
                        if len(ax) >= ws:
                            s = len(ax)//2 - ws//2
                            ax, ay, az = ax[s:s+ws], ay[s:s+ws], az[s:s+ws]
                        else:
                            ax = np.pad(ax, (0, ws-len(ax)), mode='edge')
                            ay = np.pad(ay, (0, ws-len(ay)), mode='edge')
                            az = np.pad(az, (0, ws-len(az)), mode='edge')

                        features = st.session_state.model.extract_features_from_raw(ax, ay, az)
                        pred_class, confidence = st.session_state.model.predict(features)
                        class_name = st.session_state.model.get_class_name(pred_class)
                        is_danger, reason = check_alert(pred_class, confidence)

                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"""
                            <div class='result-box'>
                                <div class='result-label'>File</div>
                                <div class='result-value'>{sel}</div>
                            </div>
                            <div class='result-box'>
                                <div class='result-label'>Predicted Activity</div>
                                <div class='result-value'>{class_name}</div>
                            </div>
                            <div class='result-box'>
                                <div class='result-label'>Confidence</div>
                                <div class='result-value'>{confidence*100:.1f}%</div>
                            </div>
                            <div class='result-box'>
                                <div class='result-label'>Reason</div>
                                <div class='result-value' style='font-size:0.95rem;'>{reason}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with c2:
                            if is_danger:
                                st.markdown("<div class='alert-danger'>🚨 ALERT — Distress Detected</div>",
                                            unsafe_allow_html=True)
                            else:
                                st.markdown("<div class='alert-safe'>✅ SAFE — No Distress</div>",
                                            unsafe_allow_html=True)

                        st.session_state.history.append({
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'type': '📱 Sensor',
                            'details': f"{sel} → {class_name} ({confidence*100:.1f}%)",
                            'danger': is_danger
                        })
            else:
                st.warning("No CSV files found")
        else:
            st.warning("Sensor data folder not found")


# ============= TEXT TAB =============
with tab2:
    st.markdown("<div class='section-header'>💬 Text Detection</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        <strong>How it works:</strong> Type a message and the system checks for distress
        keywords like <span class='tag-purple'>help</span>
        <span class='tag-purple'>danger</span>
        <span class='tag-purple'>save me</span>
        <span class='tag-purple'>emergency</span>
    </div>
    """, unsafe_allow_html=True)

    text_input = st.text_area("Enter message:", height=100, placeholder="e.g., Please help me someone...")

    if st.button("🔍 Check Text", use_container_width=True, key="text"):
        if text_input.strip():
            is_distress, keyword = check_distress_text(text_input)

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class='result-box'>
                    <div class='result-label'>Input Message</div>
                    <div class='result-value'>"{text_input}"</div>
                </div>
                """, unsafe_allow_html=True)

                if is_distress:
                    st.markdown(f"""
                    <div class='result-box'>
                        <div class='result-label'>Keyword Found</div>
                        <div class='result-value'><span class='tag-danger'>"{keyword}"</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-box'>
                        <div class='result-label'>Keyword</div>
                        <div class='result-value'><span class='tag-safe'>None found</span></div>
                    </div>
                    """, unsafe_allow_html=True)

            with c2:
                if is_distress:
                    st.markdown("<div class='alert-danger'>🚨 ALERT — Distress Keyword Found</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<div class='alert-safe'>✅ SAFE — No Distress Keywords</div>",
                                unsafe_allow_html=True)

            st.session_state.history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'type': '💬 Text',
                'details': f"'{keyword}'" if is_distress else "No keywords",
                'danger': is_distress
            })
        else:
            st.warning("Please enter some text")


# ============= VOICE TAB =============
with tab3:
    st.markdown("<div class='section-header'>🎤 Voice Detection</div>", unsafe_allow_html=True)

    voice_folder = 'data/voice'

    if os.path.exists(voice_folder):
        wav_files = sorted([f for f in os.listdir(voice_folder) if f.lower().endswith('.wav')])

        if wav_files:
            # Single file
            st.markdown("#### Select a voice file")
            selected = st.selectbox("File", wav_files, label_visibility="collapsed")

            if st.button("🔍 Analyze Voice", use_container_width=True, key="voice1"):
                path = os.path.join(voice_folder, selected)

                with st.spinner(f"Analyzing {selected}..."):
                    is_distress, keyword, recognized = st.session_state.voice_trigger.check_distress_voice_file(path)

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown(f"""
                    <div class='result-box'>
                        <div class='result-label'>File</div>
                        <div class='result-value'>🎵 {selected}</div>
                    </div>
                    <div class='result-box'>
                        <div class='result-label'>Recognized Speech</div>
                        <div class='result-value'>"{recognized}"</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if is_distress:
                        st.markdown(f"""
                        <div class='result-box'>
                            <div class='result-label'>Trigger Word</div>
                            <div class='result-value'><span class='tag-danger'>"{keyword}"</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-box'>
                            <div class='result-label'>Trigger Word</div>
                            <div class='result-value'><span class='tag-safe'>None found</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                with c2:
                    if is_distress:
                        st.markdown("<div class='alert-danger'>🚨 ALERT — Distress Speech Detected</div>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='alert-safe'>✅ SAFE — No Distress Speech</div>",
                                    unsafe_allow_html=True)

                st.session_state.history.append({
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'type': '🎤 Voice',
                    'details': f"{selected} → '{keyword}'" if is_distress else f"{selected} → Safe",
                    'danger': is_distress
                })

            # Batch analysis
            st.markdown("---")
            st.markdown("#### Analyze all files")

            if st.button("🔍 Analyze All Voice Files", use_container_width=True, key="voice_all"):
                progress = st.progress(0)
                results = []

                for i, wf in enumerate(wav_files):
                    fp = os.path.join(voice_folder, wf)
                    dist, kw, txt = st.session_state.voice_trigger.check_distress_voice_file(fp)
                    results.append({'file': wf, 'distress': dist, 'keyword': kw, 'text': txt})
                    progress.progress((i + 1) / len(wav_files))

                progress.empty()

                alert_count = sum(1 for r in results if r['distress'])
                st.markdown(f"""
                <div class='info-card'>
                    <strong>Results:</strong> {len(results)} files analyzed —
                    <span class='tag-danger'>🚨 {alert_count} Alerts</span>
                    <span class='tag-safe'>✅ {len(results) - alert_count} Safe</span>
                </div>
                """, unsafe_allow_html=True)

                for r in results:
                    if r['distress']:
                        st.markdown(f"""
                        <div class='history-row' style='border-left:4px solid #ef4444;'>
                            <span>🚨 {r['file']}</span>
                            <span class='tag-danger'>"{r['keyword']}"</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='history-row' style='border-left:4px solid #22c55e;'>
                            <span>✅ {r['file']}</span>
                            <span class='tag-safe'>Safe</span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.session_state.history.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'type': '🎤 Voice',
                        'details': f"{r['file']} → '{r['keyword']}'" if r['distress'] else f"{r['file']} → Safe",
                        'danger': r['distress']
                    })
        else:
            st.warning("No .wav files in data/voice/")
    else:
        st.error("Voice folder not found: data/voice/")


# ============= HISTORY TAB =============
with tab4:
    st.markdown("<div class='section-header'>📜 Detection History</div>", unsafe_allow_html=True)

    if st.session_state.history:
        for entry in reversed(st.session_state.history):
            if entry['danger']:
                st.markdown(f"""
                <div class='history-row' style='border-left:4px solid #ef4444;'>
                    <div>
                        <strong style='color:#ef4444;'>🚨 {entry['type']}</strong>
                        <span style='color:#7c7c8a; margin-left:0.5rem;'>{entry['details']}</span>
                    </div>
                    <span style='color:#7c7c8a; font-size:0.8rem;'>{entry['time']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='history-row' style='border-left:4px solid #22c55e;'>
                    <div>
                        <strong style='color:#22c55e;'>✅ {entry['type']}</strong>
                        <span style='color:#7c7c8a; margin-left:0.5rem;'>{entry['details']}</span>
                    </div>
                    <span style='color:#7c7c8a; font-size:0.8rem;'>{entry['time']}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-card' style='text-align:center;'>
            <p style='color:#7c7c8a;'>No detections yet. Run some tests to see history here.</p>
        </div>
        """, unsafe_allow_html=True)