# streamlit_eeg_dashboard.py
import sys
import io
import zipfile
from datetime import datetime, date

import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import plotly.express as px
import os

# PDF generation and plotting
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
DISEASES = ["Diabetes", "Hypertension", "Epilepsy", "Anxiety", "Depression", "None"]

CONDITIONS = [
    "Deep sleep / unconscious",
    "Relaxed / drowsy",
    "Calm / relaxed but awake",
    "Alert / focused / anxious",
    "Cognitive processing / memory"
]

SAMPLING_RATE = 250  # Hz (change as needed)

BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100)
}

EMOTIONS = {
    "Delta": "Deep sleep / unconscious",
    "Theta": "Relaxed / drowsy",
    "Alpha": "Calm / relaxed but awake",
    "Beta": "Alert / focused / anxious",
    "Gamma": "Cognitive processing / memory",
}

IGNORE_COLS = {"X1", "X2", "AD0", "AD1"}  # 'event' is kept for markers

# Logo and watermark paths (you already uploaded logo to /mnt/data)
LOGO_PATH = "/mnt/data/9c40c0c6-2d0b-45c7-a194-b73b43ddad3f.png"
BRAIN_WATERMARK_PATH = "/mnt/data/brain_diagram.png"  # replace with your preferred brain image (optional)

# ----------------------------
# Helper functions
# ----------------------------
def save_patient_details(details: dict):
    if "patient_details" not in st.session_state:
        st.session_state["patient_details"] = []
    # Case ID sequencing: maintain counter in session_state
    if "case_counter" not in st.session_state:
        st.session_state["case_counter"] = 1
    else:
        st.session_state["case_counter"] += 1
    case_id = f"EEG-{datetime.now().strftime('%Y%m%d')}-{st.session_state['case_counter']:03d}"
    details["CaseID"] = case_id
    st.session_state["patient_details"].append(details)

def calculate_age(dob: date):
    """Return age in years or None if out of range."""
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    if 0 <= age <= 120:
        return age
    return None

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    if len(data) < (order * 3):
        return np.zeros_like(data)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def compute_fft(signal, fs):
    N = len(signal)
    if N == 0:
        return np.array([]), np.array([])
    fft_vals = np.abs(fft(signal))[:N // 2]
    freqs = fftfreq(N, 1 / fs)[:N // 2]
    return freqs, fft_vals

def process_files(df: pd.DataFrame, events: pd.Series = None, sampling_rate: int = SAMPLING_RATE):
    N = len(df)
    time_axis = np.arange(N) / sampling_rate  # time axis
    
    # Frequency domain
    freq_axis = fftfreq(N, 1 / sampling_rate)[:N // 2]
    freq_df = pd.DataFrame({"freq": freq_axis})
    for col in df.columns:
        freqs, fft_vals = compute_fft(df[col].values, sampling_rate)
        freq_df[col] = fft_vals

    # Band-passed time domain signals
    band_time_dfs = {}
    for band_name, (low, high) in BANDS.items():
        band_df = pd.DataFrame(index=df.index, columns=df.columns)
        for col in df.columns:
            try:
                band_df[col] = bandpass_filter(df[col].values, low, high, sampling_rate, order=4)
            except Exception:
                band_df[col] = np.zeros_like(df[col].values)
        band_time_dfs[band_name] = band_df

    return {
        "raw": df.assign(time=time_axis),
        "frequency_domain": freq_df,
        "events": events.reset_index(drop=True) if events is not None else None,
        **band_time_dfs
    }

def generate_summary_and_csvs(generated_dict, filename_prefix="output"):
    summary_rows = []
    time_axis = generated_dict["raw"]["time"].values
    channels = [c for c in generated_dict["raw"].columns if c != "time"]
    csv_files = {}

    for band, (low, high) in BANDS.items():
        band_df = generated_dict.get(band)
        if band_df is None:
            continue
        band_df_with_time = band_df.copy()
        band_df_with_time.insert(0, "time", time_axis)
        csv_files[f"{band.lower()}"] = band_df_with_time

        band_powers = {}
        for ch in channels:
            vals = band_df[ch].values
            band_powers[ch] = np.nanmean(np.nan_to_num(vals) ** 2)
        # If all NaN or zeros, handle gracefully
        try:
            dominant_channel = max(band_powers, key=band_powers.get)
        except ValueError:
            dominant_channel = channels[0] if channels else "N/A"

        summary_rows.append({
            "Signal": band,
            "Standard Values": f"{low}-{high} Hz",
            "Channel": dominant_channel,
            "Test": EMOTIONS.get(band, "Unknown")
        })

    # Add frequency and raw CSV
    csv_files["frequency"] = generated_dict["frequency_domain"]
    csv_files["raw"] = generated_dict["raw"]

    return pd.DataFrame(summary_rows), csv_files

def get_dominant_eeg_state(summary_df):
    band_counts = summary_df.groupby("Test")["Signal"].count()
    if not band_counts.empty:
        return band_counts.idxmax()
    return "Unknown"

# ----------------------------
# PDF Generation (UPDATED)
# ----------------------------
def generate_medical_pdf(patient_details: dict, summary_df: pd.DataFrame):
    """
    Generates a professional hospital-style PDF report with:
      - Hospital logo in header (if LOGO_PATH exists)
      - Brain watermark background (if BRAIN_WATERMARK_PATH exists)
      - Patient Information table (includes CaseID)
      - EEG Analysis Summary table
      - Formal clinical interpretation paragraph
      - Doctor's notes and footer
    """
    buf = io.BytesIO()
    # create a doc with slightly larger margins for header/footer
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=72, bottomMargin=36)
    styles = getSampleStyleSheet()

    # Additional paragraph styles
    styles.add(ParagraphStyle(name="ReportTitle", fontSize=16, alignment=1, spaceAfter=6, leading=18))
    styles.add(ParagraphStyle(name="Institution", fontSize=10, alignment=1, textColor=colors.HexColor("#155F5F"), spaceAfter=12))
    styles.add(ParagraphStyle(name="SectionHeader", fontSize=12, textColor=colors.HexColor("#0B4F6C"), spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallCenter", fontSize=8, alignment=1, textColor=colors.grey))

    elements = []

    # --- Header: logo on left + title centered ---
    header_rows = []
    if os.path.exists(LOGO_PATH):
        try:
            logo = RLImage(LOGO_PATH, width=70, height=70)
        except Exception:
            logo = None
    else:
        logo = None

    title_para = Paragraph("<b>EEG Analysis Report</b>", styles["ReportTitle"])
    inst_para = Paragraph("S.D.M. College of Engineering & Technology, Dharwad", styles["Institution"])

    if logo is not None:
        header_table = Table([[logo, title_para], ["", inst_para]], colWidths=[80, 420])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(header_table)
    else:
        elements.append(title_para)
        elements.append(inst_para)

    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # --- Patient Information Table (include CaseID if present) ---
    elements.append(Paragraph("Patient Information", styles["SectionHeader"]))
    # Prepare patient data rows in desired order
    pdata = [
        ["Case ID", patient_details.get("CaseID", "")],
        ["Name", patient_details.get("Name", "")],
        ["Age", str(patient_details.get("Age", ""))],
        ["DOB", str(patient_details.get("DOB", ""))],
        ["Contact", patient_details.get("Contact", "")],
        ["Address", patient_details.get("Address", "")],
        ["Organization", patient_details.get("Organization", "")],
        ["Analysis Conducted", patient_details.get("Analysis", "")],
        ["Conducted By", patient_details.get("Conducted By", "")],
        ["Declared Condition", patient_details.get("Condition", "Not declared")],
        ["Report Date", patient_details.get("Date", "")]
    ]
    ptable = Table(pdata, colWidths=[140, 360])
    ptable.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(ptable)
    elements.append(Spacer(1, 12))

    # --- EEG Analysis Summary Table ---
    elements.append(Paragraph("EEG Analysis Summary", styles["SectionHeader"]))
    if summary_df is None or summary_df.empty:
        elements.append(Paragraph("No EEG analysis data available.", styles["Normal"]))
    else:
        summary_table_data = [list(summary_df.columns)] + summary_df.fillna("").values.tolist()
        col_widths = [90] + [120] * (len(summary_df.columns) - 1)
        summary_table = Table(summary_table_data, colWidths=col_widths[:len(summary_df.columns)])
        summary_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#BEE6F6")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ]))
        elements.append(summary_table)

    elements.append(Spacer(1, 12))

    # --- Formal Interpretation / Clinical Notes ---
    analyzed_state = get_dominant_eeg_state(summary_df) if summary_df is not None else "Unknown"
    declared_state = patient_details.get("Condition", "Not declared")

    interpretation_para = (
        f"The EEG signals were analyzed across standard frequency bands (Delta, Theta, Alpha, Beta, Gamma). "
        f"Dominant channels for each band are listed above. The algorithmic analysis identifies the EEG-derived dominant state as "
        f"\"{analyzed_state}\". This pattern is consistent with the clinical description: {EMOTIONS.get(analyzed_state, 'N/A')}. "
        "These results should be interpreted in the context of clinical history and examination. Any abnormal findings "
        "or discrepancies between reported symptoms and EEG patterns should be reviewed by a neurologist."
    )
    elements.append(Paragraph("Clinical Interpretation", styles["SectionHeader"]))
    elements.append(Paragraph(interpretation_para, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # --- Dominant Channel Findings (detailed lines) ---
    if summary_df is not None and not summary_df.empty:
        elements.append(Paragraph("Dominant Channel Findings", styles["SectionHeader"]))
        lines = []
        for _, row in summary_df.iterrows():
            lines.append(f"{row['Signal']} band → Dominant channel: {row['Channel']} ({row['Test']})")
        for ln in lines:
            elements.append(Paragraph(ln, styles["Normal"]))
        elements.append(Spacer(1, 12))

    # --- Doctor's notes and signature block ---
    elements.append(Paragraph("Doctor's Notes", styles["SectionHeader"]))
    elements.append(Paragraph("__________________________________________________________________", styles["Normal"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Verified by: Dr. __________________________", styles["Normal"]))
    elements.append(Spacer(1, 18))

    # --- Footer disclaimer ---
    footer_style = ParagraphStyle(name="Footer", fontSize=8, alignment=1, textColor=colors.grey)
    elements.append(Paragraph("Report generated by EEG Dashboard | Not a substitute for clinical diagnosis", footer_style))

    # --- Watermark drawing function ---
    def add_watermark(canvas, doc):
        # Draw brain watermark if exists
        try:
            if os.path.exists(BRAIN_WATERMARK_PATH):
                brain_img = ImageReader(BRAIN_WATERMARK_PATH)
                page_w, page_h = A4
                # center the watermark and make it large but faded (ReportLab doesn't directly support opacity).
                # To get faded effect, user can provide a pre-faded PNG (recommended).
                img_w = page_w * 0.6
                img_h = page_h * 0.6
                x = (page_w - img_w) / 2
                y = (page_h - img_h) / 2
                canvas.saveState()
                canvas.drawImage(brain_img, x, y, width=img_w, height=img_h, mask='auto')
                canvas.restoreState()
        except Exception:
            # silently ignore watermark errors
            pass

        # Footer page number / small text
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0] / 2.0, 18, "EEG Analysis Report - Page %d" % (doc.page))

    # Build PDF with watermark on each page
    doc.build(elements, onFirstPage=add_watermark, onLaterPages=add_watermark)

    buf.seek(0)
    return buf

def generate_plots_pdf(generated: dict, selected_channels: list):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    elements = []

    time_axis = generated["raw"]["time"].values
    freq_df = generated["frequency_domain"]

    for ch in selected_channels:
        elements.append(Paragraph(f"Channel: {ch}", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        # Raw plot
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(time_axis, generated["raw"][ch].values)
        ax.set_title(f"{ch} — Raw")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        img_buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(img_buf, format="png", dpi=150)
        plt.close(fig)
        img_buf.seek(0)
        elements.append(RLImage(img_buf, width=700, height=180))
        elements.append(Spacer(1, 8))

        # Frequency plot
        if ch in freq_df.columns:
            fig, ax = plt.subplots(figsize=(8, 2.5))
            ax.plot(freq_df["freq"].values, freq_df[ch].values)
            ax.set_title(f"{ch} — Frequency Spectrum")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.grid(True)
            img_buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(img_buf, format="png", dpi=150)
            plt.close(fig)
            img_buf.seek(0)
            elements.append(RLImage(img_buf, width=700, height=180))
            elements.append(Spacer(1, 8))

        # Band plots
        for band_name in BANDS.keys():
            if band_name in generated:
                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.plot(time_axis, generated[band_name][ch].values)
                ax.set_title(f"{ch} — {band_name} Band")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.grid(True)
                img_buf = io.BytesIO()
                plt.tight_layout()
                fig.savefig(img_buf, format="png", dpi=150)
                plt.close(fig)
                img_buf.seek(0)
                elements.append(RLImage(img_buf, width=700, height=180))
                elements.append(Spacer(1, 6))
        elements.append(PageBreak())

    doc.build(elements)
    buf.seek(0)
    return buf

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="EEG Dashboard", layout="wide")
st.title("EEG Dashboard")

# Sidebar patient form
st.sidebar.header("Patient Details")
with st.sidebar.form("patient_form", clear_on_submit=False):
    name = st.text_input("Name")
    dob = st.date_input("Date of Birth", min_value=date(1900, 1, 1), max_value=date.today())
    contact = st.text_input("Contact Number")
    date_field = st.text_input("Date (DD/MM/YYYY)", value=datetime.now().strftime("%d/%m/%Y"))
    address = st.text_input("Address")
    analysis = st.text_input("Analysis Conducted")
    organization = st.text_input("Organization")
    conducted_by = st.text_input("Conducted By")
    disease_choice = st.selectbox("Select Predefined Disease", DISEASES, index=len(DISEASES)-1)
    condition_choice = st.selectbox("Select Condition", CONDITIONS, index=0)
    save_details = st.form_submit_button("Save Patient Details")

if save_details:
    valid_age = calculate_age(dob)
    if valid_age is None:
        st.sidebar.error("❌ Invalid DOB or age out of range (0–120).")
    else:
        details = {
            "Name": name,
            "Age": valid_age,
            "DOB": dob.strftime("%d/%m/%Y"),
            "Contact": contact,
            "Date": date_field,
            "Address": address,
            "Analysis": analysis,
            "Organization": organization,
            "Conducted By": conducted_by,
            "Disease": disease_choice,
            "Condition": condition_choice
        }
        save_patient_details(details)
        st.sidebar.success("✅ Patient details saved!")

# Main processing
st.header("Run EEG Processing")
uploaded_file = st.file_uploader("Upload EEG CSV (semicolon separated)", type=["csv"])
run_button = st.button("Run")

# session keys
if "generated" not in st.session_state:
    st.session_state["generated"] = None
if "csv_files" not in st.session_state:
    st.session_state["csv_files"] = None
if "summary_df" not in st.session_state:
    st.session_state["summary_df"] = None

if run_button:
    if uploaded_file is None:
        st.error("Please upload a CSV first.")
    else:
        try:
            df = pd.read_csv(uploaded_file, sep=";")
            events = df["event"] if "event" in df.columns else None
            df = df.drop(columns=[c for c in df.columns if c in IGNORE_COLS.union({"event"})], errors="ignore")
            df = df.select_dtypes(include=[np.number])
            if df.empty:
                st.error("CSV must contain numeric EEG channel data.")
            else:
                generated = process_files(df, events=events, sampling_rate=SAMPLING_RATE)
                st.session_state["generated"] = generated
                
                filename_prefix = os.path.splitext(uploaded_file.name)[0]
                summary_df, csv_files = generate_summary_and_csvs(generated, filename_prefix)
                st.session_state["summary_df"] = summary_df
                st.session_state["csv_files"] = csv_files
                
                st.success(f"✅ EEG processed (shape {df.shape})")
                st.subheader("Summary Table")
                st.dataframe(summary_df)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Post-processing and downloads
if st.session_state.get("generated") is not None:
    st.subheader("Post-processing")
    mode = st.radio("Choose mode", ("Enable", "Disable"))
    generated = st.session_state["generated"]
    csv_files = st.session_state["csv_files"]
    summary_df = st.session_state["summary_df"]

    if mode == "Enable":
        channels = [c for c in generated["raw"].columns if c != "time"]
        selected_channels = st.multiselect("Select channels", channels, default=channels[:1])
        data_options = ["Raw", "Frequency"] + list(BANDS.keys())
        selected_data_types = st.multiselect("Select data types", data_options, default=["Raw"])

        if selected_channels:
            time_axis = generated["raw"]["time"].values
            freq_df = generated["frequency_domain"]
            events = generated.get("events")

            for ch in selected_channels:
                st.markdown(f"**Channel: {ch}**")
                if "Raw" in selected_data_types:
                    df_raw = pd.DataFrame({"time": time_axis, ch: generated["raw"][ch]})
                    fig = px.line(df_raw, x="time", y=ch, title=f"{ch} — Raw")
                    # Add event lines if available
                    if events is not None:
                        try:
                            for idx, ev in events.dropna().items():
                                t_ev = idx / SAMPLING_RATE
                                # annotation via add_vline
                                fig.add_vline(x=t_ev, line_dash="dash", annotation_text=str(ev), annotation_position="top left")
                        except Exception:
                            pass
                    st.plotly_chart(fig, use_container_width=True)

                if "Frequency" in selected_data_types and ch in freq_df.columns:
                    df_freq = pd.DataFrame({"freq": freq_df["freq"], "magnitude": freq_df[ch]})
                    st.plotly_chart(px.line(df_freq, x="freq", y="magnitude", title=f"{ch} — Spectrum"),
                                    use_container_width=True)

                for band_name in BANDS.keys():
                    if band_name in selected_data_types and band_name in generated:
                        df_band = pd.DataFrame({"time": time_axis, band_name: generated[band_name][ch]})
                        st.plotly_chart(px.line(df_band, x="time", y=band_name, title=f"{ch} — {band_name}"),
                                        use_container_width=True)

    else:  # Disable -> provide download options
        col1, col2 = st.columns(2)
        with col1:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
                for name, df in csv_files.items():
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    z.writestr(f"{name}.csv", csv_bytes)
            zip_buf.seek(0)
            st.download_button(
                label="Download all processed CSV files (ZIP)",
                data=zip_buf,
                file_name="eeg_processed_files.zip",
                mime="application/zip"
            )

        with col2:
            if "patient_details" in st.session_state and st.session_state["patient_details"]:
                patient_details = st.session_state["patient_details"][-1]
                # Medical PDF
                try:
                    pdf_buf = generate_medical_pdf(patient_details, summary_df)
                    st.download_button(
                        label="📄 Download Medical Report (PDF)",
                        data=pdf_buf.getvalue(),
                        file_name="eeg_medical_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Failed to generate medical PDF: {e}")

                # EEG Plots PDF (limit channels to avoid huge file)
                try:
                    channels = [c for c in generated["raw"].columns if c != "time"]
                    max_channels = 12
                    channels_for_pdf = channels[:max_channels]
                    plots_buf = generate_plots_pdf(generated, channels_for_pdf)
                    st.download_button(
                        label="📊 Download EEG Plots (PDF)",
                        data=plots_buf.getvalue(),
                        file_name="eeg_plots_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Failed to generate plots PDF: {e}")
            else:
                st.info("Save patient details in the sidebar to enable PDF report downloads.")

        st.success("✅ Files ready for download.")

# NOTE: We purposely do NOT display the saved patient details in the main chart area.
# They remain in st.session_state and a success message is shown when saved (sidebar).
# If you want to view saved entries in the sidebar for debugging, uncomment below:
# st.sidebar.write(st.session_state.get("patient_details", []))

# Footer / quick notes
st.markdown("---")
st.caption("CSV: semicolon-separated; numeric EEG channels only. Columns 'X1','X2','AD0','AD1' are ignored. "
           "If your CSV contains an 'event' column, events will be shown on the raw plot as dashed vertical lines.")
