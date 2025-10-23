# EEG Dashboard

A **web-based EEG analysis dashboard** built using **Streamlit**, allowing visualization, frequency analysis, and automated report generation from EEG CSV files. The tool supports patient data management, EEG signal processing, and generation of professional PDF reports for medical purposes.

---

## Features

- Upload and process EEG CSV files (semicolon-separated).
- Automatic filtering and processing of EEG frequency bands:
  - Delta (0.5–4 Hz)
  - Theta (4–8 Hz)
  - Alpha (8–13 Hz)
  - Beta (13–30 Hz)
  - Gamma (30–100 Hz)
- Compute time-domain and frequency-domain signals.
- Band-pass filtering and computation of dominant channels per band.
- Event visualization (if CSV contains an `event` column).
- Sidebar form for capturing patient details:
  - Name, DOB, Contact, Address, Organization
  - Disease and condition selection
- Generate downloadable **medical PDF report** with:
  - Hospital logo
  - Patient information table
  - EEG analysis summary table
  - Clinical interpretation
  - Doctor's notes and footer
- Generate downloadable **EEG plots PDF** for selected channels.
- Download processed CSV files (raw, frequency, and band-pass filtered data) in a ZIP.

---

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
