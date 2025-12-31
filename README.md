# LMA Pulse - Intelligent Covenant Monitoring System

**High-Performance AI-Powered Legal & Financial Analytics Platform**

LMA Pulse is a next-generation **Covenant Monitoring System** designed to automate the tracking of financial covenants in loan facility agreements. It leverages **PerforatedAI** and **Dendritic Neural Networks** to predict potential breaches with high accuracy, while integrated **GenAI** agents automatically generate legal documentation (Reservation of Rights, Waivers) upon detection of distress signals.

![LMA Pulse Dashboard](https://via.placeholder.com/800x400.png?text=LMA+Pulse+Dashboard+Preview)

---

## 🚀 Key Features

### 1. **Dendritic AI Engine**
   - **Custom Neural Architecture**: Implements `DendriticSegments` (Strategy 1) to model complex non-linear financial relationships.
   - **Breach Prediction**: Forecasts covenant breaches (Leverage Ratio, DSCR, Interest Cover) across 1-day, 7-day, 30-day, and 90-day horizons.
   - **Self-Optimizing**: The backend automatically trains the model on startup using synthesized financial datasets.

### 2. **Interactive Financial Simulator**
   - **Stress Testing**: Real-time simulation of financial stress scenarios.
   - **Dynamic Controls**: Slider-based UI to adjust Revenue (-%), Interest Rates (+bps), and EBITDA margins.
   - **Instant Feedback**: Immediately visualizes the impact of stress on covenant compliance statuses.

### 3. **Smart Search & Multi-Company Support**
   - **Global Search**: "Spotlight" style command center to instantly switch between borrower profiles.
   - **Diverse Database**: Pre-loaded with 15+ borrower profiles across Energy, Tech, Retail, and Manufacturing sectors.
   - **Context-Aware**: Dashboard adapts instantly to show the selected company's live financial health.

### 4. **Legal Automation (GenAI)**
   - **Auto-Drafting**: Generates *Reservation of Rights* letters and *Waiver Requests* based on LMA templates.
   - **Clause Intelligence**: Maps financial breaches to specific facility agreement clauses (e.g., Clause 22.1 Financial Covenants).

---

## 🛠️ Technical Architecture

### Backend (Python / FastAPI)
*   **Framework**: FastAPI for high-performance Async I/O.
*   **AI Core**: Custom `neural_engine.py` using PyTorch and `perforatedai`.
*   **Database**: SQLite for lightweight, relational data persistence.
*   **mockERP**: A sophisticated simulation of an ERP system generating realistic financial time-series data.

### Frontend (React / Electron)
*   **Framework**: React.js with Vite.
*   **Styling**: Modern CSS3 with Glassmorphism and animated gradients.
*   **Visualization**: Recharts for dynamic financial charting.
*   **Desktop wrapper**: Electron for native OS integration.

---

## ⚡ Getting Started

### Prerequisites
*   Python 3.9+
*   Node.js 16+
*   Git

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/kamaleshcit2024/lma-pulse.git
    cd lma-pulse
    ```

2.  **Backend Setup**
    ```bash
    # Create virtual environment
    python -m venv backend/venv
    source backend/venv/bin/activate  # or backend\venv\Scripts\activate on Windows

    # Install dependencies
    pip install -r backend/requirements.txt
    ```

3.  **Frontend Setup**
    ```bash
    cd frontend
    npm install
    ```

### Running the Application

**Step 1: Start the Backend (AI Server)**
```bash
# In project root
backend\venv\Scripts\python backend/main.py
```
*Wait for "Application startup complete" and "Dendritic Model Trained".*

**Step 2: Start the Frontend (Dashboard)**
```bash
# In frontend directory
npm run dev
```
*Access the dashboard at `http://localhost:5173` (or the Electron window).*

---

## 🧪 Testing the Features

1.  **Search**: Type **"Apex"** or **"Solaris"** in the top search bar and hit Enter.
2.  **Simulation**: Navigate to the **"Simulator"** tab (sidebar). Adjust the **Revenue Change** slider to **-15%**. Watch the "Projected Status" turn red/critical.
3.  **AI Prediction**: The system continuously runs the `DendriticSegments` model in the background to update risk scores.

---

## 📁 Directory Structure

```
lma-pulse/
├── backend/
│   ├── main.py                 # API Entry Point
│   ├── neural_engine.py        # Dendritic Neural Network Implementation
│   ├── mock_erp.py             # Financial Data Simulator
│   ├── covenant_engine.py      # Logic for testing covenants
│   └── models.py               # Pydantic Data Models
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx   # Main view
│   │   │   ├── Simulator.jsx   # Stress testing tool
│   │   │   └── Sidebar.jsx     # Navigation
│   │   └── services/api.js     # Axios API wrapper
└── README.md
```

---

## 📄 License
Confidential & Proprietary - LMA Pulse Project Team.
