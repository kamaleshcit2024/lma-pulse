# LMA Pulse

LMA Pulse is a Covenant Monitoring System integrated with PerforatedAI for financial forecasting.

## Dendritic Implementation Details

### Strategy 1 (Parent Module: DendriticSegments)
We utilized **Strategy 1** to achieve high-precision covenant breach forecasting. This approach designates `DendriticSegments` as the parent module, allowing the PerforatedAI library to treat each segment as a primary unit of computation. This aligns our custom logic with the library's internal `DendriticLinear` scaling, effectively managing complex financial non-linearities (Frozen GAAP adjustments).

Key implementation details:
- **Parent Module**: `DendriticSegments` (Custom Class)
- **Library Modification**: `modules_perforatedai.py` modified to recognized `DendriticSegments` and maintain it as the parent.
- **Ignored Layers**: `DendriticLayer` wrapper is explicitly ignored to prevent conflict with internal logic.

## Project Structure
- **/backend**: FastAPI application, custom segments, and neural engine.
- **/frontend**: Electron/React application.
- **/perforatedai**: Modified PerforatedAI library.

## Getting Started
1. Install dependencies: `pip install -r backend/requirements.txt`
2. Run backend: `python backend/main.py`
3. Run frontend: `cd frontend && npm start`
