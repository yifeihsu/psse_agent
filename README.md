# PS-LLM-Agent

PS-LLM-Agent is a platform for generating power system operational data and using that data to train and fine-tune Large Language Models (LLMs). The project features comprehensive tool-use capabilities through a Model Context Protocol (MCP) server that interfaces with MATPOWER, enabling LLMs to perform complex power system calculations like State Estimation, Topology Error Processing, and Harmonics Analysis.

## Repository Structure

The repository is organized into the following key components:

- **`Transmission/`**: Core MATLAB and Python scripts for generating foundational transmission data, including State Estimation (SE) measurements, topology scenarios (e.g., node-breaker models), and Harmonics State Estimation (HSE) traces.
- **`Harmonics/`**: Scripts focusing on harmonics data processing and analysis.
- **`IEEE_14_OpenDSS/`**: OpenDSS models and scripts tailored for the IEEE 14-bus test system.
- **`mcp_server/`**: A Model Context Protocol (MCP) server that wraps MATPOWER functionalities, exposing power system tools directly to LLM agents.
- **`scripts/`**: Utility scripts for various tasks, including running fine-tuning (e.g., Unsloth), verifying parameters, testing server integrations, and generating data splits.
- **`data/`**: The generated datasets, predominantly in JSONL format, used for Supervised Fine-Tuning (SFT) of the LLMs (e.g., `sft_final.jsonl`, `sft_with_tools.jsonl`).
- **`docs/`**: Project documentation, presentation slides, and parameter range specifications.

## Key Features

1. **Synthetic Data Generation**: Robust tools combining Python and MATLAB (MATPOWER) to generate realistic power system scenarios, including normal operations, topological errors, and harmonic distortions.
2. **LLM Tool-Use (MCP)**: A fully functional MATPOWER MCP server that allows an LLM agent to execute load flows and state estimation directly during inference.
3. **Supervised Fine-Tuning (SFT) Ready**: Scripts seamlessly convert MATPOWER executions and measurements into structured conversational traces (`.jsonl`) optimized for fine-tuning open-source models (via `unsloth`).

## Getting Started

### Prerequisites
- **Python 3.10+** (Recommended)
- **MATLAB** (with MATPOWER installed and accessible via the MATLAB Engine API for Python)
- **Gurobi** (optional but recommended for certain continuous optimization tasks)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd ps_llm_agent
   ```

2. **Install Python dependencies:**
   A virtual environment is highly recommended.
   ```bash
   pip install -r requirements.txt # (If provided, otherwise install package dependencies manually)
   # Key dependencies often include: matlabengine, numpy, pandas, unsloth, etc.
   ```

3. **Start the MCP Server:**
   To allow an LLM to interact with MATPOWER:
   ```bash
   cd mcp_server
   python run_http_server.py
   # Standard stdio version:
   # python matpower_server.py
   ```

### Data Generation & Fine-Tuning

- **Generate Data**: Use the scripts in `Transmission/` (e.g., `generate_measurements.py`, `generate_hse_traces.py`) to build the raw datasets.
- **Build SFT Traces**: Convert raw data into LLM training formats using `Transmission/build_sft_traces.py`.
- **Fine-Tune**: Use `scripts/test_unsloth.py` or equivalent to run the fine-tuning process on the generated `data/sft_final.jsonl`.

## License
Refer to `LICENSE.md` in the root directory for licensing information.
