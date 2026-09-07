PYTHON ?= python3
TRACE ?= traces/mooncake_trace.jsonl
OUTPUT_DIR ?= artifacts/mooncake
SGLANG_PYTHON_ROOT ?= ../sglang/python
PRIORITY ?= depth

.PHONY: analyze analyze-short web download-traces test clean-generated

analyze:
	FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-work MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src SGLANG_PYTHON_ROOT=$(SGLANG_PYTHON_ROOT) $(PYTHON) -m trace_analysis $(TRACE) --output-dir $(OUTPUT_DIR) --output-prefix mooncake --priority $(PRIORITY)

analyze-short:
	FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-work MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src SGLANG_PYTHON_ROOT=$(SGLANG_PYTHON_ROOT) $(PYTHON) -m trace_analysis traces/mooncake_trace_short.jsonl --output-dir artifacts/short --output-prefix mooncake_short

web:
	FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-work MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src SGLANG_PYTHON_ROOT=$(SGLANG_PYTHON_ROOT) $(PYTHON) -m trace_analysis.web --host 0.0.0.0 --port 8080

download-traces:
	$(PYTHON) scripts/download_hf_traces.py

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

clean-generated:
	@echo "Generated outputs are retained intentionally; remove a specific output directory manually."
