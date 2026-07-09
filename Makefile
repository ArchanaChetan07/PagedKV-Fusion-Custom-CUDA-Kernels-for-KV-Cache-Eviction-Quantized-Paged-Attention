.PHONY: install install-cuda lint test test-gpu bench bench-quality \
        demo profile docker-build docker-test clean

# --- Development (works on any machine, no GPU) -----------------------------

install:        ## Install reference-path package + dev tools
	pip install -e ".[dev]"

lint:           ## Static checks (ruff)
	ruff check pagedkv_fusion tests benchmarks integration scripts

test:           ## CPU test suite (torch/GPU-gated tests skip cleanly)
	pytest tests/ -v --durations=10

demo:           ## Run the full pipeline end-to-end on the reference path
	python scripts/run_end_to_end_demo.py

bench:          ## Memory + eviction-latency benchmarks (CPU sections)
	python benchmarks/bench_eviction.py --num-blocks 1024 16384 --out results/eviction_bench.json
	python benchmarks/bench_quant_attention.py --out results/quant_bench.json

bench-quality:  ## Downstream-accuracy proxy benchmark (see disclaimer in output)
	python benchmarks/bench_downstream_proxy.py --out results/downstream_proxy.json

# --- GPU-only targets (require nvcc + a CUDA device) ------------------------

install-cuda:   ## Build the CUDA extension (fails without CUDA toolchain)
	PAGEDKV_FORCE_CUDA=1 pip install -e ".[cuda,dev]" --no-build-isolation

test-gpu: install-cuda  ## Kernel-vs-reference correctness on GPU
	pytest tests/test_kernels_gpu.py -v

profile: install-cuda   ## Nsight Compute/Systems traces
	python scripts/profile_kernels.py both --out results/profiles

# --- Docker (reproducible CUDA build environment) ---------------------------

docker-build:   ## Build the CUDA dev image (see docker/Dockerfile.cuda)
	docker build -f docker/Dockerfile.cuda -t pagedkv-fusion:cuda .

docker-test: docker-build  ## Run the GPU test suite inside the container
	docker run --rm --gpus all pagedkv-fusion:cuda make test-gpu

clean:
	rm -rf build *.egg-info **/__pycache__ .pytest_cache .ruff_cache
