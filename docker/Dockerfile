# ============================== 1. Base ==============================
FROM registry-offlinebiz.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/ubuntu:ubuntu22_cuda12.8

COPY id_rsa.pub id_rsa /root/.ssh/
RUN chmod 600 /root/.ssh/id_rsa && mkdir -p /sources

USER root
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

# Build-time proxy only.
ENV http_proxy=http://10.217.142.63:8080
ENV https_proxy=http://10.217.142.63:8080
ENV no_proxy=localhost,127.0.0.1,pip.sankuai.com,pypi.sankuai.com,.sankuai.com

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=4

ARG PIP_CONFIG_FILE=/dev/null
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=300
ARG PIP_DISABLE_PIP_VERSION_CHECK=1
ARG PIP_NO_CACHE_DIR=1

# SCOPE (official upstream) is the center of the image; slime, tau2-bench and
# CooperBench arrive as its pinned submodules.
# Official main head as of 2026-11-01; submodule pins: slime 8efb1166 /
# tau2-bench c5b2d228 / CooperBench 44927734.
ARG SCOPE_COMMIT=14db64c655d55aa60eb24dac3184cb50e02e8a3c
# Training stack pins, aligned with the slime submodule's docker recipe.
ARG SGLANG_COMMIT=bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2
ARG MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
ARG APEX_COMMIT=10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
ARG TMS_COMMIT=dc6876905830430b5054325fa4211ff302169c6b
ARG MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c

ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=/usr/local/cuda-12.8/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST=8.0
ENV NVTE_CUDA_ARCHS=80
ENV NVTE_FRAMEWORK=pytorch

# ====================== 2. Ubuntu apt dependencies ======================
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        build-essential cmake ninja-build pkg-config \
        protobuf-compiler libprotobuf-dev \
        libnuma-dev libibverbs-dev librdmacm-dev \
        ibverbs-providers \
        libaio-dev libssl-dev libffi-dev \
        libsndfile1 ffmpeg rsync \
    && rm -rf /var/lib/apt/lists/*
# No python3.12-dev/python3.12-venv: this minimal base has no system Python
# 3.12 (jammy ships 3.10). Python comes from uv below.

# ==================== 3. uv + Python 3.12 environment ====================
# The old base shipped uv; this minimal CUDA base does not. Install uv, then
# let it fetch a managed CPython 3.12 and create /opt/venv.
ARG UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN curl --retry 3 --retry-delay 5 -fsSL \
        https://astral.sh/uv/install.sh -o /tmp/uv-install.sh && \
    UV_UNMANAGED_INSTALL=/usr/local/bin sh /tmp/uv-install.sh && \
    rm /tmp/uv-install.sh && \
    uv --version

# uv only creates the environment. Package installation uses pip.
RUN uv venv --python 3.12 /opt/venv && \
    /opt/venv/bin/python -m ensurepip --upgrade

# No system python in this base; default python3/pip to the venv.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:${PATH}

RUN source /opt/venv/bin/activate && \
    python -m pip install --upgrade \
        pip==25.3 setuptools==79.0.1 wheel==0.45.1 \
        packaging ninja cmake pybind11 \
        setuptools-rust setuptools-scm psutil

# Pin the compiled stack so later installs cannot silently bump it. cudnn/nccl
# stay at the versions torch 2.9.1 metadata requires; the upstream recipe's
# cuDNN 9.16.0.29 override (H200 workaround) is NOT applied.
RUN mkdir -p /opt/scope-build && \
    printf '%s\n' \
        'pip==25.3' \
        'setuptools==79.0.1' \
        'wheel==0.45.1' \
        'torch==2.9.1+cu128' \
        'torchvision==0.24.1+cu128' \
        'torchaudio==2.9.1+cu128' \
        'triton==3.5.1' \
        'numpy==1.26.4' \
        'scipy==1.15.3' \
        'transformers==4.57.1' \
        'huggingface-hub==0.36.2' \
        'kernels==0.9.0' \
        'tokenizers==0.22.1' \
        'openai==2.6.1' \
        'litellm==1.79.0' \
        'weave==0.52.15' \
        'wandb==0.22.3' \
        'ray==2.50.1' \
        'cuda-python==12.9.0' \
        'sgl-kernel==0.3.21' \
        'flashinfer-python==0.6.3' \
        'flashinfer-cubin==0.6.3' \
        'torch-memory-saver==0.0.9' \
        'torchao==0.9.0' \
        'torchcodec==0.8.0' \
        'quack-kernels==0.2.4' \
        'nvidia-cutlass-dsl==4.3.4' \
        'flash-attn==2.7.4.post1' \
        'ring-flash-attn==0.1.8' \
        'transformer-engine==2.10.0' \
        'transformer-engine-cu12==2.10.0' \
        'transformer-engine-torch==2.10.0' \
        'sglang-router==0.3.2' \
        'gymnasium==1.2.2' \
        'compressed-tensors==0.12.2' \
        'nvidia-cudnn-cu12==9.10.2.21' \
        'nvidia-nccl-cu12==2.27.5' \
        'sglang==0.5.9+scope.bbe9c7eeb520' \
        > /opt/scope-build/constraints.txt

ARG PIP_CONSTRAINT=/opt/scope-build/constraints.txt

# ==================== 4. SCOPE project + submodules ====================
# Clone the official upstream at the pinned commit and initialize all pinned
# submodules completely (full history so in-place patches and rebuilds work;
# Megatron's own nested submodules are handled in step 5 separately).
RUN git clone https://github.com/CHATS-lab/scope_usim.git /opt/scope_usim && \
    git -C /opt/scope_usim checkout "${SCOPE_COMMIT}" && \
    git -C /opt/scope_usim submodule update --init --recursive

# ================== 5. Training stack per slime's recipe ==================
# SGLang and Megatron versions come from the slime submodule's docker recipe
# (sglang v0.5.9 base + megatron 3714d81d + apex 10417ace + tms dc687690).
# Install torch first: every compiled extension below builds against it.
RUN source /opt/venv/bin/activate && \
    python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 && \
    python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda == '12.8'"

RUN source /opt/venv/bin/activate && \
    python -m pip install numpy==1.26.4 scipy==1.15.3 cuda-python==12.9.0

# Rust toolchain for SGLang's Rust extension.
RUN curl --retry 3 --retry-delay 5 -fsSL \
        https://sh.rustup.rs -o /tmp/slime-rustup.sh && \
    CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup \
        sh /tmp/slime-rustup.sh -y \
        --profile minimal \
        --default-toolchain stable \
        --no-modify-path && \
    rm /tmp/slime-rustup.sh

RUN git clone https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM && \
    git -C /opt/Megatron-LM checkout "${MEGATRON_COMMIT}" && \
    git -C /opt/Megatron-LM submodule update --init --recursive

# Patched SGLang source, then editable install (before megatron: megatron's
# pyproject and slime's requirements both resolve against installed sglang).
RUN git clone https://github.com/sgl-project/sglang.git /opt/SGLang && \
    git -C /opt/SGLang checkout "${SGLANG_COMMIT}" && \
    git -C /opt/SGLang apply --check \
        /opt/scope_usim/slime/docker/patch/latest/sglang.patch && \
    git -C /opt/SGLang apply \
        /opt/scope_usim/slime/docker/patch/latest/sglang.patch && \
    git -C /opt/SGLang diff > /opt/scope-build/sglang-local.patch

# Memory-saver fork pinned by the slime recipe, built against installed torch.
RUN source /opt/venv/bin/activate && \
    TMS_CUDA_MAJOR=12 MAX_JOBS="${MAX_JOBS}" \
    python -m pip install --no-build-isolation \
        "git+https://github.com/fzyzcjy/torch_memory_saver.git@${TMS_COMMIT}"

# sgl-kernel wheel first, then the custom sglang-router fork wheel.
RUN source /opt/venv/bin/activate && \
    python -m pip install sgl-kernel==0.3.21 && \
    python -m pip install \
        https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl

# Editable-install the patched SGLang. Label pins the downstream version;
# source identity is SGLANG_COMMIT (setuptools-scm must not see newer tags).
RUN source /opt/venv/bin/activate && \
    export CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup && \
    export PATH="/opt/cargo/bin:${PATH}" && \
    export CARGO_BUILD_JOBS="${MAX_JOBS}" && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.9+scope.bbe9c7eeb520 \
        python -m pip install --no-build-isolation \
        -e /opt/SGLang/python

# Megatron-LM: apply slime's megatron patch (carries trailing whitespace
# upstream; ignore blank-at-eol in the check), then editable install.
RUN cd /opt/Megatron-LM && \
    git apply --check /opt/scope_usim/slime/docker/patch/latest/megatron.patch && \
    git apply /opt/scope_usim/slime/docker/patch/latest/megatron.patch && \
    git -c core.whitespace=-blank-at-eol diff --check && \
    git -C /opt/Megatron-LM diff > /opt/scope-build/megatron-local.patch && \
    source /opt/venv/bin/activate && \
    python -m pip install --no-build-isolation -e . && \
    python -c "import pathlib, sysconfig; p = pathlib.Path(sysconfig.get_path('purelib')) / 'megatron_source.pth'; p.write_text('/opt/Megatron-LM\n')"

# FA2 for SM80 (no Hopper FA3 in this profile).
RUN source /opt/venv/bin/activate && \
    MAX_JOBS="${MAX_JOBS}" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTN_CUDA_ARCHS=80 \
    python -m pip install -v --no-build-isolation \
        flash-attn==2.7.4.post1

# TE 2.10.0 has no prebuilt cp312+cu128 wheel (the guessed GitHub URL 404s and
# PyPI ships transformer_engine_torch as sdist only), so the torch extension is
# built from source. The minimal base lacks system NCCL/cuDNN dev headers; the
# venv's nvidia wheels carry them, and the CUDA toolkit provides CCCL. Export
# them via CPATH exactly like the previous successful 1.0.2 image recipe.
RUN source /opt/venv/bin/activate && \
    python -m pip install \
        "transformer-engine[core-cu12]==2.10.0" && \
    export NCCL_INCLUDE=/opt/venv/lib/python3.12/site-packages/nvidia/nccl/include && \
    export CUDNN_INCLUDE=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/include && \
    test -f "${NCCL_INCLUDE}/nccl.h" && \
    test -f "${CUDNN_INCLUDE}/cudnn.h" && \
    CCCL_INCLUDE="" && \
    for dir in \
        "${CUDA_HOME}/include" \
        "${CUDA_HOME}/include/cccl" \
        "${CUDA_HOME}/targets/x86_64-linux/include" \
        "${CUDA_HOME}/targets/x86_64-linux/include/cccl"; do \
        if [ -f "${dir}/nv/target" ]; then \
            CCCL_INCLUDE="${dir}"; \
            break; \
        fi; \
    done && \
    if [ -z "${CCCL_INCLUDE}" ]; then \
        echo "ERROR: nv/target is missing from the CUDA Toolkit; CCCL development headers are required."; \
        exit 1; \
    fi && \
    echo "Using CCCL headers: ${CCCL_INCLUDE}" && \
    export CPATH="${CUDA_HOME}/include:${CUDA_HOME}/targets/x86_64-linux/include:${CCCL_INCLUDE}:${NCCL_INCLUDE}:${CUDNN_INCLUDE}${CPATH:+:${CPATH}}" && \
    NVTE_BUILD_USE_NVIDIA_WHEELS=0 \
    NVTE_PYTORCH_FORCE_BUILD=TRUE \
    MAX_JOBS="${MAX_JOBS}" \
    python -m pip install -v --no-build-isolation \
        "transformer-engine[pytorch]==2.10.0"

# The TE cu12 core wheel keeps the known NVIDIA packaging defect (py3-none
# filename, cpNNN internal WHEEL tag), which makes pip check report
# "transformer_engine_cu12 2.10.0 is not supported on this platform". Fix only
# this defect in place and update RECORD, like the 1.0.2 recipe did for 2.16.1.
RUN printf '%s\n' \
        'import base64, csv, hashlib, io, re, subprocess' \
        'from pathlib import Path' \
        'from importlib.metadata import distribution' \
        'd = distribution("transformer-engine-cu12")' \
        'assert d.version == "2.10.0", d.version' \
        'files = list(d.files)' \
        'wheel_rel = next(f for f in files if f.name == "WHEEL")' \
        'record_rel = next(f for f in files if f.name == "RECORD")' \
        'wheel_path = Path(d.locate_file(wheel_rel))' \
        'record_path = Path(d.locate_file(record_rel))' \
        'old = wheel_path.read_text()' \
        'tag_lines = [line for line in old.splitlines() if line.startswith("Tag:")]' \
        'assert len(tag_lines) == 1, tag_lines' \
        'new_line, n = re.subn(r"^Tag: cp\d+-cp\d+-(.+)$", r"Tag: py3-none-\1", tag_lines[0])' \
        'assert n == 1 and new_line != tag_lines[0], tag_lines[0]' \
        'libs = [f for f in files if f.name == "libtransformer_engine.so"]' \
        'assert len(libs) == 1, libs' \
        'assert not any(".cpython-" in str(f) and str(f).endswith(".so") for f in files), [str(f) for f in files if str(f).endswith(".so")]' \
        'symbols = subprocess.check_output(["nm", "-D", str(d.locate_file(libs[0]))], text=True)' \
        'assert "PyInit_" not in symbols, "Unexpected CPython extension in TE core"' \
        'Path("/opt/scope-build/te-core-WHEEL.original").write_text(old)' \
        'new = old.replace(tag_lines[0], new_line)' \
        'payload = new.encode()' \
        'rows = list(csv.reader(io.StringIO(record_path.read_text())))' \
        'matches = [row for row in rows if row[0] == str(wheel_rel)]' \
        'assert len(matches) == 1, matches' \
        'matches[0][1] = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()' \
        'matches[0][2] = str(len(payload))' \
        'wheel_path.write_bytes(payload)' \
        'with record_path.open("w", newline="") as out: csv.writer(out).writerows(rows)' \
        'print("Corrected TE cu12 2.10.0 internal wheel tag to match its py3-none filename; RECORD updated.")' \
        > /opt/scope-build/fix_te_core_wheel_tag.py && \
    /opt/venv/bin/python /opt/scope-build/fix_te_core_wheel_tag.py

# Apex from source at the commit pinned by the slime recipe.
RUN source /opt/venv/bin/activate && \
    APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    MAX_JOBS="${MAX_JOBS}" \
    NVCC_APPEND_FLAGS="--threads 2" \
    python -m pip install -v --no-build-isolation \
        "git+https://github.com/NVIDIA/apex.git@${APEX_COMMIT}"

# mbridge (ISEEKYAN fork, pinned by the slime recipe) is needed by
# convert_hf_to_torch_dist. Distinct from Megatron-Bridge (not installed).
RUN source /opt/venv/bin/activate && \
    python -m pip install --no-deps \
        "git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}"

# ============== 6. SCOPE project install, per its README ==============
# Apply the Co-Training compatibility patch to the slime submodule first
# (Single does not need it; baking it in makes every launcher work).
RUN git -C /opt/scope_usim/slime apply --check \
        /opt/scope_usim/patches/slime_cotrain_combined.patch && \
    git -C /opt/scope_usim/slime apply \
        /opt/scope_usim/patches/slime_cotrain_combined.patch

# README order: scope extras -> slime submodule -> tau2-bench submodule.
# slime itself is installed --no-deps, so its requirements.txt must be installed
# explicitly (same pattern as the 1.0.2 recipe and the upstream slime recipe).
RUN source /opt/venv/bin/activate && \
    cd /opt/scope_usim && \
    python -m pip install -e ".[slime,p4g,tau2]" && \
    python -m pip install -r ./slime/requirements.txt && \
    python -m pip install --no-deps --no-build-isolation -e ./slime && \
    python -m pip install -e ./external/tau2-bench

# ====================== 7. Permissions and cleanup ======================
# Reconcile the grpc stack before pip check. The company mirror's latest
# smg-grpc-proto (0.4.28, pulled via sglang's smg-grpc-proto>=0.3.3) demands
# grpcio>=1.81.1, whose grpcio-health-checking in turn demands protobuf>=7 —
# incompatible with wandb's protobuf<7. The consistent set verified against the
# mirror: smg-grpc-proto 0.4.10 (last release with grpcio>=1.78.0) + grpcio trio
# 1.78.0 (protobuf >=6.31.1,<7) + protobuf 6.33.6.
RUN source /opt/venv/bin/activate && \
    python -m pip install \
        smg-grpc-proto==0.4.10 \
        grpcio==1.78.0 grpcio-health-checking==1.78.0 grpcio-reflection==1.78.0 \
        protobuf==6.33.6 && \
    python -m pip check && \
    python -m pip freeze > /opt/scope-build/pip-freeze.txt && \
    printf 'scope=%s\nsglang=%s\nmegatron=%s\napex=%s\ntms=%s\nmbridge=%s\n' \
        "${SCOPE_COMMIT}" "${SGLANG_COMMIT}" "${MEGATRON_COMMIT}" \
        "${APEX_COMMIT}" "${TMS_COMMIT}" "${MBRIDGE_COMMIT}" \
        > /opt/scope-build/source-versions.txt

RUN chmod -R a+rwX /opt/venv
