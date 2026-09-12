# Pinned local model setup

The final runtime uses the Windows x64 CUDA build below and Qwen3.5-4B in
Unsloth's Q4_K_M quantization. All inference runs on
`127.0.0.1:18085`; no account, API key, remote inference endpoint, or model-supplied
Python code is needed. Downloading public binaries and weights requires network
access during setup. The server subsequently starts with `--offline`.

## Sources and exact identities

| Component | Pinned identity |
|---|---|
| llama.cpp | [b10809](https://github.com/ggml-org/llama.cpp/releases/tag/b10809), commit `5266f24da75dc449bd56cbed7addb9c8e4a6a73e` |
| Windows runtime ZIP | `llama-b10809-bin-win-cuda-12.4-x64.zip`; 253,938,543 bytes; SHA-256 `c77bfcd9ed8d91e8721a2d6a290b907fddd4fa5412a47b21c6fa1709116b85f9` |
| Windows CUDA DLL ZIP | `cudart-llama-bin-win-cuda-12.4-x64.zip`; 391,443,627 bytes; SHA-256 `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6` |
| Measured Windows executable | `llama-server.exe`; SHA-256 `cb29f66008d4d73cce17cab2c2569ab318eb244b0b2ca2a15024b44d90dfcd3f` |
| Official base model | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a); revision observed during setup `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Third-party quantization | [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523); revision `e87f176479d0855a907a41277aca2f8ee7a09523` |
| Model file | `Qwen3.5-4B-Q4_K_M.gguf`; 2,740,937,888 bytes; SHA-256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |

ZIP hashes were checked against the official GitHub release asset digests. The
model hash was checked against the Unsloth repository's Hugging Face LFS metadata. The
executable hash is a local measurement after extraction. These checks establish
artifact identity; they are not third-party attestation of an experiment.

The GGUF is an Unsloth artifact, not an official Qwen GGUF. Its metadata names
Qwen/Qwen3.5-4B as the base model but does not attest the exact revision used for
conversion. The base revision above records what was observed during setup;
it must not be presented as a verified conversion lineage. The pinned
llama.cpp source includes the `qwen35` architecture, and this build loaded the
model successfully. No vision projector (`mmproj`) is loaded.

Licenses remain with their authors: [llama.cpp MIT](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/LICENSE),
[Qwen Apache-2.0](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/LICENSE),
and the separate [NVIDIA CUDA 12.4 license](https://docs.nvidia.com/cuda/archive/12.4.1/eula/index.html).
The repository does not redistribute model weights or CUDA DLLs.

## Windows PowerShell: download and verify

Run from the repository root. The runtime is stored in a sibling directory.
This uses an already working NVIDIA driver; it installs no driver or service.

```powershell
$ErrorActionPreference = 'Stop'
$runtimeDir = [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path '../rcwt-local-runtime'))
$binaryDir = Join-Path $runtimeDir 'bin'
$downloadDir = Join-Path $runtimeDir 'downloads'
$modelDir = Join-Path $runtimeDir 'models'
$null = New-Item -ItemType Directory -Force -Path $binaryDir, $downloadDir, $modelDir
$releaseUrl = 'https://github.com/ggml-org/llama.cpp/releases/download/b10809'
$artifacts = @(
    @{
        Name = 'llama-b10809-bin-win-cuda-12.4-x64.zip'
        Url = "$releaseUrl/llama-b10809-bin-win-cuda-12.4-x64.zip"
        Hash = 'c77bfcd9ed8d91e8721a2d6a290b907fddd4fa5412a47b21c6fa1709116b85f9'
        Bytes = 253938543; Directory = $downloadDir; Archive = $true
    },
    @{
        Name = 'cudart-llama-bin-win-cuda-12.4-x64.zip'
        Url = "$releaseUrl/cudart-llama-bin-win-cuda-12.4-x64.zip"
        Hash = '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'
        Bytes = 391443627; Directory = $downloadDir; Archive = $true
    },
    @{
        Name = 'Qwen3.5-4B-Q4_K_M.gguf'
        Url = 'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf?download=true'
        Hash = '00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'
        Bytes = 2740937888; Directory = $modelDir; Archive = $false
    }
)
foreach ($artifact in $artifacts) {
    $destination = Join-Path $artifact.Directory $artifact.Name
    if (-not (Test-Path -LiteralPath $destination)) {
        curl.exe --fail --location --retry 3 --output $destination $artifact.Url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $($artifact.Name)" }
    }
    $actual = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $artifact.Hash -or (Get-Item -LiteralPath $destination).Length -ne $artifact.Bytes) {
        throw "Artifact identity mismatch: $($artifact.Name)"
    }
    if ($artifact.Archive) { Expand-Archive -LiteralPath $destination -DestinationPath $binaryDir -Force }
}
$serverExe = Join-Path $binaryDir 'llama-server.exe'
$expectedExe = 'cb29f66008d4d73cce17cab2c2569ab318eb244b0b2ca2a15024b44d90dfcd3f'
if ((Get-FileHash -LiteralPath $serverExe -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedExe) {
    throw 'Executable identity mismatch'
}
& $serverExe --version
```

## Windows PowerShell: start the measured configuration

Use the same shell variables from setup. Reserve the server for one experiment;
do not send smoke tests or interactive requests during a timed campaign.

```powershell
if (Get-NetTCPConnection -LocalPort 18085 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 18085 already has a listener; inspect it before starting another server.'
}
$modelPath = Join-Path $modelDir 'Qwen3.5-4B-Q4_K_M.gguf'
$launchArgs = '-m "' + $modelPath + '" ' +
    '--alias rcwt-local-qwen35-4b --host 127.0.0.1 --port 18085 ' +
    '--ctx-size 4096 --parallel 1 --gpu-layers 99 --fit off ' +
    '--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 ' +
    '--batch-size 512 --ubatch-size 128 --threads 6 --threads-http 4 ' +
    '--cache-ram 0 --no-cache-prompt --offline --no-mmproj --no-webui --no-agent ' +
    '--no-ui-mcp-proxy --cors-origins localhost --no-cors-credentials ' +
    '--jinja --reasoning-budget 0 --log-colors off --log-timestamps --perf'
$serverProcess = Start-Process -FilePath $serverExe -ArgumentList $launchArgs `
    -WorkingDirectory $binaryDir -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runtimeDir 'server.stdout.log') `
    -RedirectStandardError (Join-Path $runtimeDir 'server.stderr.log')
$serverProcess.Id
```

After loading finishes, `/health` returns `{"status":"ok"}`. Check in another
PowerShell window:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:18085/health' -TimeoutSec 10
Invoke-RestMethod -Uri 'http://127.0.0.1:18085/v1/models' -TimeoutSec 10
```

## Linux Bash: equivalent source build and launch

The release has no Linux CUDA prebuilt asset. This alternative builds the same
source commit with an **already installed** CUDA toolkit, CMake, C++ compiler,
Git and build dependencies described in the [pinned build guide](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/docs/build.md).
It was not executed in the Windows experiment. The resulting executable has a
different hash and must be recorded as a new runtime; Windows timing results
must not be attributed to this build.

```bash
set -euo pipefail
runtime_dir="$(realpath -m ../rcwt-local-runtime-linux)"
mkdir -p "$runtime_dir/models"
source_dir="$runtime_dir/llama.cpp"
runtime_commit=5266f24da75dc449bd56cbed7addb9c8e4a6a73e
test ! -e "$source_dir"  # Use a fresh build directory.
git init "$source_dir"
git -C "$source_dir" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "$source_dir" fetch --depth 1 origin "$runtime_commit"
git -C "$source_dir" checkout --detach FETCH_HEAD
test "$(git -C "$source_dir" rev-parse HEAD)" = "$runtime_commit"
cmake -S "$source_dir" -B "$source_dir/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build "$source_dir/build" --config Release --target llama-server -j 2

model_path="$runtime_dir/models/Qwen3.5-4B-Q4_K_M.gguf"
model_hash=00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4
if [ ! -f "$model_path" ]; then
  curl --fail --location --retry 3 --output "$model_path" \
    'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf?download=true'
fi
printf '%s  %s\n' "$model_hash" "$model_path" | sha256sum --check -
server_exe="$source_dir/build/bin/llama-server"
"$server_exe" --version
sha256sum "$server_exe"

"$server_exe" -m "$model_path" \
  --alias rcwt-local-qwen35-4b --host 127.0.0.1 --port 18085 \
  --ctx-size 4096 --parallel 1 --gpu-layers 99 --fit off \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 512 --ubatch-size 128 --threads 6 --threads-http 4 \
  --cache-ram 0 --no-cache-prompt --offline --no-mmproj --no-webui --no-agent \
  --no-ui-mcp-proxy --cors-origins localhost --no-cors-credentials \
  --jinja --reasoning-budget 0 --log-colors off --log-timestamps --perf
```

The Bash server runs in the foreground. Stop that process before changing its
configuration. Binding fails if this port is already occupied. From another
terminal, inspect readiness with `curl --fail http://127.0.0.1:18085/health`.

## Tokenization, metering and hardware limits

The [client](../src/rcwt_local_model.py) uses `/tokenize` with `add_special=false`
and `parse_special=false`, and `/detokenize` for the memory cap. These endpoints
use the loaded Qwen tokenizer. Chat template tokens and instructions are counted
separately in the actual inference `usage` returned by llama.cpp. The client
sends `cache_prompt=false`, a fixed seed,
`chat_template_kwargs={"enable_thinking":false}`, and bounded `max_tokens`.
The per-request sampling profile is `temperature=0.7`, `top_p=0.8`, `top_k=20`,
`min_p=0.0`, `presence_penalty=1.5`, and llama.cpp `repeat_penalty=1.0`.
These are the official [Qwen3.5 non-thinking general-task settings](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md).
The server command above leaves sampling to the client; a different client
must explicitly send the same settings. Qwen3.5 does not officially support
Qwen3's textual thinking switches, so non-thinking is controlled by the
template parameter and the server's zero reasoning budget.
Schema-constrained output uses the OpenAI-style `response_format.json_schema`
object. The [pinned server documentation](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/tools/server/README.md)
describes these endpoints, `usage` and server `timings`.

The measured host had an AMD Ryzen 7 5700G CPU (8 cores, 16 logical processors),
an RTX 3060 Ti with 8 GiB VRAM and 32 GiB RAM. The server uses six CPU threads
and requests full GPU offload with `--gpu-layers 99`. The GGUF alone occupies
about 2.74 GB on disk; GPU buffers, CUDA runtime, KV cache and desktop
applications require additional memory. Approximately 1,816 MiB VRAM remained
free after the six-case Qwen3.5 setup pilot on that host. This is a
workload-dependent observation, not a hardware minimum
or a guarantee that another machine can load the configuration. If memory is
insufficient, stop this server, choose fewer GPU layers and record a new runtime
configuration before a new experiment. Do not silently change layers, context
size or precision within a frozen run.

The provider API charge is US$0 because inference is local. Electricity,
depreciation and total ownership cost were not measured. Report real token
counts, wall time and training/search overhead rather than treating local
computation as free. Server prompt caching is disabled to keep timed requests
independent; a single inference slot does not eliminate desktop load, thermal
effects or run-to-run variation.

## Development pilots are separate evidence

Earlier Qwen3 direct-action diagnostics, a Qwen3 thinking pilot and a Qwen3.5
direct-action pilot were retained as development work. They are not pooled
with the final experiment. The subsequent evidence-before-action Qwen3.5 pilot
passed five of six training probes, including all four initial steps, but still
failed the pending-payment probe. It supplied evidence to choose the acting
interface; it does not establish a memory-policy gain, held-out generalization
or production readiness.

Known partial development costs from recorded calls were 5,404 input / 473
output tokens and 11.52 seconds for ten Qwen3 direct-action diagnostic calls;
5,443 input / 2,984 output tokens and 56.44 seconds for the six-call Qwen3 thinking
pilot; and 5,521 input / 371 output tokens and 12.53 seconds for six Qwen3.5
direct-action calls. Thinking-pilot output counts include its reasoning tokens;
only a hash and character count of reasoning text were retained. These figures
exclude other setup, aborted runs and subsequent pilots, so they are not the
total project development cost. Each experiment's training/search and held-out
costs must be reported separately from these preliminary diagnostics.
