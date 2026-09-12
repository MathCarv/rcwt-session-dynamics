#requires -Version 7.2
param(
    [Alias('Action')][ValidateSet('start', 'status', 'stop')][string]$Mode = 'status',
    [Alias('OutputDir')][string]$OutputDirectory = '.runs/r2_runtime_01',
    [string]$RuntimeRoot = '',
    [ValidateRange(1, 60)][int]$ReadinessSeconds = 60
)

# Public helper; machine paths belong only in the exclusive private receipts.
# Dot-sourcing defines functions only, for synthetic tests. It never dispatches.
$script:R2Project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$script:R2HelperPath = $PSCommandPath
$script:R2ConfigSha = '8ba7e358f11f584af81006d96287d1a86ec1d4529ec475b8bb2f78fa7ef45515'
$script:R2ExeSha = 'cb29f66008d4d73cce17cab2c2569ab318eb244b0b2ca2a15024b44d90dfcd3f'
$script:R2ModelSha = '00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'
$script:R2CreatedThisInvocation = $false
$script:R2LaunchObservation = $null

function Resolve-R2Ordinary([string]$Path, [switch]$MissingAllowed) {
    if (-not $Path -or $Path.IndexOfAny([char[]]@('"', "`r", "`n", [char]0)) -ge 0) { throw 'Invalid local path.' }
    $absolute = [IO.Path]::GetFullPath($Path)
    if ($absolute.StartsWith('\\') -or $absolute -notmatch '^[A-Za-z]:\\') { throw 'Only local Windows drive paths are permitted.' }
    $current = [IO.Path]::GetPathRoot($absolute)
    foreach ($part in $absolute.Substring($current.Length).Split([char]'\', [StringSplitOptions]::RemoveEmptyEntries)) {
        $current = Join-Path $current $part
        try { $entry = Get-Item -LiteralPath $current -Force -ErrorAction Stop } catch [Management.Automation.ItemNotFoundException] {
            if ($MissingAllowed) { continue }
            throw 'Required local path is missing.'
        }
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse paths are forbidden.' }
    }
    $absolute
}

function Get-R2Sha([string]$Path) {
    $checked = Resolve-R2Ordinary $Path
    if (-not (Test-Path -LiteralPath $checked -PathType Leaf)) { throw 'Expected a regular file.' }
    (Get-FileHash -LiteralPath $checked -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
}

function Get-R2TextSha([string]$Text) {
    [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($Text))).ToLowerInvariant()
}

function Write-R2NewJson([string]$Path, $Value) {
    $checked = Resolve-R2Ordinary $Path -MissingAllowed
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 20) + "`n")
    $stream = [IO.File]::Open($checked, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
    $expected = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
    if ((Get-R2Sha $checked) -cne $expected) { throw 'Receipt readback differs.' }
    $expected
}

function Read-R2Json([string]$Path) {
    $checked = Resolve-R2Ordinary $Path
    Get-Content -LiteralPath $checked -Raw -Encoding UTF8 | ConvertFrom-Json -AsHashtable -ErrorAction Stop
}

function Resolve-R2Output([string]$Directory, [switch]$MissingAllowed) {
    if (-not [IO.Path]::IsPathFullyQualified($Directory)) { $Directory = Join-Path $script:R2Project $Directory }
    $checked = Resolve-R2Ordinary $Directory -MissingAllowed:$MissingAllowed
    $parent = [IO.Path]::GetFullPath((Join-Path $script:R2Project '.runs'))
    if ([IO.Path]::GetDirectoryName($checked) -ine $parent -or [IO.Path]::GetFileName($checked) -notmatch '^r2_runtime_[A-Za-z0-9_-]+$') {
        throw 'Runtime evidence must be an immediate r2_runtime_* directory under project .runs.'
    }
    $checked
}

function Get-R2Arguments([string]$ModelPath) {
    '-m "' + $ModelPath + '" --alias rcwt-local-qwen35-4b --host 127.0.0.1 --port 18085 --ctx-size 4096 --parallel 1 --gpu-layers 99 --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 512 --ubatch-size 128 --threads 6 --threads-http 4 --cache-ram 0 --no-cache-prompt --offline --no-mmproj --no-webui --no-agent --no-ui-mcp-proxy --cors-origins localhost --no-cors-credentials --jinja --reasoning-budget 0 --log-colors off --log-timestamps --perf'
}

function Get-R2Contract([string]$LocalRuntimeRoot) {
    $config = Resolve-R2Ordinary (Join-Path $script:R2Project 'docs/rcwt_agent_runtime.json')
    if ((Get-R2Sha $config) -cne $script:R2ConfigSha) { throw 'Frozen runtime configuration changed.' }
    $settings = Read-R2Json $config
    $runtime = Resolve-R2Ordinary $LocalRuntimeRoot
    if (-not (Test-Path -LiteralPath $runtime -PathType Container)) { throw 'Runtime root must be an ordinary directory.' }
    $binary = Resolve-R2Ordinary (Join-Path $runtime 'llama-b10809/llama-server.exe')
    $model = Resolve-R2Ordinary (Join-Path $runtime ('models/' + $settings.model_file))
    $binarySize = (Get-Item -LiteralPath $binary).Length
    $modelSize = (Get-Item -LiteralPath $model).Length
    if ($binarySize -ne 9216 -or $modelSize -ne 2740937888 -or $modelSize -ne $settings.model_bytes) { throw 'Pinned executable/model size mismatch.' }
    if ((Get-R2Sha $binary) -cne $script:R2ExeSha -or (Get-R2Sha $model) -cne $script:R2ModelSha) { throw 'Pinned executable/model hash mismatch.' }
    [ordered]@{ runtime_root = $runtime; executable = $binary; model = $model
        executable_sha256 = $script:R2ExeSha; executable_bytes = $binarySize
        model_sha256 = $script:R2ModelSha; model_bytes = $modelSize
        runtime_config_sha256 = $script:R2ConfigSha; helper_sha256 = (Get-R2Sha $script:R2HelperPath)
        arguments = (Get-R2Arguments $model); host = '127.0.0.1'; port = 18085 }
}

function Get-R2PortConnections {
    @(Get-NetTCPConnection -ErrorAction Stop | Where-Object { $_.LocalPort -eq 18085 })
}

function Get-R2Identity($Contract, [Diagnostics.Process]$Process) {
    $null = $Process.Handle # Hold the exact process object before later termination.
    if ($Process.HasExited) { throw 'Owned process is not running.' }
    $cim = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $Process.Id) -ErrorAction Stop
    if (-not $cim) { throw 'Process identity could not be observed.' }
    $quoted = '"' + $Contract.executable + '" ' + $Contract.arguments
    $unquoted = $Contract.executable + ' ' + $Contract.arguments
    if ($cim.ExecutablePath -cne $Contract.executable -or $Process.MainModule.FileName -cne $Contract.executable -or
        ($cim.CommandLine -cne $quoted -and $cim.CommandLine -cne $unquoted)) { throw 'Executable or exact command line mismatch.' }
    [ordered]@{ pid = $Process.Id; process_start_time_utc = $Process.StartTime.ToUniversalTime().ToString('o')
        process_creation_time_utc = $cim.CreationDate.ToUniversalTime().ToString('o')
        executable = $cim.ExecutablePath; command_line = $cim.CommandLine }
}

function Assert-R2Identity($Expected, $Observed) {
    foreach ($key in @('pid', 'process_start_time_utc', 'process_creation_time_utc', 'executable', 'command_line')) {
        if ($Expected[$key] -cne $Observed[$key]) { throw ('Owned process identity changed: ' + $key) }
    }
}

function Read-R2OwnedBundle([string]$Directory) {
    $intentHash = Get-R2Sha (Join-Path $Directory 'intent.json')
    $startedHash = Get-R2Sha (Join-Path $Directory 'started.json')
    $intent = Read-R2Json (Join-Path $Directory 'intent.json')
    $started = Read-R2Json (Join-Path $Directory 'started.json')
    $contract = Get-R2Contract $intent.contract.runtime_root
    if ($intent.schema -cne 'rcwt-r2-runtime-intent/1' -or $started.schema -cne 'rcwt-r2-runtime-started/1' -or
        $intent.output_directory -cne $Directory -or $started.intent_sha256 -cne (Get-R2Sha (Join-Path $Directory 'intent.json'))) { throw 'Owned runtime receipts do not bind.' }
    foreach ($key in $contract.Keys) {
        if ($intent.contract[$key] -cne $contract[$key]) { throw 'Runtime contract differs from launch intent.' }
    }
    if ($started.identity.pid -isnot [int] -and $started.identity.pid -isnot [long]) { throw 'Invalid owned process identifier.' }
    if ($started.identity.pid -le 0) { throw 'Invalid owned process identifier.' }
    $bundle = [ordered]@{ intent = $intent; started = $started; contract = $contract
        intent_sha256 = $intentHash; started_sha256 = $startedHash }
    Assert-R2BundleStable $Directory $bundle
    $bundle
}

function Assert-R2BundleStable([string]$Directory, $Bundle) {
    if ((Get-R2Sha (Join-Path $Directory 'intent.json')) -cne $Bundle.intent_sha256 -or
        (Get-R2Sha (Join-Path $Directory 'started.json')) -cne $Bundle.started_sha256 -or
        (Get-R2Sha $script:R2HelperPath) -cne $Bundle.contract.helper_sha256 -or
        (Get-R2Sha (Join-Path $script:R2Project 'docs/rcwt_agent_runtime.json')) -cne $Bundle.contract.runtime_config_sha256) {
        throw 'Owned runtime receipts or verification source changed during the operation.'
    }
}

function Invoke-R2Health([int]$TimeoutMilliseconds = 2000) {
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.UseProxy = $false
    $handler.AllowAutoRedirect = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromMilliseconds($TimeoutMilliseconds)
    try {
        $response = $client.GetAsync('http://127.0.0.1:18085/health').GetAwaiter().GetResult()
        try {
            if ([int]$response.StatusCode -ne 200) { return $false }
            $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json -AsHashtable -ErrorAction Stop
            return ($body.status -ceq 'ok')
        } finally { $response.Dispose() }
    } catch { return $false } finally { $client.Dispose(); $handler.Dispose() }
}

function Assert-R2Listener([int]$OwnedPid) {
    $listeners = @(Get-R2PortConnections | Where-Object { $_.State -eq 'Listen' })
    if ($listeners.Count -ne 1 -or $listeners[0].LocalAddress -cne '127.0.0.1' -or $listeners[0].OwningProcess -ne $OwnedPid) { throw 'The owned process is not the sole loopback listener.' }
}

function Get-R2StableLog([string]$Path) {
    $checked = Resolve-R2Ordinary $Path
    $stream = [IO.File]::Open($checked, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
    try {
        $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)).ToLowerInvariant()
        $length = $stream.Length
        $stream.Position = 0
        $again = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)).ToLowerInvariant()
        if ($hash -cne $again) { throw 'Closed log changed during exclusive read.' }
    } finally { $stream.Dispose() }
    [ordered]@{ sha256 = $hash; bytes = $length; exclusive_read = $true; stable_after_owned_exit = $true; call_accounting_complete_claim = $false }
}

function Get-R2Status([string]$Directory) {
    $bundle = Read-R2OwnedBundle $Directory
    $contract, $started = $bundle.contract, $bundle.started
    $pidValue = [int]$started.identity.pid
    $observed = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $pidValue) -ErrorAction Stop
    $healthRequests = 0
    $healthStatus = 'not_requested'
    if ($observed) {
        if (Test-Path -LiteralPath (Join-Path $Directory 'closed.json')) { throw 'A closed receipt cannot authorize a live process.' }
        $process = [Diagnostics.Process]::GetProcessById($pidValue)
        try {
            Assert-R2Identity $started.identity (Get-R2Identity $contract $process)
            Assert-R2Listener $pidValue
            $healthRequests = 1
            if (-not (Invoke-R2Health)) { throw 'Owned runtime health check failed.' }
            $healthStatus = 'ok'
            Assert-R2Identity $started.identity (Get-R2Identity $contract $process)
            Assert-R2Listener $pidValue
        } finally { $process.Dispose() }
        $state = 'READY'
    } elseif (Test-Path -LiteralPath (Join-Path $Directory 'closed.json')) {
        $closed = Read-R2Json (Join-Path $Directory 'closed.json')
        if ($closed.started_sha256 -cne (Get-R2Sha (Join-Path $Directory 'started.json')) -or $closed.identity.pid -ne $pidValue) { throw 'Closed ownership receipt differs.' }
        foreach ($name in @('server.stdout.log', 'server.stderr.log')) {
            $log = Get-R2StableLog (Join-Path $Directory $name)
            if ($log.sha256 -cne $closed.logs[$name].sha256 -or $log.bytes -ne $closed.logs[$name].bytes) { throw 'Closed runtime log bytes differ.' }
        }
        $state = 'CLOSED'
    } else { $state = 'NOT_RUNNING' }
    Assert-R2BundleStable $Directory $bundle
    [ordered]@{ schema = 'rcwt-r2-runtime-status/1'; pass = $true; status = $state; read_only = $true
        inference_calls = 0; health_requests = $healthRequests; health_status = $healthStatus
        pid = $pidValue; process_start_time_utc = $started.identity.process_start_time_utc
        executable_sha256 = $contract.executable_sha256; model_sha256 = $contract.model_sha256; model_bytes = $contract.model_bytes
        runtime_config_sha256 = $contract.runtime_config_sha256; helper_sha256 = $contract.helper_sha256
        started_receipt_sha256 = $bundle.started_sha256
        intent_receipt_sha256 = $bundle.intent_sha256
        arguments_sha256 = (Get-R2TextSha $contract.arguments); host = '127.0.0.1'; port = 18085 }
}

function Start-R2Runtime([string]$Directory, [string]$LocalRuntimeRoot, [int]$Seconds) {
    if (-not $LocalRuntimeRoot) { throw 'Start requires an explicit RuntimeRoot.' }
    if (Test-Path -LiteralPath $Directory) { throw 'Runtime evidence directory already exists; refusing restart or overwrite.' }
    $contract = Get-R2Contract $LocalRuntimeRoot
    if (@(Get-R2PortConnections).Count -ne 0) { throw 'Port 18085 is not free; no process changed.' }
    $parent = Resolve-R2Ordinary ([IO.Path]::GetDirectoryName($Directory))
    $null = New-Item -ItemType Directory -Path $Directory -ErrorAction Stop
    $script:R2CreatedThisInvocation = $true
    $intent = [ordered]@{ schema = 'rcwt-r2-runtime-intent/1'; created_at_utc = [DateTime]::UtcNow.ToString('o')
        output_directory = $Directory; contract = $contract; inference_policy = 'local_only_no_paid_api' }
    $intentHash = Write-R2NewJson (Join-Path $Directory 'intent.json') $intent
    $out = Join-Path $Directory 'server.stdout.log'
    $err = Join-Path $Directory 'server.stderr.log'
    if ((Test-Path -LiteralPath $out) -or (Test-Path -LiteralPath $err)) { throw 'Refusing existing log targets.' }
    $process = Start-Process -FilePath $contract.executable -ArgumentList $contract.arguments -WorkingDirectory ([IO.Path]::GetDirectoryName($contract.executable)) -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    $timer = [Diagnostics.Stopwatch]::StartNew()
    try {
        $script:R2LaunchObservation = [ordered]@{ pid = $process.Id; process_start_time_utc = $process.StartTime.ToUniversalTime().ToString('o') }
        $identity = Get-R2Identity $contract $process
        $started = [ordered]@{ schema = 'rcwt-r2-runtime-started/1'; started_at_utc = [DateTime]::UtcNow.ToString('o')
            intent_sha256 = $intentHash; identity = $identity; readiness = 'NOT_YET_OBSERVED' }
        $startedHash = Write-R2NewJson (Join-Path $Directory 'started.json') $started
        $ready = $false
        $healthRequests = 0
        while ($timer.Elapsed.TotalSeconds -lt $Seconds) {
            Assert-R2Identity $identity (Get-R2Identity $contract $process)
            $listeners = @(Get-R2PortConnections | Where-Object { $_.State -eq 'Listen' })
            if ($listeners.Count -gt 0) {
                Assert-R2Listener $identity.pid
                $remaining = [int][Math]::Floor(($Seconds * 1000) - $timer.Elapsed.TotalMilliseconds)
                if ($remaining -le 0) { break }
                $healthRequests++
                if ((Invoke-R2Health -TimeoutMilliseconds ([Math]::Min(2000, $remaining))) -and $timer.Elapsed.TotalSeconds -le $Seconds) { $ready = $true; break }
            }
            if ($timer.Elapsed.TotalSeconds -lt $Seconds) { Start-Sleep -Milliseconds 100 }
        }
        if (-not $ready) { throw 'Readiness deadline expired; owned process and logs preserved for explicit stop.' }
        Assert-R2Identity $identity (Get-R2Identity $contract $process)
        Assert-R2Listener $identity.pid
        Assert-R2BundleStable $Directory ([ordered]@{ contract = $contract; intent_sha256 = $intentHash; started_sha256 = $startedHash })
        $readyElapsed = $timer.Elapsed.TotalSeconds
        if ($readyElapsed -gt $Seconds) { throw 'Readiness verification exceeded its deadline.' }
        $null = Write-R2NewJson (Join-Path $Directory 'ready.json') ([ordered]@{ schema = 'rcwt-r2-runtime-ready/1'
            observed_at_utc = [DateTime]::UtcNow.ToString('o'); started_sha256 = $startedHash
            health_requests = $healthRequests; inference_calls = 0; readiness_elapsed_seconds = $readyElapsed })
        [ordered]@{ schema = 'rcwt-r2-runtime-start-result/1'; pass = $true; status = 'READY'; pid = $identity.pid
            process_start_time_utc = $identity.process_start_time_utc; health_requests = $healthRequests; inference_calls = 0
            started_receipt_sha256 = $startedHash; intent_receipt_sha256 = $intentHash; helper_sha256 = $contract.helper_sha256 }
    } finally { $process.Dispose() }
}

function Stop-R2Runtime([string]$Directory) {
    if (Test-Path -LiteralPath (Join-Path $Directory 'closed.json')) { return Get-R2Status $Directory }
    $bundle = Read-R2OwnedBundle $Directory
    $contract, $identity = $bundle.contract, $bundle.started.identity
    $observed = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + [int]$identity.pid) -ErrorAction Stop
    $termination = 'ALREADY_EXITED'
    if ($observed) {
        $process = [Diagnostics.Process]::GetProcessById([int]$identity.pid)
        try {
            Assert-R2Identity $identity (Get-R2Identity $contract $process)
            $null = Write-R2NewJson (Join-Path $Directory ('stop-intent-' + [Guid]::NewGuid().ToString('N') + '.json')) ([ordered]@{
                schema = 'rcwt-r2-runtime-stop-intent/1'; observed_at_utc = [DateTime]::UtcNow.ToString('o')
                started_sha256 = (Get-R2Sha (Join-Path $Directory 'started.json')); identity = $identity })
            Assert-R2Identity $identity (Get-R2Identity $contract $process)
            Assert-R2BundleStable $Directory $bundle
            $process.Kill() # Only the already acquired, identity-checked handle.
            if (-not $process.WaitForExit(10000)) { throw 'Owned process did not exit within the stop deadline.' }
            $termination = 'OWNED_PROCESS_KILL'
        } finally { $process.Dispose() }
    }
    if (@(Get-CimInstance Win32_Process -Filter ('ProcessId = ' + [int]$identity.pid) -ErrorAction Stop).Count -ne 0) { throw 'Process identifier still exists; closure not asserted.' }
    $logTimer = [Diagnostics.Stopwatch]::StartNew()
    $logs = $null
    do {
        try {
            $logs = [ordered]@{}
            foreach ($name in @('server.stdout.log', 'server.stderr.log')) { $logs[$name] = Get-R2StableLog (Join-Path $Directory $name) }
        } catch { $logs = $null; if ($logTimer.Elapsed.TotalSeconds -ge 10) { throw }; Start-Sleep -Milliseconds 100 }
    } while ($null -eq $logs)
    Assert-R2BundleStable $Directory $bundle
    $closed = [ordered]@{ schema = 'rcwt-r2-runtime-closed/1'; closed_at_utc = [DateTime]::UtcNow.ToString('o')
        started_sha256 = (Get-R2Sha (Join-Path $Directory 'started.json')); identity = $identity
        termination = $termination; logs = $logs; inference_calls = 0
        limitation = 'Stable logs after owned-process exit do not independently establish complete call accounting or graceful application-buffer flush.' }
    $closedHash = Write-R2NewJson (Join-Path $Directory 'closed.json') $closed
    [ordered]@{ schema = 'rcwt-r2-runtime-stop-result/1'; pass = $true; status = 'CLOSED'; pid = $identity.pid
        process_start_time_utc = $identity.process_start_time_utc; closed_receipt_sha256 = $closedHash; inference_calls = 0 }
}

if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    $ownedDirectory = $null
    try {
        if (-not $IsWindows) { throw 'This local runtime helper requires Windows.' }
        $null = Get-R2TextSha '' # Check required .NET APIs before creating anything.
        $ownedDirectory = Resolve-R2Output $OutputDirectory -MissingAllowed:($Mode -eq 'start')
        $result = switch ($Mode) {
            'start' { Start-R2Runtime $ownedDirectory $RuntimeRoot $ReadinessSeconds }
            'status' { Get-R2Status $ownedDirectory }
            'stop' { Stop-R2Runtime $ownedDirectory }
        }
        $result | ConvertTo-Json -Depth 10 -Compress
    } catch {
        $kind = $_.Exception.GetType().Name
        # Status is strictly read-only, including failure paths.
        if (($Mode -eq 'stop' -or ($Mode -eq 'start' -and $script:R2CreatedThisInvocation)) -and $ownedDirectory -and (Test-Path -LiteralPath (Join-Path $ownedDirectory 'intent.json'))) {
            try {
                $null = Write-R2NewJson (Join-Path $ownedDirectory ('error-' + $Mode + '-' + [Guid]::NewGuid().ToString('N') + '.json')) ([ordered]@{
                    schema = 'rcwt-r2-runtime-error/1'; observed_at_utc = [DateTime]::UtcNow.ToString('o'); mode = $Mode; error_type = $kind
                    launch_observation = $script:R2LaunchObservation
                    message = 'Runtime operation failed; prior receipts and logs are preserved. No unowned process was stopped.' })
            } catch { }
        }
        [ordered]@{ schema = 'rcwt-r2-runtime-status/1'; pass = $false; status = 'FAIL'; error_type = $kind
            inference_calls = 0; message = 'Owned local R2 runtime operation failed; inspect private receipts.' } | ConvertTo-Json -Compress
        exit 1
    }
}
