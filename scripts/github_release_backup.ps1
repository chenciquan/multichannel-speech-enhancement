param([string]$Tag='research-snapshot-g35-20261113')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
$cacheRoot=Join-Path $taskRoot '.cache/github_upload_20261113'
$env:GIT_TERMINAL_PROMPT='0';$env:GCM_INTERACTIVE='Never'
$credentialReply="protocol=https`nhost=github.com`n`n" | git credential fill
if($LASTEXITCODE -ne 0) { throw 'Git credentials unavailable' }
$fields=@{}
foreach($line in $credentialReply) { $p=$line.IndexOf('=');if($p -gt 0) {$fields[$line.Substring(0,$p)]=$line.Substring($p+1)} }
if(!$fields['password']) { throw 'Credential secret missing' }
$headers=@{Authorization=('Bearer '+$fields['password']);Accept='application/vnd.github+json';'X-GitHub-Api-Version'='2022-11-28';'User-Agent'='mcse-research-backup'}
$api='https://api.github.com/repos/chenciquan/multichannel-speech-enhancement'
$profile=Invoke-RestMethod https://api.github.com/user -Headers $headers
if($profile.login -ne 'chenciquan') { throw 'Unexpected account' }
$repo=Invoke-RestMethod $api -Headers $headers
if(!$repo.private -or !$repo.permissions.push) { throw 'Requires intended private repository and push permission' }
$targetCommit=(& git -C $taskRoot rev-parse HEAD).Trim()
$release=$null
$priorStatePath=Join-Path $cacheRoot 'upload_state.json'
if(Test-Path -LiteralPath $priorStatePath) {
    $prior=Get-Content -LiteralPath $priorStatePath -Raw -Encoding UTF8|ConvertFrom-Json
    if($prior.tag -ne $Tag) {throw 'Existing upload state uses another tag'}
    $release=Invoke-RestMethod "$api/releases/$($prior.release_id)" -Headers $headers
}
if(!$release) {
    try { $release=Invoke-RestMethod "$api/releases/tags/$Tag" -Headers $headers }
    catch { if([int]$_.Exception.Response.StatusCode -ne 404) {throw} }
}
if(!$release) {
    $body=@{tag_name=$Tag;target_commitish=$targetCommit;name='Research snapshot through G35 (2026-11-13)';draft=$true;prerelease=$false;body='Lossless research archive. Code and readable evidence are in the repository. ZIP parts restore data/, artifacts/, third_party/. Environment and caches excluded. SHA256 manifest included. No experiments rerun.'}|ConvertTo-Json
    $release=Invoke-RestMethod "$api/releases" -Method Post -Headers $headers -ContentType application/json -Body $body
}
$state=[ordered]@{status='uploading';release_id=$release.id;tag=$Tag;repository=$repo.full_name;private=$true;target_commit=$targetCommit;assets=@()}
function Save-State { $state|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $cacheRoot 'upload_state.json') -Encoding UTF8 }
Save-State
Add-Type -AssemblyName System.Net.Http
$client=[System.Net.Http.HttpClient]::new()
$client.Timeout=[TimeSpan]::FromMinutes(90)
$client.DefaultRequestHeaders.Authorization=[System.Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer',$fields['password'])
$client.DefaultRequestHeaders.UserAgent.ParseAdd('mcse-research-backup')
$client.DefaultRequestHeaders.Accept.ParseAdd('application/vnd.github+json')
$uploadBase=($release.upload_url -split '\{')[0]
$uploaded=@{}
foreach($a in (Invoke-RestMethod "$api/releases/$($release.id)/assets?per_page=100" -Headers $headers)) {
    if($null -ne $a -and $a.name) {$uploaded[$a.name]=$a}
}
function Upload-File([string]$filePath,[string]$expectedHash) {
    $file=Get-Item -LiteralPath $filePath
    if($uploaded.ContainsKey($file.Name)) {
        $asset=$uploaded[$file.Name]
        if($asset.size -ne $file.Length -or ($asset.digest -and $asset.digest -ne ('sha256:'+$expectedHash))) { throw ('Existing asset differs: '+$file.Name) }
    } else {
        Write-Output ('Uploading '+$file.Name+' ('+$file.Length+' bytes)')
        $stream=[System.IO.File]::OpenRead($file.FullName)
        $content=[System.Net.Http.StreamContent]::new($stream,1048576)
        $content.Headers.ContentType=[System.Net.Http.Headers.MediaTypeHeaderValue]::new('application/octet-stream')
        $content.Headers.ContentLength=$file.Length
        try {
            $url=$uploadBase+'?name='+[uri]::EscapeDataString($file.Name)
            $response=$client.PostAsync($url,$content).GetAwaiter().GetResult()
            $responseBody=$response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            if(!$response.IsSuccessStatusCode) { throw ('GitHub upload HTTP '+[int]$response.StatusCode+': '+$responseBody) }
            $asset=$responseBody|ConvertFrom-Json
        } finally { $content.Dispose();$stream.Dispose() }
        if($asset.size -ne $file.Length -or ($asset.digest -and $asset.digest -ne ('sha256:'+$expectedHash))) { throw ('Uploaded size/digest differs: '+$file.Name) }
        $uploaded[$file.Name]=$asset
    }
    $state.assets+=@{name=$file.Name;bytes=$file.Length;sha256=$expectedHash;id=$asset.id;url=$asset.browser_download_url;provider_digest=$asset.digest}
    Save-State
    Write-Output ('Verified '+$file.Name)
}
try {
    $processed=@{}
    while($true) {
        $progressPath=Join-Path $cacheRoot 'progress.json'
        if(Test-Path -LiteralPath $progressPath) {
            $progress=Get-Content -LiteralPath $progressPath -Raw -Encoding UTF8|ConvertFrom-Json
            $ready=@(Get-ChildItem -LiteralPath $cacheRoot -Filter 'research-part-*.zip'|Sort-Object Name|Select-Object -First $progress.completed_parts)
            foreach($file in $ready) {
                if(!$processed.ContainsKey($file.Name)) {
                    $fileHash=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                    Upload-File $file.FullName $fileHash
                    $processed[$file.Name]=$true
                }
            }
        }
        if(Test-Path -LiteralPath (Join-Path $cacheRoot 'archive_complete.json')) {
            $done=Get-Content -LiteralPath (Join-Path $cacheRoot 'archive_complete.json') -Raw -Encoding UTF8|ConvertFrom-Json
            if($processed.Count -eq $done.parts) {break}
        }
        Start-Sleep -Seconds 10
    }
    foreach($name in @('research-file-manifest.json','SHA256SUMS.txt')) {
        $filePath=Join-Path $cacheRoot $name
        Upload-File $filePath ((Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLowerInvariant())
    }
    $remoteAssets=@(Invoke-RestMethod "$api/releases/$($release.id)/assets?per_page=100" -Headers $headers)
    foreach($a in $state.assets) {
        $found=@($remoteAssets|Where-Object {$_.name -eq $a.name})
        if($found.Count -ne 1 -or $found[0].size -ne $a.bytes) {throw ('Remote asset verification failed: '+$a.name)}
    }
    $release=Invoke-RestMethod "$api/releases/$($release.id)" -Method Patch -Headers $headers -ContentType application/json -Body (@{draft=$false}|ConvertTo-Json)
    $state.status='complete';$state.release_url=$release.html_url;Save-State
    Write-Output ('Release complete: '+$release.html_url)
} catch {
    $state.status='failed';$state.error=$_.Exception.Message;Save-State
    throw
} finally {$client.Dispose()}
