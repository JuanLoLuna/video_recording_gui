<#
.SYNOPSIS
  Read-only USB inventory for the recording laptop (Windows).

.DESCRIPTION
  Answers: which USB host controllers exist, what generation each is, and which
  controller every FLIR camera hangs off. Two cameras on the SAME controller
  share that controller's bandwidth; cameras on different controllers do not.

  Run in PowerShell (no admin needed):
      powershell -ExecutionPolicy Bypass -File scripts\usb_topology.ps1

  Paste the full output back. Nothing is modified.
#>

$ErrorActionPreference = "SilentlyContinue"

function Get-ParentChain($instanceId) {
    # Walk DEVPKEY_Device_Parent up to the root; returns objects, nearest first.
    $chain = @()
    $current = $instanceId
    for ($i = 0; $i -lt 12 -and $current; $i++) {
        $parent = (Get-PnpDeviceProperty -InstanceId $current -KeyName "DEVPKEY_Device_Parent").Data
        if (-not $parent) { break }
        $dev = Get-PnpDevice -InstanceId $parent
        $chain += [pscustomobject]@{ InstanceId = $parent; Name = $dev.FriendlyName; Class = $dev.Class }
        $current = $parent
    }
    return $chain
}

Write-Host "=== Machine ===" -ForegroundColor Cyan
Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model | Format-List
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List
"RAM: {0:N1} GB" -f ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)

Write-Host "`n=== USB host controllers ===" -ForegroundColor Cyan
# xHCI = USB 3.x capable. "Enhanced" (EHCI) = USB 2.0 only.
$controllers = Get-PnpDevice -Class USB | Where-Object { $_.FriendlyName -match "Host Controller" }
$controllers | Select-Object FriendlyName, Status, InstanceId | Format-Table -AutoSize -Wrap

Write-Host "=== Root hubs ===" -ForegroundColor Cyan
Get-PnpDevice -Class USB | Where-Object { $_.FriendlyName -match "Root Hub" } |
    Select-Object FriendlyName, InstanceId | Format-Table -AutoSize -Wrap

Write-Host "=== FLIR / Teledyne cameras (USB VID 1E10) and their controller ===" -ForegroundColor Cyan
$cams = Get-PnpDevice | Where-Object { $_.InstanceId -match "VID_1E10" -and $_.Status -eq "OK" }
if (-not $cams) {
    Write-Host "No device with VID_1E10 found. Plug the cameras in, or list all USB devices below." -ForegroundColor Yellow
}
foreach ($cam in $cams) {
    $chain = Get-ParentChain $cam.InstanceId
    $ctrl = $chain | Where-Object { $_.Name -match "Host Controller" } | Select-Object -First 1
    $hubs = ($chain | Where-Object { $_.Name -match "Hub" } | ForEach-Object { $_.Name }) -join " -> "
    [pscustomobject]@{
        Camera     = $cam.FriendlyName
        InstanceId = $cam.InstanceId
        Controller = if ($ctrl) { $ctrl.Name } else { "<not found>" }
        ViaHubs    = $hubs
    } | Format-List
}

Write-Host "=== All USB devices with their negotiated speed note ===" -ForegroundColor Cyan
Write-Host "(Windows does not expose link speed via PnP; use USBTreeView or camera_inventory.py for that.)"
Get-PnpDevice -Class USB -Status OK |
    Where-Object { $_.FriendlyName -notmatch "Root Hub|Host Controller" } |
    Select-Object FriendlyName, InstanceId | Format-Table -AutoSize -Wrap

Write-Host "=== Recording drive ===" -ForegroundColor Cyan
Get-PhysicalDisk | Select-Object FriendlyName, MediaType, BusType, @{n="SizeGB";e={[math]::Round($_.Size/1GB)}} | Format-Table -AutoSize
Get-Volume | Where-Object DriveLetter | Select-Object DriveLetter, FileSystem, @{n="FreeGB";e={[math]::Round($_.SizeRemaining/1GB)}}, @{n="SizeGB";e={[math]::Round($_.Size/1GB)}} | Format-Table -AutoSize
