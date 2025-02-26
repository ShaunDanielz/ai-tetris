# Get all Python files in the current directory that contain "-"
Get-ChildItem -Path . -Filter "*.py" | Where-Object { $_.Name -match "-" } | ForEach-Object {
    $newName = $_.Name -replace "-", "_"
    Rename-Item -Path $_.FullName -NewName $newName
}

Write-Output "Filenames have been updated to use underscores instead of hyphens."
