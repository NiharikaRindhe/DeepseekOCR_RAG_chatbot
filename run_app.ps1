Write-Host "Checking dependencies..."
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "Starting Streamlit App..."
    streamlit run app.py
} else {
    Write-Host "Dependency installation failed."
}
