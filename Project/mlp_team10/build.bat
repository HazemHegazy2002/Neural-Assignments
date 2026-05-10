@echo off
echo ============================================
echo  MLP Team10 - Build EXE
echo ============================================

echo [1/3] Installing PyInstaller...
pip install pyinstaller --quiet

echo [2/3] Building executable...
pyinstaller --onefile --noconsole --name "MLP_Team10" ^
  --add-data "index.html;." ^
  --add-data "css;css" ^
  --add-data "js;js" ^
  launcher.py

echo [3/3] Done!
echo.
echo Your EXE is in the "dist" folder: dist\MLP_Team10.exe
echo Double-click it to launch the MLP website in your browser.
pause
