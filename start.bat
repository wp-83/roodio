@echo off
:: ==============================================================================
::  Roodio - Start Script (Windows - works in CMD, PowerShell, or double-click)
::  Run: start.bat
:: ==============================================================================
echo.
echo Roodio - Starting servers via PowerShell...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1"

echo.
pause
