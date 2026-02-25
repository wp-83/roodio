@echo off
:: ==============================================================================
::  Roodio - Setup Script (Windows - works in CMD, PowerShell, or double-click)
::  Run: setup.bat
:: ==============================================================================
echo.
echo Roodio - Running setup via PowerShell...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

echo.
pause
