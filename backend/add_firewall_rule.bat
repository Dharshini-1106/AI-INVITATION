@echo off
REM ============================================================
REM Allow inbound TCP port 8000 so the phone can reach the
REM FastAPI backend over WiFi (or USB LAN).
REM
REM Run this as Administrator (right-click -> Run as administrator).
REM ============================================================
echo Adding firewall rule: InvitationSense Backend 8000 (TCP 8000)...
netsh advfirewall firewall delete rule name="InvitationSense Backend 8000" >nul 2>&1
netsh advfirewall firewall add rule name="InvitationSense Backend 8000" dir=in action=allow protocol=TCP localport=8000 profile=private,public
if %errorlevel%==0 (
  echo.
  echo SUCCESS: Firewall rule added. The phone can now reach
  echo   http://10.11.3.54:8000/api/v1
  echo.
) else (
  echo.
  echo FAILED: You must run this file as Administrator.
  echo Right-click it and choose "Run as administrator".
  echo.
)
pause
