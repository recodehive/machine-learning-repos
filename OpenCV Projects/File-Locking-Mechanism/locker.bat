@ECHO OFF

title Folder Locker

if EXIST "Control Panel.{21EC2020-3AEA-1069-A2DD-08002B30309D}" goto UNLOCK

if NOT EXIST Locker goto MDLOCKER

:CONFIRM

echo Are you sure u want to Lock the folder(Y/N)

set/p "cho=>"

if %cho%==Y goto LOCK

if %cho%==y goto LOCK

if %cho%==n goto END

if %cho%==N goto END

echo Invalid choice.

goto CONFIRM

:LOCK

ren Locker "Control Panel.{21EC2020-3AEA-1069-A2DD-08002B30309D}"

attrib +h +s "Control Panel.{21EC2020-3AEA-1069-A2DD-08002B30309D}"

echo Folder locked

goto End

:UNLOCK

echo Enter password to Unlock folder
SET PASS=Mugundh
C:\ProgramData\Anaconda3\python.exe "%CD%//Face_Recognition_Script//Face recognition.py" %PASS%> Output
SET /p MYVAR=<Output
if %MYVAR%==%PASS% goto DONE
for /f "tokens=*" %%a in (Output) do (
  if %%a==%PASS% goto DONE
)
goto FAIL

:DONE
attrib -h -s "Control Panel.{21EC2020-3AEA-1069-A2DD-08002B30309D}"

ren "Control Panel.{21EC2020-3AEA-1069-A2DD-08002B30309D}" Locker

echo Folder Unlocked successfully
pause

goto End

:FAIL
pause
echo Invalid password

goto End

:MDLOCKER

md Locker

echo Locker created successfully

goto End

:End