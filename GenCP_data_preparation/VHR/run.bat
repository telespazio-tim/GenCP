@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: run.bat  –  Tile Generator
:: ============================================================

:: ============================================================
:: MANUAL SETTINGS REQUIRED BEFORE RUNNING
:: ============================================================
::
:: The following paths must be manually set before running this file:
::
::   QGIS_DIR
::       Main installation directory of QGIS.
::       Example:
::       C:\Program Files\QGIS 3.38.2
::
::   MY_SCRIPT
::       Full path of the Python script to be executed with QGIS Python.
::       Example:
::       C:\...\osm_style_processor.py
::
::   DEFAULT_SHP_DIR
::       Default folder containing the input shapefiles.
::       This value is used if --shp_dir is not provided from the command line.
::
::   DEFAULT_QML_DIR
::       Default folder containing QGIS style files (.qml).
::       This value is used if --qml_dir is not provided from the command line.
::
::   DEFAULT_OUTPUT
::       Default folder where the generated tiles will be saved.
::       This value is used if --output is not provided from the command line.
::
::   DEFAULT_TILE_SIZE
::       Default tile size in pixels.
::
::   DEFAULT_DPI
::       Default output DPI value.
::
:: Raster input is not defined as a default value in this script.
:: Therefore, either --raster_dir or --raster_files must be provided
:: when running the batch file.
:: ===============================================================

:: :: ============================================================
:: Usage examples:
::
:: 1) Using a raster folder:
:: All raster files in the given folder will be processed.
::
:: run.bat --raster_dir "C:\data\rasters"
::
:: 2) Using individual raster files:
:: Only the specified raster files will be processed.
::
:: run.bat --raster_files "C:\data\r1.tif" "C:\data\r2.tif"
::
:: 3) Using all parameters:
:: This example explicitly defines the shapefile folder, QML style
:: folder, raster folder, output folder, tile size, DPI, and excluded
:: categories.
::
:: run.bat --shp_dir "C:...\shp" --qml_dir "C:...\styles" ^
:: --raster_dir "C:...\rasters" --output "C:...\tiles" ^
:: --tile_size 512 --dpi 150 --exclude places pofw
::
:: Notes:
:: - --raster_dir is used when all raster files in a folder will be processed.
:: - --raster_files is used when only selected raster files will be processed.
:: - --shp_dir, --qml_dir, and --output can be omitted only if
:: DEFAULT_SHP_DIR, DEFAULT_QML_DIR, and DEFAULT_OUTPUT are defined below.
:: - Either --raster_dir or --raster_files must be provided.
:: ============================================================

:: ── QGIS installation directory ────────────────────────────────────────
set "QGIS_DIR=C:\Program Files\QGIS 3.44.11"

:: ── Python script directory ───────────────────────────────────────
set "MY_SCRIPT=.\osm_style_processor.py"

:: ── Default parameter values ───────────────────────────
set "DEFAULT_TILE_SIZE=256"
set "DEFAULT_DPI=96"

:: ── Parse command-line arguments ──────────────────────
set "ARG_SHP_DIR="
set "ARG_QML_DIR="
set "ARG_OUTPUT="
set "ARG_RASTER_DIR="
set "ARG_RASTER_FILES="
set "ARG_TILE_SIZE="
set "ARG_DPI="
set "ARG_EXCLUDE="
set "COLLECTING_FILES=0"
set "COLLECTING_EXCLUDE=0"

:parse_loop
if "%~1"=="" goto parse_done

if /i "%~1"=="--shp_dir"      ( set "ARG_SHP_DIR=%~2"    & set COLLECTING_FILES=0 & set COLLECTING_EXCLUDE=0 & shift & shift & goto parse_loop )
if /i "%~1"=="--qml_dir"      ( set "ARG_QML_DIR=%~2"    & set COLLECTING_FILES=0 & set COLLECTING_EXCLUDE=0 & shift & shift & goto parse_loop )
if /i "%~1"=="--output"       ( set "ARG_OUTPUT=%~2"      & set COLLECTING_FILES=0 & set COLLECTING_EXCLUDE=0 & shift & shift & goto parse_loop )
if /i "%~1"=="--raster_dir"   ( set "ARG_RASTER_DIR=%~2" & set COLLECTING_FILES=0 & set COLLECTING_EXCLUDE=0 & shift & shift & goto parse_loop )
if /i "%~1"=="--tile_size"    ( set "ARG_TILE_SIZE=%~2"   & set COLLECTING_FILES=0 & set COLLECTING_EXCLUDE=0 & shift & shift & goto parse_loop )
if /i "%~1"=="--dpi"          ( set "ARG_DPI=%~2"         & set COLLECTING_FILES=0 & set COLLECTING_EXCLUDE=0 & shift & shift & goto parse_loop )

if /i "%~1"=="--raster_files" ( set COLLECTING_FILES=1    & set COLLECTING_EXCLUDE=0 & shift & goto parse_loop )
if /i "%~1"=="--exclude"      ( set COLLECTING_FILES=0    & set COLLECTING_EXCLUDE=1 & shift & goto parse_loop )

:: Collect multiple values for raster_files and exclude
if "!COLLECTING_FILES!"=="1" (
    if "!ARG_RASTER_FILES!"=="" (
        set "ARG_RASTER_FILES=%~1"
    ) else (
        set "ARG_RASTER_FILES=!ARG_RASTER_FILES! %~1"
    )
    shift & goto parse_loop
)
if "!COLLECTING_EXCLUDE!"=="1" (
    if "!ARG_EXCLUDE!"=="" (
        set "ARG_EXCLUDE=%~1"
    ) else (
        set "ARG_EXCLUDE=!ARG_EXCLUDE! %~1"
    )
    shift & goto parse_loop
)

shift & goto parse_loop
:parse_done

:: ── Apply default values ─────────────────────────────────────
if "!ARG_SHP_DIR!"==""   set "ARG_SHP_DIR=!DEFAULT_SHP_DIR!"
if "!ARG_QML_DIR!"==""   set "ARG_QML_DIR=!DEFAULT_QML_DIR!"
if "!ARG_OUTPUT!"==""    set "ARG_OUTPUT=!DEFAULT_OUTPUT!"
if "!ARG_TILE_SIZE!"=="" set "ARG_TILE_SIZE=!DEFAULT_TILE_SIZE!"
if "!ARG_DPI!"==""       set "ARG_DPI=!DEFAULT_DPI!"

:: ── Check raster input source ──────────────────────────────────
if "!ARG_RASTER_DIR!"=="" if "!ARG_RASTER_FILES!"=="" (
    echo [ERROR] Either --raster_dir or --raster_files must be provided!
    echo.
    echo Examples:
    echo   run.bat --raster_dir "C:\data\rasters"
    echo   run.bat --raster_files "C:\data\r1.tif" "C:\data\r2.tif"
    pause
    exit /b 1
)

:: ── Check and load the QGIS environment ────────────────────────────────
if not exist "%QGIS_DIR%\bin\o4w_env.bat" (
    echo [ERROR] QGIS main folder was not found: "%QGIS_DIR%"
    pause
    exit /b 1
)
call "%QGIS_DIR%\bin\o4w_env.bat"

set "PY_BAT="
if exist "%QGIS_DIR%\bin\python-qgis-ltr.bat" (
    set "PY_BAT=%QGIS_DIR%\bin\python-qgis-ltr.bat"
    echo [INFO] LTR version is being used.
) else if exist "%QGIS_DIR%\bin\python-qgis.bat" (
    set "PY_BAT=%QGIS_DIR%\bin\python-qgis.bat"
    echo [INFO] Standard QGIS version is being used.
)

if "!PY_BAT!"=="" (
    echo [ERROR] python-qgis.bat or python-qgis-ltr.bat was not found!
    pause
    exit /b 1
)

:: ── Create the Python argument string ──────────────────────────
set "PY_ARGS=--shp_dir "!ARG_SHP_DIR!" --qml_dir "!ARG_QML_DIR!" --output "!ARG_OUTPUT!" --tile_size !ARG_TILE_SIZE! --dpi !ARG_DPI!"

if not "!ARG_RASTER_DIR!"==""   set "PY_ARGS=!PY_ARGS! --raster_dir "!ARG_RASTER_DIR!""
if not "!ARG_RASTER_FILES!"=="" set "PY_ARGS=!PY_ARGS! --raster_files !ARG_RASTER_FILES!"
if not "!ARG_EXCLUDE!"==""      set "PY_ARGS=!PY_ARGS! --exclude !ARG_EXCLUDE!"

:: ── Summary ─────────────────────────────────────────────────────
echo.
echo [PARAMETRELER]
echo   shp_dir    : !ARG_SHP_DIR!
echo   qml_dir    : !ARG_QML_DIR!
echo   output     : !ARG_OUTPUT!
if not "!ARG_RASTER_DIR!"==""   echo   raster_dir : !ARG_RASTER_DIR!
if not "!ARG_RASTER_FILES!"=="" echo   raster_files: !ARG_RASTER_FILES!
echo   tile_size  : !ARG_TILE_SIZE!px
echo   dpi        : !ARG_DPI!
if not "!ARG_EXCLUDE!"==""      echo   exclude    : !ARG_EXCLUDE!
echo.

:: ── Run ─────────────────────────────────────────────────
echo [PROCESS] Starting...
call "!PY_BAT!" "%MY_SCRIPT%" !PY_ARGS!

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Script finished with an error. Kod: %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo [DONE] Process completed successfully.
exit /b 0
