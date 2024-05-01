# CelFDrive

## Conda env installation

conda env create -f environment-gpu-windows.yml --prefix E:\Scott\Software\Anaconda\Envs\CellToolsGPU
conda env create -n CellToolsGPU -f E:\Scott\GitHub\CellTools\environment-gpu-windows.yml
conda config --describe E:\Scott\Software\Anaconda\CellToolsGPU E:\Scott\Software\Anaconda\CellToolsGPU
conda config --add envs_dirs E:\Scott\Software\Anaconda\Envs
conda config --add pkgs_dirs E:\Scott\Software\Anaconda\Packages

### Improvements

Selector needs a save progress
clicker needs better checks that coords are within image bounds
selector image none safety check

maybe an overall try/catch that skips a set and marks as blurry if corruped
