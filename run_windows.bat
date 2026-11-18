@echo off
cd /d "%~dp0"
python train.py --train data/train.csv --test data/test.csv --output outputs --epochs 100 --batch-size 32
pause
