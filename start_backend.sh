#!/bin/bash
cd /Users/gminneci/Code/Mantile
python3 -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
