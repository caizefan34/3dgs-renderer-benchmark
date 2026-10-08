#!/bin/bash
cd /root/epic05_p11
nohup bash p11_final_resume.sh > logs/final_resume.log 2>&1 &
echo "FINAL_RESUME_PID=$!"
