#!/bin/bash
# Launch the P11 resume pipeline detached with correct logging.
cd /root/epic05_p11
nohup bash p11_resume.sh L1 > logs/resume.log 2>&1 &
echo "RESUME_PID=$!"
