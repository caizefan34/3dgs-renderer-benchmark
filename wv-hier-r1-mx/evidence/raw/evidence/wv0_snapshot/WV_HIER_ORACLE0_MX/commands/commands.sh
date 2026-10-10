#!/bin/bash
# WV-HIER-ORACLE0-MX command log
# Campaign driver (9 states across GPUs 4-7):
cd /dev/shm && bash /dev/shm/_wv0_campaign_driver.sh
# Per-state invocation:
#   python _wv0_run.py _wv0_campaign_state.py <gpu> <out.json> <state_id> 540p 8 <out.pt> 0
# Analysis:
#   python _wv0_aggregate.py /dev/shm/wv0_out
#   python _wv0_gate.py /dev/shm/wv0_out
#   python _wv0_verdict.py /dev/shm/wv0_out
#   python _wv0_figures.py /dev/shm/wv0_out
#   python _wv0_report.py /dev/shm/wv0_out
