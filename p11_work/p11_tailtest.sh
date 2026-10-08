#!/bin/bash
cd /root/epic05_p11
tail --version 2>&1 | head -1
echo "--- test1: tail -3 single file"
tail -3 logs/stategen_room.log > /dev/null && echo OK1 || echo FAIL1
echo "--- test2: tail -n 3 single file"
tail -n 3 logs/stategen_room.log > /dev/null && echo OK2 || echo FAIL2
echo "--- test3: tail -3 multi file"
tail -3 logs/stategen_room.log logs/stategen_bicycle.log > /dev/null && echo OK3 || echo FAIL3
echo "--- test4: tail -n 3 multi file"
tail -n 3 logs/stategen_room.log logs/stategen_bicycle.log > /dev/null && echo OK4 || echo FAIL4
echo "--- test5: tail -n 8 pipeline"
tail -n 8 logs/pipeline.log > /dev/null && echo OK5 || echo FAIL5
