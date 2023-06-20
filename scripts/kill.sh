sudo lsof -w /dev/accel0 | awk 'NR>1 {print $2}' | xargs sudo kill -9
