set -x
set -e
ps aux | grep $1 | awk '{printf $2" "}' | xargs kill -9
