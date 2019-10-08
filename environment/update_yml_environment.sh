#!/bin/bash

progress(){
  echo -n "$0: Please wait ."
  while true
  do
    sleep 3
    echo -n "."
  done
}

dobackup(){
  conda env export > environment/environment.yml    
}

progress &

MYSELF=$!

dobackup

kill $MYSELF >/dev/null 2>&1

echo -n "... done."