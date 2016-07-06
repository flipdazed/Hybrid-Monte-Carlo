# This is a quick bash script to run a load of files
# just remove the echo and make it a direct call to python
# the &disown makes the operations parallel

clear; for i in results/*.py; do echo "python" $i " &disown"; done
