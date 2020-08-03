# get the monotonic time offset between 2 machine
import os
import subprocess

USENAME = 'bst'
IP = '10.10.20.30'

out = subprocess.check_output("""
ssh {}@{}  <<'ENDSSH'
python timer.py
# exit
ENDSSH
""".format(USENAME, IP), shell = True)

print(out)