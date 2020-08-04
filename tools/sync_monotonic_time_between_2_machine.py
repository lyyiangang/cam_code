# get the monotonic time offset between 2 machine. we now the monotonic time change after reboot. and it's different between different machines.
# this script try to figure the monotonic time difference between 2 machine.
# 1. run benchmark script on target machine via ssh
# 2. run the same benchmark script on local machine
# 3. compare the real time and mono time on both machine, and find the diff between the 2 mono time
import os
import subprocess
import threading
import paramiko
import numpy as np
import ipdb
import re

USERNAME = ''
IP = '10.10.12.128'
PASSWORD=''

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

TIMER_STR = """
import time
for ii in range(1000000):
    print(time.time(), ' ', time.monotonic())
"""
np_cur_machine_outs = None
lock_np_cur_machine_outs = threading.Lock()
def local_time_thread_fun():
    global np_cur_machine_outs
    timer_py_file = 'timer.py'
    with open(timer_py_file, 'w') as fid:
        fid.write(TIMER_STR)
    cur_machine_outs = subprocess.check_output('python {}'.format(timer_py_file), shell = True)
    cur_machine_outs = cur_machine_outs.decode('utf-8').split('\n')
    results = []
    for line in cur_machine_outs:
        vals = line.split(' ')
        vals = list(filter(lambda v: v != '', vals))
        results += vals
    lock_np_cur_machine_outs.acquire()
    np_cur_machine_outs = np.array(results).reshape(-1, 2).astype(np.float64)
    lock_np_cur_machine_outs.release()

thread_local_time = threading.Thread(target= local_time_thread_fun)
thread_local_time.start()
# thread_local_time.join()

client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.WarningPolicy())
client.connect(IP, username = USERNAME, password = PASSWORD)
stdin, remote_outs, stderr = client.exec_command('echo "{}" > timer.py'.format(TIMER_STR))
stdin, remote_outs, stderr = client.exec_command('python3 timer.py')
remote_outs = [item.strip('\n') for item in remote_outs]
np_remote_outs = []
for line in remote_outs:
    vals = line.split(' ')
    vals = list(filter(lambda v: v != '', vals))
    np_remote_outs += vals
np_remote_outs = np.array(np_remote_outs).reshape(-1, 2).astype(np.float64)
client.close()
while True:
    if thread_local_time.isAlive():
        print(f'waiting for local time thread stop')
        continue
    else:
        break

assert len(np_remote_outs) == len(np_cur_machine_outs), 'count of outputs are not equal to each other. {} vs {}'.format(len(np_remote_outs), len(np_cur_machine_outs))
print(f'remote real time in range {np.min(np_remote_outs[:, 0])}, {np.max(np_remote_outs[:, 0])}')
print(f'local real time in range {np.min(np_cur_machine_outs[:, 0])}, {np.max(np_cur_machine_outs[:, 0])}')
n_items = len(np_cur_machine_outs)
local_realtime, local_mono = np_cur_machine_outs[n_items // 2]
# ipdb.set_trace()
idx = find_nearest(np_remote_outs[:, 0], local_realtime)
monotonic_latency = np_remote_outs[idx, 1] - local_mono
realtime_latency = np_remote_outs[idx, 0] - local_realtime
print('find remote index {}, mono diff(remote - local):{}, realtime diff:{}, based on local time {}'.format(idx, monotonic_latency, realtime_latency, local_realtime))
