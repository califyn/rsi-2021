import subprocess
import os
import multiprocessing
import sys

def run_cmd(cmds, gpu):
    for cmd in cmds:
        print("running " + cmd)
        subprocess.run(cmd  + " --silent --gpu=" + str(gpu), shell=True)
        #with open('test.log', 'wb') as f:
        #    process = subprocess.Popen(cmd + " --gpu=" + str(gpu),shell=True, stdout=subprocess.PIPE)
        #    for c in iter(lambda: process.stdout.read(1), b''):
        #        sys.stdout.buffer.write(c)
        #        f.buffer.write(c)

if __name__ == "__main__":
    trials = [1, 2, 3, 4]
    #noises=[0,0.05]
    #names=["inf","20"]
    #test_noises=[0]
    long_cmd = []
    short_cmd = []

    for k in trials:
        for i in [5120,2560,1280,640]:
            long_cmd.append("cd ..; python pendulum.py --verbose --data_size=" + str(i) + " --bsz=" + str(i//10) + " --epochs=" + str(5120 // i * 100) + " --save_every=" + str(5120 // i * 20) + " --path_dir=" + str(k) + "dataset_" + str(i) + "_sup --mode=supervised")
            long_cmd.append("cd ..; python pendulum.py --verbose --data_size=" + str(i) + " --bsz=" + str(i//10) + " --epochs=" + str(5120 // i * 100) + " --save_every=" + str(5120 // i * 20) + " --path_dir=" + str(k) + "dataset_" + str(i))
    print(len(long_cmd))
    print(len(short_cmd))

    for cmds in [long_cmd]:
        jobs = []
        for i in range(3, 8):
            print(str(i))
            this_cmds = []
            j = i % 5
            while j < len(cmds):
                this_cmds.append(cmds[j])
                j += 5
            p = multiprocessing.Process(target=run_cmd, args=(this_cmds,i))
            jobs.append(p)
            print("starting " + str(i))
            p.start()
        for job in jobs:
            job.join()

    for cmds in [short_cmd]:
        run_cmd(cmds, 3)
