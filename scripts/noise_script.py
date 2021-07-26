import subprocess
import os
import multiprocessing
import sys

def run_cmd(cmds, gpu):
    for cmd in cmds:
        print("running " + cmd)
        subprocess.run(cmd + " --silent --gpu=" + str(gpu), shell=True)
        #with open('test.log', 'wb') as f:
        #    process = subprocess.Popen(cmd + " --gpu=" + str(gpu),shell=True, stdout=subprocess.PIPE)
        #    for c in iter(lambda: process.stdout.read(1), b''):
        #        sys.stdout.buffer.write(c)
        #        f.buffer.write(c)

if __name__ == "__main__":
    noises = [0,0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.125,0.2,0.25]
    names = ["inf", "1000", "400", "200", "100", "40", "20", "10","8","5","4"]
    #test_noises = [0,0.01,0.125,0.25]
    test_noises = [0]
    trials = [0, 1, 2, 3, 4]
    #noises=[0,0.05]
    #names=["inf","20"]
    #test_noises=[0]
    long_cmd = []
    short_cmd = []

    for k in trials:
        for i in range(0, len(noises)):
            #long_cmd.append("python pendulum.py --verbose --both_noise --noise=" + str(noises[i]) + " --path_dir=" + str(k) + "gnoise_" + names[i] + "_sup --mode=supervised");
            #long_cmd.append("python pendulum.py --verbose --both_noise --noise=" + str(noises[i]) + " --path_dir=" + str(k) + "gnoise_" + names[i]);
            #long_cmd.append("python pendulum.py --verbose --noise=" + str(noises[i]) + " --path_dir=" + str(k) + "nnoise_" + names[i] + "_sup --mode=supervised");
            #long_cmd.append("python pendulum.py --verbose --noise=" + str(noises[i]) + " --path_dir=" + str(k) + "nnoise_" + names[i]);
            long_cmd.append("cd ..; python pendulum.py --verbose --gnoise=" + str(noises[i]) + " --nnoise=0 --path_dir=" + str(k) + "inoise_" + names[i] + "_sup --mode=supervised");
            long_cmd.append("cd ..; python pendulum.py --verbose --gnoise=" + str(noises[i]) + " --nnoise=0 --path_dir=" + str(k) + "inoise_" + names[i]);
            short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "inoise_" + names[i]);
            short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "inoise_" + names[i] + "_sup");
            #for j in range(0, len(test_noises)):
                #short_cmd.append("python pendulum.py --verbose --mode=analysis --noise=" + str(test_noises[j]) + " --path_dir=" + str(k) + "gnoise_" + names[i] + "_sup");
                #short_cmd.append("python pendulum.py --verbose --mode=analysis --noise=" + str(test_noises[j]) + " --path_dir=" + str(k) + "gnoise_" + names[i]);
                #short_cmd.append("python pendulum.py --verbose --mode=analysis --noise=" + str(test_noises[j]) + " --path_dir=" + str(k) + "nnoise_" + names[i] + "_sup");
                #short_cmd.append("python pendulum.py --verbose --mode=analysis --noise=" + str(test_noises[j]) + " --path_dir=" + str(k) + "nnoise_" + names[i]);

    print(len(long_cmd))
    print(len(short_cmd))

    """for cmds in [long_cmd]:
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
            job.join()"""

    for cmds in [short_cmd]:
        run_cmd(cmds, 3)
