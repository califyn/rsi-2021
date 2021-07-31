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
    trials = [0, 1, 2, 3, 4]#, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    #noises=[0,0.05]
    #names=["inf","20"]
    #test_noises=[0]
    long_cmd = []
    short_cmd = []

    for k in trials:
        #for i in [0.9,0.75,0.6,0.5,0.45]:
            #long_cmd.append("cd ..; python pendulum.py --verbose --crop=" + str(i) + " --crop_c=" + str(1-i/2) + "," + str(1-i/2) + " --path_dir=" + str(k) + "crop_" + str(int(i*1000)) + "_sup --mode=supervised");
            #long_cmd.append("cd ..; python pendulum.py --verbose --crop=" + str(i) + " --crop_c=" + str(1-i/2) + "," + str(1-i/2) + " --path_dir=" + str(k) + "crop_" + str(int(i*1000)));
            #short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "crop_" + str(int(i*1000)));
            #short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "crop_" + str(int(i*1000)) + "_sup");

        #for i in [0,4,8]:
            #for j in [0.25,0.5,1,2,4]:
                #long_cmd.append("cd ..; python pendulum.py --verbose --t_window=" + str(i) + "," + str(i+j) + " --path_dir=" + str(k) + "twindow_" + str(int(i*1000+j*10)) + "_sup --mode=supervised");
                #long_cmd.append("cd ..; python pendulum.py --verbose --t_window=" + str(i) + "," + str(i+j) + " --path_dir=" + str(k) + "twindow_" + str(int(i*1000+j*10)));
                #short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "twindow_" + str(int(i*1000+j*10)));
                #short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "twindow_" + str(int(i*1000+j*10)) + "_sup");

        #for i in [0.25,0.5,1,2,4]:
            #long_cmd.append("cd ..; python pendulum.py --verbose --t_range=" + str(i) + " --path_dir=" + str(k) + "trange_" + str(int(i*1000)) + "_sup --mode=supervised");
            #long_cmd.append("cd ..; python pendulum.py --verbose --t_range=" + str(i) + " --path_dir=" + str(k) + "trange_" + str(int(i*1000)));
            #short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "trange_" + str(int(i*1000)));
            #short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "trange_" + str(int(i*1000)) + "_sup");

        for i in [0.1,0.2,0.3,0.4,0.6,0.8]:
            long_cmd.append("cd ..; python pendulum.py --verbose --gaps=4," + str(i) + " --path_dir=" + str(k) + "midgapm_" + str(int(i*1000)) + "_sup --mode=supervised");
            long_cmd.append("cd ..; python pendulum.py --verbose --gaps=4," + str(i) + " --path_dir=" + str(k) + "midgapm_" + str(int(i*1000)));
            short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "midgapm_" + str(int(i*1000)));
            short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "midgapm_" + str(int(i*1000)) + "_sup");
            
        for i in [1,2,3,4,6,8,10]:
            long_cmd.append("cd ..; python pendulum.py --verbose --gaps=" + str(i) + ",0.4 --path_dir=" + str(k) + "spreadgapm_" + str(int(i)) + "_sup --mode=supervised");
            long_cmd.append("cd ..; python pendulum.py --verbose --gaps=" + str(i) + ",0.4 --path_dir=" + str(k) + "spreadgapm_" + str(int(i)));
            short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "spreadgapm_" + str(int(i)));
            short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "spreadgapm_" + str(int(i)) + "_sup");

        #for i in [0.1,0.2,0.25,0.3,0.35,0.4,0.425,0.45,0.475]:
            #long_cmd.append("cd ..; python pendulum.py --verbose --mink=" + str(i) + " --maxk=" + str(1 - i) + " --path_dir=" + str(k) + "extra_" + str(int(i*1000)) + "_sup --mode=supervised");
            #long_cmd.append("cd ..; python pendulum.py --verbose --mink=" + str(i) + " --maxk=" + str(1 - i) + " --path_dir=" + str(k) + "extra_" + str(int(i*1000)));
            #short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "extra_" + str(int(i*1000)));
            #short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "extra_" + str(int(i*1000)) + "_sup");

        #for i in [0,0.1,0.2,0.3,0.35,0.4,0.45]:
            #long_cmd.append("cd ..; python pendulum.py --verbose --gaps=3,0.5 --mink=" + str(i) + " --maxk=" + str(1 - i) + " --path_dir=" + str(k) + "extragap_" + str(int(i*1000)) + "_sup --mode=supervised");
            #long_cmd.append("cd ..; python pendulum.py --verbose --gaps=3,0.5 --mink=" + str(i) + " --maxk=" + str(1 - i) + " --path_dir=" + str(k) + "extragap_" + str(int(i*1000)));
            #short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "extragap_" + str(int(i*1000)));
            #short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + str(k) + "extragap_" + str(int(i*1000)) + "_sup");

    print(long_cmd)
    print(short_cmd)

    input("continue...")
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
