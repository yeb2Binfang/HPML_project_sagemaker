import os
from subprocess import Popen, PIPE

def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float('nan')
    return number


def getUtilization():
    nvidia_smi = "nvidia-smi"
    p = Popen([nvidia_smi,
               "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu",
               "--format=csv,noheader,nounits"], stdout=PIPE)
    stdout, stderror = p.communicate()
    output = stdout.decode('UTF-8')

    lines = output.split(os.linesep)

    numDevices = len(lines) - 1
    utilization = 0.0
    for g in range(numDevices):
        line = lines[g]
        # print(line)
        vals = line.split(', ')
        gpuUtil = safeFloatCast(vals[2]) / 100
        utilization += gpuUtil
    return gpuUtil / numDevices