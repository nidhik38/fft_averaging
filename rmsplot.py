import numpy as np
import scipy as sy
import scipy.fftpack as syfp
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import signal
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import csv file
df =pd.read_csv(r'E:\CHROME DOWNLOADS\for rms off dtaa\x1.csv',encoding='utf-8')
df=df.dropna()

t=df['t']
x=df['v']
N = np.int(np.prod(t.shape))  # length of the array
Fs = 200  # sample rate (Hz)
T = 1 / Fs
w = np.int(np.floor(Fs))  # width of the window for computing RMS
steps = np.int_(np.floor(N / w))  # Number of steps for RMS
t_RMS = np.zeros((steps, 1)) # Create array for RMS time values
x_RMS = np.zeros((steps, 1))  # Create array for RMS values
for i in range(0, steps):
    t_RMS[i] = np.mean(t[(i * w):((i + 1) * w)])
    x_RMS[i] = np.sqrt(np.mean(x[(i * w):((i + 1) * w)] ** 2))
plt.figure(1)
plt.plot(t_RMS, x_RMS,'r')
plt.xlabel('Time (seconds)')
plt.ylabel('RMS Accel (g)')
plt.title('RMS - ' + 'RMS')
#plt.grid()
plt.show()