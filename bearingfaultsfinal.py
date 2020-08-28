import pandas as pd
import numpy as np
import math 
import scipy as sy
from configparser import ConfigParser
import scipy.fftpack as syfp
from scipy import signal
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
#import pandas as pd
import os
import csv
import math 
df =pd.read_csv(r'C:\Users\ABC\OneDrive\Pictures\Screenshots\New folder\velocitybasf.csv')
df=df.dropna()
df.columns
df=df.dropna()

config_object = ConfigParser()
config_object.read("userinfo2.ini")
userinfo2 = config_object["user_info2"]
power=int(userinfo2["p"]) #power in KW 
windowsize=0
windowsize1=64
samplewindow=64

for row in range(0,len(df),samplewindow):
    df1=df.loc[windowsize:windowsize1]
    velocity = df1['v']

    if power in range(15,300):	
	    for values in velocity:
	# print(type(values))
	        if float(values)>0.28 and float(values)<1.12:
		        print("good")
	        elif float(values)>=1.12 and float(values)<2.8:
		        print("satisfactory")
	        elif float(values)>=2.8 and float(values)<=7.10:

		        print("unsatisfactory")

	        elif float(values)>=11.2 and float(values)<=45.0:

		        print("unaccepetable")
                config_object = ConfigParser()
                config_object.read("userinfo.ini")
                userinfo = config_object["user_info"]

                NB=int(userinfo["NB"]) #Number of Rolling Element or Ball
                BD=int(userinfo["BD"]) #Rolling Element or Ball Diameter
                PD=int(userinfo["PD"]) #pitch circle diameter of the bearing
                angle=int(userinfo["angle"]) #Contact Angle
                RPM=float(userinfo["RPM"])#rpm 
                shaftspeed=RPM/60 #Shaft Rotational Speed
                
				Fcir=((1/2)*(1-(BD/PD)*math.cos(angle)))
                FTF=Fcir*shaftspeed #fundamental train frequency
                print("the FTF is",FTF)
				FTFFinal=int(FTF)
                print(FTFFinal) 

                Bfir=((NB/2)*(1+(BD/PD)*math.cos(angle)))
                BPFI=Bfir*shaftspeed #ball pass frequency of inner race
                BPFIFinal=int(BPFI)
                print(BPFIFinal) 
                
				Bfor=((NB/2)*(1-(BD/PD)*math.cos(angle)))
                BPFO=Bfor*shaftspeed #ball pass frequency  of outer race
                BPFOFinal=int(BPFO)
                print(BPFOFinal) 

                Bsf=((PD/(2*BD))*(1-((BD/PD)*(math.cos(angle)))**2))
                BSF=Bsf*shaftspeed #ball spin frequency
                print("the BSF is",BSF)
                BSFFinal=int(BSF)

                df =pd.read_csv(r'C:\Users\ABC\OneDrive\Pictures\Screenshots\New folder\basfdata.csv')
                df=df.dropna()
                df.columns
                df=df.dropna()
                final_list = []
                maxamplitudes=4
                windowsize=0
                windowsize1=1024
                df1=df.loc[windowsize:windowsize1]
                Fs=float(userinfo["Fs"])
                T = 1 / Fs
                m = df1['v']
                N = len(m)
                z = signal.detrend(m)
                yf=fft(z)
                xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
                x =xf 
                y = abs(yf[0:N//2])
                plt.plot(x,y)
                plt.show()
                l1 = y
                l2 = [int(i) for i in l1]
                l3= x
                l4 = [int(i) for i in l3]
                a=np.array(l2)
                index_list= np.argpartition(a, -4)[-4:]#calculates indexes of four maximum peak values from the array
                print("The index of highest peak is",index_list)
                for i in range(0, maxamplitudes): #loop is used to calculate 4 maximum amplitudes of fft plot
                    max1 = 0
         
                    for j in range(len(l2)):     
                        if l2[j] > max1:
                            max1 = l2[j]
                    l2.remove(max1)
                    final_list.append(max1)#4 maximum amplitudes of fft plot is appended into final_list
                    print("max amplitude values are",final_list)
                res_list = [l4[i] for i in index_list] #for the index list it defines frqn corresponding to index
                print ("fft frqn corresponding to highest peak is: " + str(res_list)) 
                for element in res_list:#for element present in res_list(freqn) 
                    if element in range(BPFIFinal-8,BPFIFinal+8):
                        print("their is faut in innerrace of bearing")
                    elif element in range(BPFOFinal-8,BPFOFinal+8):
                        print("their is faut in outerrace of bearing")
                    elif element in range(BSFFinal-8,BSFFinal+8):
                        print("their is faut in ball/rollers of bearing")
					elif element in range(FTFFinal-8,FTFFinal+8):
                        print("their is faut in cage of bearing")
                    else:
                        pass  
	        else:
		        print("motor is off")

    windowsize = windowsize + samplewindow
    windowsize1=windowsize1+samplewindow   