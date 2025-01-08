import numpy as np
import pyaudio as pa
import struct
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# cmap=plt.get_cmap('gist_ncar')
cmap=plt.get_cmap('gist_rainbow')

CHUNK = 1024 * 2
FORMAT = pa.paInt16
CHANNELS = 1
RATE = 44100  # in Hz

p = pa.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK,
)

def colr(y_fft,x_fft) :
    x_norm=(x_fft/(x_fft[-1]))
    rgba=cmap(x_norm[100:2050])
     
    
    # rgba_a= np.average(rgba, axis=0)
    rgba_a= np.average(rgba, axis=0, weights=y_fft[100:2050]*1000)
    
    for i in range(4):
        if rgba_a[i]>1 :
            rgba_a[i]=1.
    
    return rgba_a


fig, (ax, ax1, ax2) = plt.subplots(3)
x_fft = np.linspace(0, RATE, CHUNK)
x = np.arange(0, 2 * CHUNK, 2)
x_c=x
ax2.set_facecolor("violet")
(line,) = ax.plot(x, np.random.rand(CHUNK), "r")
# (line_fft,) = ax1.plot(x_fft, np.random.rand(CHUNK), "b")
(line_fft,) = ax1.semilogx(x_fft, np.random.rand(CHUNK), "b")
ax.set_ylim(-32000, 32000)
ax.set_xlim = (0, CHUNK)
ax1.set_xlim(20, RATE / 2)
ax1.set_ylim(0, 1)
fig.show()


while 1:
    data = stream.read(CHUNK)
    dataInt = struct.unpack(str(CHUNK) + "h", data)
    line.set_ydata(dataInt)
    y_fft = np.abs(np.fft.fft(dataInt)) * 2 / (11000 * CHUNK)
    line_fft.set_ydata(y_fft)
    ax2.set_facecolor(colr(y_fft,x_fft))
    # ax2.set_facecolor(cmap(0.8))
    fig.canvas.draw()
    fig.canvas.flush_events()
