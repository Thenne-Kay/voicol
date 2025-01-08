import numpy as np
import pyaudio as pa
import struct
import matplotlib.pyplot as plt
import matplotlib.colors as colors


cmap = plt.get_cmap("gist_rainbow")


def filtre(arr,lim) :
    mask=arr>lim
    return arr*mask


def colr(y_fft, x_fft):
    # x_norm = x_fft / (x_fft[-1])
    x_norm = x_fft[:200] / (x_fft[199])
    rgba = cmap(x_norm)

    # y_fft=filtre(y_fft,0.001)
    weights = (y_fft[:200]/np.max(y_fft[:200]))*1000
    weights=weights.astype(np.int64)

    # rgba_a= np.average(rgba, axis=0)
    rgba_a = np.average(rgba, axis=0, weights=weights)

    for i in range(4):
        if rgba_a[i] > 1:
            rgba_a[i] = 1.0

    return rgba_a

def test_signal(freq, x, chunk=2048, sr=44100) :
    x=x/(sr*2)
    w=2*np.pi*freq
    a=w.reshape([-1,1]) @x.reshape([1,-1])
    s=10000*np.sin(a)
    y = (np.sum(s, axis=0)).astype(np.int16)

    return y.tobytes()


def main() :

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
        # output=True,
        frames_per_buffer=CHUNK,
    )


    fig, (ax, ax1, ax2) = plt.subplots(3)
    x_fft = np.linspace(0, RATE, CHUNK)
    x = np.arange(0, 2 * CHUNK, 2)
    ax2.set_facecolor("violet")
    (line,) = ax.plot(x, np.random.rand(CHUNK), "r")
    # (line_fft,) = ax1.plot(x_fft, np.random.rand(CHUNK), "b")
    (line_fft,) = ax1.semilogx(x_fft, np.random.rand(CHUNK), "b")
    ax.set_ylim(-32000, 32000)
    ax.set_xlim = (0, CHUNK)
    ax1.set_xlim(20, RATE / 2)
    ax1.set_ylim(0, 1)
    fig.show()
    count=0

    # freq=np.arange(10,10000,10)

    while 1:

        # for f in freq :
            # data =test_signal(np.array([100]),x)
            data = stream.read(CHUNK)
            dataInt = struct.unpack(str(CHUNK) + "h", data)
            line.set_ydata(dataInt)
            # y_fft = np.abs(np.fft.fft(dataInt)) / CHUNK
            y_fft = np.abs(np.fft.fft(dataInt)) / (11000 * CHUNK)
            # y_fft = np.abs(np.fft.fft(dataInt)) * 2 / (11000 * CHUNK)
            line_fft.set_ydata(y_fft)
            # color_mark = cmap(np.random.rand())
            
            if count%2==0 :
                color_mark = colr(y_fft, x_fft)
                count=0
                
            count+=1
            
            line_fft.set_color(color_mark)    
            line.set_color(color_mark)    
            ax2.set_facecolor(color_mark)
            # ax2.set_title(color_mark)
            # ax2.set_facecolor(cmap(0.8))
            fig.canvas.draw()
            fig.canvas.flush_events()


main()
