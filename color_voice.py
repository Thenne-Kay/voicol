import numpy as np
import pyaudio as pa
import struct
import matplotlib.pyplot as plt
import matplotlib.colors as colors


cmap = plt.get_cmap("gist_rainbow")


def filtre(arr, lim):
    mask = arr > lim
    return arr * mask


def colr(y_fft, x_fft):
    x_norm=((x_fft[:300]/x_fft[299])*9)+1  #upper limit of freq ==6463
    x_norm=np.log10(x_norm)
    rgba = cmap(x_norm)

    # y_fft=filtre(y_fft,0.001)
    weights = (y_fft[:300] / np.max(y_fft[:300]))
    weights=filtre(weights,0.01) * 100
    weights = weights.astype(np.int64)
    rgba_a = np.average(rgba**2, axis=0, weights=weights)
    rgba_a=np.sqrt(rgba_a)

    for i in range(4):
        if rgba_a[i] > 1:
            rgba_a[i] = 1.0

    return rgba_a


def test_signal(freq, x, RATE=44100):
    x = x / (RATE * 2)
    w = 2 * np.pi * freq
    a = w.reshape([-1, 1]) @ x.reshape([1, -1])
    s = 10000 * np.sin(a)
    y = (np.sum(s, axis=0)).astype(np.int16)

    return y.tobytes()

def make_stream(CHUNK = 1024 * 2,FORMAT = pa.paInt16, CHANNELS = 1, RATE = 44100 ):
        p = pa.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            # output=True,
            frames_per_buffer=CHUNK,
        )
        
        return stream

def make_fig(RATE=44100, CHUNK=2*1024) :
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
                
        return fig, ax2, line, line_fft, x, x_fft


def read_stream_data(stream, CHUNK, RATE, duration, fig_par):
    fig, ax2, line, line_fft, _,x_fft=fig_par
    for i in range(int(RATE / CHUNK * duration/4)):
        data = stream.read(CHUNK)
        update(data, fig, ax2, line, line_fft, x_fft,i)

def read_test_data(freq , RATE, fig_par):
    fig, ax2, line, line_fft, x, x_fft=fig_par
    data = test_signal(freq, x, RATE)
    update(data, fig, ax2, line, line_fft, x_fft)


def update(data,fig,ax_color,line,line_fft,x_fft,count=0,interval=3,CHUNK=1024*2) :
    dataInt = struct.unpack(str(CHUNK) + "h", data)
    line.set_ydata(dataInt)
    y_fft = np.abs(np.fft.fft(dataInt)) / (11000 * CHUNK)
    line_fft.set_ydata(y_fft)

    if count % interval == 0:
        color_mark = colr(y_fft, x_fft)
        line_fft.set_color(color_mark)
        line.set_color(color_mark)
        ax_color.set_facecolor(color_mark)
        # ax_color.set_title(color_mark)

    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    CHUNK = 1024
    FORMAT = pa.paInt16
    CHANNELS = 2
    RATE = 44100  # in Hz

    STREAM=make_stream(CHANNELS=CHANNELS, CHUNK=CHUNK)
    fig_par=make_fig()
    fig_par[0].show()
    read_stream_data(STREAM,CHUNK,RATE,5*60,fig_par)
    

    STREAM.stop_stream()
    STREAM.close()
    
    # freq=np.arange(100,7000,10)    
    
    # while 1:
        
    #     for f in freq:
    #         read_test_data(np.array([4000]),RATE,fig_par)
    #         read_test_data(f,RATE,fig_par)
        


main()
