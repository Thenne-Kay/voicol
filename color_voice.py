import numpy as np
import pyaudio as pa
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as colors
import librosa as rosa

import wave

import time 


cmap = plt.get_cmap("gist_rainbow")


class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for subclasses of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()


def filtre(arr, lim):
    mask = arr > lim
    return arr * mask


def colr_mix(y_fft, x_fft):
    x_norm = ((x_fft[:300] / x_fft[299]) * 9) + 1  # upper limit of freq ==6463
    x_norm = np.log10(x_norm)
    rgba = cmap(x_norm)

    # y_fft=filtre(y_fft,0.001)
    weights = y_fft[:300] / np.max(y_fft[:300])
    weights = filtre(weights, 0.01) * 100
    weights = weights.astype(np.int64)
    rgba_a = np.average(rgba**2, axis=0, weights=weights)
    rgba_a = np.sqrt(rgba_a)

    for i in range(4):
        if rgba_a[i] > 1:
            rgba_a[i] = 1.0

    return rgba_a


def normalize_scale(scale, lim=300):
    scale_norm = ((scale[:lim] / scale[lim - 1]) * 9) + 1
    scale_norm = np.log10(scale_norm)

    return scale_norm


def colr(y_fft, x_norm, lim=300, filter_low=0.01):
    # y_fft=filtre(y_fft,0.001)
    weights = y_fft[:lim] / np.max(y_fft[:lim])
    weights = filtre(weights, filter_low) * 100
    weights = weights.astype(np.int64)

    x_rms = np.average(x_norm**2, axis=0, weights=weights)
    x_rms = np.sqrt(x_rms)

    rgba = cmap(x_rms)

    return rgba


def test_signal(freq, x, RATE=48000):
    x = x / (RATE * 2)
    w = 2 * np.pi * freq
    a = w.reshape([-1, 1]) @ x.reshape([1, -1])
    s = 1000000000 * np.sin(a)
    y = (np.sum(s, axis=0)).astype(np.int32)

    return y.tobytes()


def make_stream(CHUNK=1024 * 2, FORMAT=pa.paInt16, CHANNELS=1, RATE=44100):
    p = pa.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK,
    )

    return stream


def make_fig(RATE=44100, CHUNK=2 * 1024) :
    fig, (ax, ax1, ax2) = plt.subplots(3)
    x_fft = np.linspace(0, RATE, CHUNK)
    x = np.arange(0, 2 * CHUNK, 2)
    ax2.set_facecolor("pink")

    ax2_background = ax2.imshow(
        np.zeros((1, 1, 4)),
        aspect="auto",
        extent=[*ax2.get_xlim(), *ax2.get_ylim()],
        zorder=-1,
        animated=True,
    )

    (line,) = ax.plot(x, np.random.rand(CHUNK), "r", animated=1)
    # (line_fft,) = ax1.plot(x_fft, np.random.rand(CHUNK), "b")
    (line_fft,) = ax1.semilogx(x_fft, np.random.rand(CHUNK), "b", animated=1)
    ax.set_ylim(-32000, 32000)
    ax.set_xlim = (0, CHUNK)
    ax1.set_xlim(20, RATE / 2)
    ax1.set_ylim(0, 1)

    return fig, ax2_background, line, line_fft, x, x_fft


def read_stream_data(stream, CHUNK, RATE, duration, fig_par,bm):
    fig, ax2, line, line_fft, _, x_fft = fig_par
    x_norm = normalize_scale(x_fft, lim=300)

    for i in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        update(data, fig, ax2, line, line_fft, x_norm,bm,i, CHUNK=CHUNK)


def read_file_data(filename, CHUNK, fig_par, stream, bm):
    fig, ax2, line, line_fft, _, x_fft = fig_par
    x_norm = normalize_scale(x_fft, lim=300)

    wf = wave.open("./data/" + filename, "rb")
    data = wf.readframes(CHUNK)
    count=0

    while data != "":
        stream.write(data)
        update2(data, fig, ax2, line, line_fft, x_norm,bm,count=count ,CHUNK=CHUNK)
        count=count+1
        data = wf.readframes(CHUNK)


def read_test_data(freq, RATE, fig_par,bm):
    fig, ax2, line, line_fft, x, x_fft = fig_par
    x_norm = normalize_scale(x_fft, lim=300)
    data = test_signal(freq, x, RATE)
    update(data, fig, ax2, line, line_fft, x_norm,bm, CHUNK=1024)


def update(data, fig, ax_color, line, line_fft, x_norm,bm, count=0, interval=10, CHUNK=1024):
    # ns=struct.calcsize(data)
    dataInt = struct.unpack(str(CHUNK) + "i", data)
    dataInt=np.array(dataInt)/100000
    y_fft = np.abs(np.fft.fft(dataInt)) / (4800 * CHUNK * 2)
    

    
    line.set_ydata(dataInt)
    line_fft.set_ydata(y_fft)

    if count % interval == 0:
        color_mark = colr(y_fft, x_norm)
        line_fft.set_color(color_mark)
        line.set_color(color_mark)
        ax_color.set_array(np.array(color_mark).reshape((1,1,4)))
        
       
        
        
    bm.update()    



def update2(data, fig, ax_color, line, line_fft, x_norm,bm ,count=0, interval=20, CHUNK=1024 * 2):

    dataInt = struct.unpack(str(CHUNK) + "i", data)
    dataInt=np.array(dataInt)/100000
    y_fft = np.abs(np.fft.fft(dataInt)) / (4800 * CHUNK)
    

    
    line.set_ydata(dataInt)
    line_fft.set_ydata(y_fft)

    if count % interval == 0:
        color_mark = colr(y_fft, x_norm)
        line_fft.set_color(color_mark)
        line.set_color(color_mark)
        ax_color.set_array(np.array(color_mark).reshape((1,1,4)))
        
       
        
        
    bm.update()    


    

    # fig.canvas.draw()
    # fig.canvas.flush_events()


def main():
    CHUNK = 1024
    FORMAT = pa.paInt16
    CHANNELS = 2
    RATE = 48000  # in Hz

    fig_par = make_fig(RATE=48000, CHUNK=CHUNK)
    bm=BlitManager(fig_par[0].canvas,[fig_par[1],fig_par[2],fig_par[3]])
    fig_par[0].show()    
    
    STREAM = make_stream(CHANNELS=CHANNELS, CHUNK=CHUNK, RATE=48000)
    # read_stream_data(STREAM,CHUNK,RATE,60,fig_par,bm)
    read_file_data("1c.wav", CHUNK, fig_par, STREAM, bm)

    # # STREAM.stop_stream()
    # # STREAM.close()

    # freq=np.arange(100,7000,10)

    # while 1:

    #     for f in freq:
    #         # read_test_data(np.array([4000]),RATE,fig_par,bm)
    #         read_test_data(f,RATE,fig_par,bm)


main()
