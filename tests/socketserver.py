import zmq
import numpy as np
#import bokeh.plotting
#import bokeh.io
import matplotlib.pyplot as plt
import trlib_test_matrix_msg_pb2

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5678")

#bokeh.plotting.output_server('foo')
#p_state = bokeh.plotting.figure()
#alpha = []
#color = []
#color = ['#cc0000' for i in range(4)]
#alpha = [1.0, 0.0, 0.1, 0.5]
#ds = bokeh.models.ColumnDataSource(data=dict(alphas=alpha, colors=color))
#p = bokeh.plotting.figure()
#p.rect('', '', 0.9, 0.9, source=ds, color='colors', alpha='alphas')
#bokeh.io.show(p)

while True:
    message = socket.recv()
    msg = trlib_test_matrix_msg_pb2.trlib_matrix_message()
    msg.ParseFromString(message)
    dat = np.array(msg.data._values).reshape(msg.m, msg.n)
    print dat
    plt.matshow(dat)
    plt.show()

    #color = []
    #alpha = []
    #the_color = '#cc0000'

    #for col in range(msg.n):
    #    for rowi in range(msg.m):
    #        color.append(the_color)
    #        alpha.append(dat[rowi, col])
   
    #ds.data['alphas'] = alpha
    #ds.data['colors'] = color

    #bokeh.io.push()

    socket.send(b"OK")
    
