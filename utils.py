import numpy as np
import h5py
import onnxruntime as onnxrt
import matplotlib.pyplot as plt

def load_h5(filename,mode='r'):
    with h5py.File(filename,mode) as f:
        data = f['data'][:]
    return data

def bmode(IQ, DR):
    assert (DR > 0), 'The dynamic range DR in dB must be > 0.'
    I = np.abs(IQ)
    I = 20*np.log10(I/np.max(I)) + DR
    I[np.where(I < 0)] = 0
    I = (255*I/DR).astype('uint8')
    return I

def display(grid_x, grid_z, image, title):
    plt.pcolormesh(grid_x, grid_z, image, cmap='Greys_r',shading='gouraud')
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.title(title,fontsize=16)
    plt.show()

def inference(model, inputs):
    inputs = np.expand_dims(inputs, axis=0)
    model = onnxrt.InferenceSession(model,providers=['CUDAExecutionProvider'])
    inputs = {model.get_inputs()[0].name:inputs.real,
              model.get_inputs()[1].name:inputs.imag}
    outputs = model.run(None, inputs)
    outputs = outputs[0][0, 0, :, :] + 1j*outputs[1][0, 0, :, :]
    return outputs

def symmetry_padding(data, shape):
    h, w = data.shape[-2], data.shape[-1]
    h_out, w_out = shape[0], shape[1]

    h_gap, w_gap = h_out - h, w_out - w
    w_gap_l = int((w_out-w)/2)
    w_gap_r = w_gap-w_gap_l

    line_fill = data[...,h-h_gap:h,:]
    line_fill = line_fill[...,::-1,:]
    data = np.concatenate((data, line_fill), axis=-2)
    print('Down padding {}, Original depth {}, Current depth {}'.format(h_gap, h, h_out))

    line_fill = data[..., :, 0:w_gap_l]
    line_fill = line_fill[..., : ,::-1]

    data = np.concatenate((line_fill,data),axis=-1)
    print('Left padding {}, Original width {}, Current width {}'.format(w_gap_l, w, w + w_gap_l))

    line_fill = data[..., :, w_gap_l + w - w_gap_r:w_gap_l + w]
    line_fill = line_fill[..., :,::-1]

    data = np.concatenate((data,line_fill),axis=-1)
    print('Right padding {}, Original width {}, Current width {}'.format(w_gap_r, w + w_gap_l, w + w_gap_l+w_gap_r))

    return data, np.array([h, w, h_gap, w_gap_l, w_gap_r])

def recover_shape(data, shape_info):
    h, w, h_gap, w_gap_l, w_gap_r = shape_info
    data = data[..., 0:h, :]
    data = data[..., w_gap_l: w_gap_l+w]
    return data