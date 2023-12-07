
import numpy as np
import tensorcircuit as tc
import tensorflow as tf
import struct
import os
# from tensorcircuit import keras
# from tensorflow import keras

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

K = tc.set_backend("tensorflow")

Rx = tc.gates.rx
Ry = tc.gates.ry
Rz = tc.gates.rz

def entangling_layer(c, qubits):
    """
    qubits: [0,1,2,3,4..] qubit index list
    Return a layer of CZ entangling gates on "qubits" (arranged in a circular topology)
    """
    for q0, q1 in zip(qubits, qubits[1:]):  ## CNOT and CZ can both be well
        c.cz(q0 ,q1) 

    if len(qubits) != 2:  
        c.cz(qubits[0], qubits[-1])

def inputs_encoding(c, input_params):
    """
    params = np.array(size=(n_qubit,))
    here the input_params are scaled input and scaling will be put into the torch layer
    """
    for qubit, input in enumerate(input_params):
        c.rx(qubit, input)  


def entangling_layer_trainable_noise(c, qubits, symbols, px, py, pz, seed):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    the length of symbols equals to the number of qubits
    """
    for i, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
        c.exp1(q0, q1, theta=symbols[i], unitary=tc.gates._zz_matrix)
        c.depolarizing(q0, px=px, py=py, pz=pz, status=seed[i, 0])  
        c.depolarizing(q1, px=px, py=py, pz=pz, status=seed[i, 1])

    if len(qubits) != 2:
        c.exp1(qubits[0], qubits[-1], theta=symbols[-1], unitary=tc.gates._zz_matrix)
        c.depolarizing(qubits[0], px=px, py=py, pz=pz, status=seed[-1, 0])  
        c.depolarizing(qubits[-1], px=px, py=py, pz=pz, status=seed[-1, 1])


def entangling_layer_trainable(c, qubits, symbols):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    the length of symbols equals to the number of qubits
    """
    for i, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
        c.exp1(q0, q1, theta=symbols[i], unitary=tc.gates._zz_matrix)
        # c.depolarizing(q0, px=px, py=py, pz=pz, status=seed[i, 0]) ## 加入noise
        # c.depolarizing(q1, px=px, py=py, pz=pz, status=seed[i, 1])

    if len(qubits) != 2:
        c.exp1(qubits[0], qubits[-1], theta=symbols[-1], unitary = tc.gates._zz_matrix)
        # c.depolarizing(qubits[0], px=px, py=py, pz=pz, status=seed[-1, 0]) ## 加入noise
        # c.depolarizing(qubits[-1], px=px, py=py, pz=pz, status=seed[-1, 1])

def quantum_circuit(inputs, params, lamda_scaling, q_weights_final):
    """
    Prepares a data re-uploading circuit on "qubits" with 'n_layers' layers
    inputs: len(inputs) = n_qubits
    weights.shape (n_layers+1, n_qubits, 3) as each qubit has three parameters to be encoded
    Only weights are trainable
    lambda_scaling: (n_qubits*n_layers, )
    """
    # lambda_scaling with shape (layers, qubits)

    weights_var = params[: ,: ,:2]
    weight_ent = params[: ,: ,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
    # print(scaled_inputs.shape)
    
    qubits = range(n_qubits)
    ## the first Hardmard layer with no repeating
    for qubit in qubits:
        c.H(qubit)   ## transform them into two |+> state
    
    for layer in range(n_layers):  
        # Variational layer
        for qubit in range(n_qubits): ## data reuploading
            c.ry(qubit, theta=inputs[qubit ] *lamda_scaling[layer, qubit])
            # c.rz(qubit, theta=inputs[qubit]*lamda_scaling[layer, qubit, 1])

        for qubit in qubits:
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer ,qubit, 0])
            # c.ry(qubit, theta=weights_var[layer,qubit, 1])
            c.rz(qubit, theta=weights_var[layer ,qubit, 1])

        ## entangling layer, weight_ent
        entangling_layer_trainable(c, qubits, weight_ent[layer, :]) ## there is no variantional qubits
        # Encoding layer
        # for qubit in qubits:
        # inputs_encoding(inputs[layer, :])  
    # # Last varitional layer
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        # c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 1])

    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([tc.gates.x(), [i]])) for i in range(n_qubits)] +
        #     [K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    )
    outputs = K.reshape(outputs, [-1])
    return outputs


def quantum_circuit_fixing(inputs, params, q_weights_final, qweights_basis):

    weights_var = params[: ,: ,:3]
    weight_ent = params[: ,: ,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)

    # print(scaled_inputs.shape)
    
    qubits = range(n_qubits)
    # the first Hardmard layer with no repeating
    for qubit in qubits:
        c.H(qubit)   # transform them into two |+> state
   
    for layer in range(n_layers):  
        # Variational layer
        for qubit in range(n_qubits):  # data reuploading
            c.ry(qubit, theta=K.asin(inputs[qubit]))
            c.rz(qubit, theta=K.asin(inputs[qubit]))
        for qubit in qubits:
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rz(qubit, theta=weights_var[layer, qubit, 0])
            c.ry(qubit, theta=weights_var[layer, qubit, 1])
            c.rz(qubit, theta=weights_var[layer, qubit, 2])

        ## entangling layer, weight_ent
        entangling_layer_trainable(c, qubits, weight_ent[layer, :])

    # # Last varitional layer
    for qubit in qubits:
        c.rz(qubit, theta=q_weights_final[qubit, 0])
        c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 2])

    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        + [K.real(c.expectation([Rx(qweights_basis[i]), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([tc.gates.y(), [i]])) for i in range(int(n_qubits / 2), n_qubits)]
    )
    outputs = K.reshape(outputs, [-1])  

    return outputs


def quantum_circuit_simple(inputs, params, q_weights_final):

    weights_var = params[: ,: ,:3]
    weight_ent = params[: ,: ,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)

    # print(scaled_inputs.shape)
    
    qubits = range(n_qubits)
    # the first Hardmard layer with no repeating
    for qubit in qubits:
        c.H(qubit)   # transform them into two |+> state
   
    for layer in range(n_layers):  
        # Variational layer
        for qubit in range(n_qubits):  # data reuploading
            c.ry(qubit, theta=K.asin(inputs[qubit]))
            c.rz(qubit, theta=K.asin(inputs[qubit]))
        for qubit in qubits:
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rz(qubit, theta=weights_var[layer, qubit, 0])
            c.ry(qubit, theta=weights_var[layer, qubit, 1])
            c.rz(qubit, theta=weights_var[layer, qubit, 2])

        ## entangling layer, weight_ent
        entangling_layer_trainable(c, qubits, weight_ent[layer, :])

    # # Last varitional layer
    for qubit in qubits:
        c.rz(qubit, theta=q_weights_final[qubit, 0])
        c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 2])

    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([Rx(qweights_basis[i]), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([tc.gates.y(), [i]])) for i in range(int(n_qubits / 2), n_qubits)]
    )
    outputs = K.reshape(outputs, [-1])  

    return outputs


def quantum_circuit_sample(inputs, params, q_weights_final, status):

    # lambda_scaling with shape (layers, qubits)
    weights_var = params[:,:,:3]
    weight_ent = params[:,:,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
    random_key = status

    qubits = range(n_qubits)
    # the first Hardmard layer with no repeating
    for qubit in qubits:
        c.H(qubit)   # transform them into two |+> state

    for layer in range(n_layers):
        # Variational layer
        for qubit in range(n_qubits):  # data reuploading
            c.rz(qubit, theta=inputs[qubit])
            c.ry(qubit, theta=inputs[qubit])
            c.rz(qubit, theta=inputs[qubit])

        for qubit in qubits:
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rz(qubit, theta=weights_var[layer, qubit, 0])
            c.ry(qubit, theta=weights_var[layer, qubit, 1])
            c.rz(qubit, theta=weights_var[layer, qubit, 2])

        # entangling layer, weight_ent
        entangling_layer_trainable(c, qubits, weight_ent[layer, :]) ## there is no variantional qubits

    # # Last varitional layer
    for qubit in qubits:
        c.rz(qubit, theta=q_weights_final[qubit, 0])
        c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 2])

    # outputs = K.stack(
    #     [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
    #     # + [K.real(c.expectation([tc.gates.x(), [i]])) for i in range(n_qubits)]
    #     + [K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    # )

    outputs_z = c.measure_jit(*qubits, status=random_key)[0]

    # outputs_z = c.sample(allow_state=True)

    # for qubit in qubits:
    #     c.rx(qubit, theta=basis_params[qubit, ])  # trainable basis for theta=pi/2, measurement Y basis

    # outputs_t = c.measure_jit(qubits, with_prob=True, status=random_key)
    outputs = K.reshape(outputs_z, [-1])

    return outputs



def quantum_encoding_large(inputs, params, q_weights_final):
    """
    Args:
        inputs: input data for angle encoding, with shape 8*8
        params: variational parameters
        q_weights_final:
    Returns: observable expectation
    """

    weights_var = params[:, :, :3]
    weight_ent = params[:, :, -1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)

    qubits = range(n_qubits)

    for layer in range(n_layers):  # 每一行的所有参数都表示所有的变分参数 theta0-----theta9

        quantum_rx_large_encoding(c, inputs)  # encoding

        for qubit in qubits:
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer, qubit, 0])
            c.ry(qubit, theta=weights_var[layer, qubit, 1])
            c.rz(qubit, theta=weights_var[layer, qubit, 2])
        # entangling layer, weight_ent
        entangling_layer_trainable(c, qubits, weight_ent[layer, :])
        # Last varitional layer
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 2])

    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([tc.gates.x(), [i]])) for i in range(n_qubits)] +
        # +[K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    )
    outputs = K.reshape(outputs, [-1])

    return outputs


def quantum_encoding_qout(inputs, params, q_weights_final):
    """
    Args:
        inputs: input data for angle encoding, with shape 8*8
        params: variational parameters
        q_weights_final:
    Returns: observable expectation
    """

    weights_var = params[:, :, :2]
    weight_ent = params[:, :, -1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)

    qubits = range(n_qubits)

    for layer in range(n_layers):  

        quantum_rx_large_encoding(c, inputs)  # encoding

        for qubit in qubits:
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer, qubit, 0])
            # c.ry(qubit, theta=weights_var[layer, qubit, 1])
            c.rz(qubit, theta=weights_var[layer, qubit, 1])
        # entangling layer, weight_ent
        entangling_layer_trainable(c, qubits, weight_ent[layer, :])
        # Last varitional layer
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        # c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 1])

    # outputs = K.stack(
    #     [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
    #     # + [K.real(c.expectation([tc.gates.x(), [i]])) for i in range(n_qubits)] +
    #     # +[K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    # )
    outputs = tf.reshape(tf.math.abs(c.state()), [-1])

    return outputs


def quantum_encoding_large_G(inputs, params, q_weights_final, lams):
    """
    Args:
        inputs: input data for angle encoding, with shape 8*8
        params: variational parameters
        q_weights_final:
    Returns: observable expectation
    """
    weights_var = params[:, :, :3]
    weight_ent = params[:, :, -1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)

    qubits = range(n_qubits)

    quantum_rx_large_encoding(c, inputs, lams)  # encoding

    for layer in range(n_layers):  

        for qubit in qubits:
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rz(qubit, theta=weights_var[layer, qubit, 0])
            c.ry(qubit, theta=weights_var[layer, qubit, 1])
            c.rz(qubit, theta=weights_var[layer, qubit, 2])
        # entangling layer, weight_ent
        entangling_layer_trainable(c, qubits, weight_ent[layer, :])
        # Last varitional layer
    for qubit in qubits:
        c.rz(qubit, theta=q_weights_final[qubit, 0])
        c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 2])

    if n_qubits == 8:
        outputs = K.stack(
            [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
            # + [K.real(c.expectation([tc.gates.x(), [i]])) for i in range(n_qubits)] +
            + [K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
        )
    elif n_qubits == 16:
        outputs = K.stack(
            [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
            # + [K.real(c.expectation([tc.gates.x(), [i]])) for i in range(n_qubits)] +
            # + [K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
        )
    outputs = K.reshape(outputs, [-1])

    return outputs


def quantum_rx_large_encoding(c, inputs):
    # lams with shape (8,8)
    n_features = tf.keras.backend.int_shape(inputs)[0]
    n_qubits = c.circuit_param['nqubits']
    qubits_per_features = int(n_features /n_qubits)
    for i in range(n_qubits):
        for j in range(qubits_per_features):
            c.rx(i, theta=inputs[ i *qubits_per_features +j])
            c.ry(i, theta=inputs[i * qubits_per_features + j])




def quantum_circuit_Noise_001(inputs, params, q_weights_final):
    """
    Adding the noise into the quantum circuit followed by the tow qubit gates.

    Args:
        inputs:
        params:
        q_weights_final:
    """
    # choosing small learning rate.
    px, py, pz = 0.01, 0.01, 0.001  # small noise value, noise seriously influence the performance of
    weights_var = params[:, :, :3]
    weights_ent = params[:, :, -1]
    # 对inputs进行拆分，一部分是noise seed 一部分是data
    n_layers, n_qubits, _ = weights_var.shape
    input_data = inputs[:n_qubits]
    seeds = inputs[n_qubits:]
    seeds = K.reshape(seeds, (n_qubits, 2))

    c = tc.Circuit(n_qubits)
    qubits = range(n_qubits)

    for layer in range(n_layers):

        for qubit in range(n_qubits):  # re-uploading data encoding
            c.rz(qubit, theta=K.asin(input_data[qubit]))
            c.ry(qubit, theta=K.asin(input_data[qubit]))
        # Variational layer
        for qubit in qubits:
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rz(qubit, theta=weights_var[layer, qubit, 0])
            c.ry(qubit, theta=weights_var[layer, qubit, 1])
            c.rz(qubit, theta=weights_var[layer, qubit, 2])

        # training entangling layer
        entangling_layer_trainable_noise(c, qubits, weights_ent[layer, :], px, py, pz, seeds)

        # last variational layer for constructing Haar unitary
    for qubit in qubits:
        c.rz(qubit, theta= q_weights_final[qubit, 0])
        c.ry(qubit, theta= q_weights_final[qubit, 1])
        c.rz(qubit, theta= q_weights_final[qubit, 2])

    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([Rx(training_basis[i]), [i]])) for i in range(n_qubits)]
        # +[K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    )

    outputs = K.reshape(outputs, [-1])

    return outputs











