#!/usr/bin/env python
""" models.py: 

Handles online modeling and model creation classes.

# VERSION UPDATES
0.0.1 (Apr/17/2018) : initial release

"""
__author__ = "Vinicius G. Goecks"
__version__ = "0.0.1"
__date__ = "April 17, 2018"

# import
import numpy as np
import matplotlib.pyplot as plt

import threading
import sys, os
import time

import tensorflow as tf

class ThreadingModeling(object):
    """ Running the modeling functions on the background:

    The run() method will be started and it will run in the background
    until the application exits.

    The updated model is queried whenever it is needed.
    """

    def __init__(self, memory_buffer, batch_size=1, update_model_dt=0,
                 run_id='test'):
        # keep track of current epi and time step to know model is updated
        self.run_id = run_id
        self.epi_n = 0
        self.step_n = 0
        self.track_model = []
        self.hist_train = []

        # initialize memory buffer
        self.memory = memory_buffer

        # create initial model
        self.batch_size = batch_size
        self.update_model_dt = update_model_dt  # how often should be model
        # be updated (sec) (assuming we
        #  have CPU power for that)
        self.__init_model()

        # run model updates in the background forever
        self.thread = threading.Thread(target=self.__update_model, args=())
        self.thread.daemon = True    # kills background thread when main
        self.keep_computing_model = True  # flag to stop daemon
        # function is over
        self.thread.start()

    def __init_model(self):
        """ Initialize pre-defined model.
        """
        print('[*] Initializing model...')
        model = Sequential()

        # model.add(Dropout(0.2, input_shape=(input_dim,)))
        model.add(Dense(220,
                        input_shape=(self.memory.n_inputs,),
                        kernel_initializer='normal',
                        activation='relu'))

        # model.add(Dropout(0.2))
        model.add(Dense(160, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(130, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.memory.n_outputs,
                        kernel_initializer='normal', activation='linear'))

        # compile model
        model.compile(loss='mse', optimizer='adam')

        # save model internally and dump on file
        self.model = model
        self.model.save('./experiments/' + self.run_id + '/initial_model.h5')

    def __update_model(self):
        """ Receive new batch of data and update model.
        """
        while self.keep_computing_model:
            start_time = time.time()

            # receive new data
            input_data, output_data = self.memory.generate_batch(
                batch_size=self.batch_size)

            # only update model when validation set is ready to use
            if self.memory.val_data_filled:
                if input_data is not None:
                    # update model if data is not None

                    # prepare validation data
                    val_input = self.memory.val_data['data_in']
                    val_output = self.memory.val_data['data_out']

                    # # TODO: both options below seem to work fine, but need to do
                    # # some research (or testing) to see if they are equivalent
                    # option 1
                    hist = self.model.fit( input_data, output_data, epochs=1,
                                   steps_per_epoch=1, verbose=0,
                                   validation_data=(val_input, val_output),
                                   validation_steps=1)
                    # # option 2
                    # self.model.train_on_batch(input_data, output_data)

                    # update list that tracks when model was updated
                    # print('[*] Model updated.')
                    self.track_model.append((self.epi_n+1, self.step_n))

                    # save fit history
                    self.hist_train.append( (hist.history['loss'][0],
                                             hist.history['val_loss'][0]) )

                # follow specified time delay
                time_compute = time.time() - start_time
                if time_compute < self.update_model_dt:
                    # computed too fast, way a bit to follow dt
                    time.sleep(self.update_model_dt - time_compute)

    def compare_models(self):
        """After training, load different models and plot time history
        of their predictions so one can visually compare them.
        """
        # list models
        model_names = ['/initial_model.h5', '/final_model.h5']

        # prepare validation data
        val_input = self.memory.val_data['data_in']
        val_states = val_input[:,0:self.memory.n_states]
        val_controls = val_input[:,-self.memory.n_controls]
        n_steps = self.memory.val_data_size

        # make sure arrays have same structure
        val_states = val_states.reshape(n_steps, self.memory.n_states)
        val_controls = val_controls.reshape(n_steps, self.memory.n_controls)

        # MAIN LOOP
        for i in range(len(model_names)):
            # load model
            model = load_model('./experiments/' + self.run_id + model_names[i])
            pred_states = np.zeros((n_steps, self.memory.n_states))

            # step-by-step prediction (using validation data)
            current_state = val_states[0,:]
            control = val_controls[0,:]

            # store data
            pred_states[0,:] = current_state

            for j in range(1,n_steps):

                # predict next states
                # format input data and predict different in next states
                input_data = np.hstack((current_state, control))
                delta_next_state = model.predict(input_data.reshape(
                    1, self.memory.n_inputs))

                # return next states
                next_state = current_state + delta_next_state[0]

                # update states and controls
                current_state = next_state
                control = val_controls[j,:]

                # store data
                pred_states[j,:] = current_state

            # plot predicted data            
            plt.figure()
            
            # plot states
            for l in range(self.memory.n_states):
                plt.subplot(self.memory.n_states+1, 1, l+1)
                if l == 0:
                    plt.title('Model comparison: {}'.format(model_names[i]))

                plt.plot(val_states[:, l], '-', label='x{}'.format(l))
                plt.plot(pred_states[:, l], '--', label='pred_x{}'.format(l))
                plt.grid()
                plt.legend(loc='best')

            # plot controls
            plt.subplot(self.memory.n_states+1, 1, self.memory.n_states+1)
            for m in range(self.memory.n_controls):
                plt.plot(val_controls[:, m], label='u{}'.format(m))

            plt.grid()
            plt.legend(loc='best')



    def predict_next_states(self, current_state, control):
        """ Predict next states using current model based on current states and
        control performed.
        """
        # format input data and predict different in next states
        input_data = np.hstack((current_state, control))
        delta_next_state = self.model.predict(input_data.reshape(
            1, self.memory.n_inputs))

        # return next states
        next_state = current_state + delta_next_state[0]
        return next_state

    def close(self):
        """ Raise flag to stop daemon thread.
        """
        self.keep_computing_model = False