# Software Engineering - Machine Learning -> juanvergaramunoz, 01/22/2019


** Project: Machine Learning Examples
** Creator: juanvergaramunoz
** Year: 2019


## Description

This Software Engineering repository includes general examples of Machine Learning techniques

The target of this project is to provide a set of Machine Learning tools (still in development)

**This project has the following CLASSES:**

1) "DNN Class - Tensorflow.py" has the "DNN_Model" Class -- FUNCTIONS:
    
       - __init__(self, x_size, y_size = 1)
       
       - _get_placeholders(self, x_size,y_size)
       
       - create_training_examples(self, function, samples = 1000)
       
       - _create_layer(self, prev_layer, n_prev, n_post, layer = "", ReLU = True)
       
       - _create_network(self)
       
       - save_session(self, sess, name, identifier = 0)
       
       - restore_session(self, sess, path)
       
       - create_model(self, middle_layers = None, x_size = None, y_size = None)
       
       - initialize_model(self, learning_rate)
       
       - train_model(self, X_train, Y_train, training_steps = 40000, learning_rate = 0.000078, batch_size = 20, plot_cost = True, print_variables = [20,None])
       
       - test_model(self, X_test, Y_test)
       
       - print_all_variables(self, Var_Num = 0)
       
       - save_all_variables(self, Var_Num = 0, values_dict = {})
       
       - plot_image(self, n, obj, percentage_shown, y_label = "", title = None)
 
 


**This project has the following EXAMPLES:**

1) "Customizable DNN Example - Tensorflow.ipynb"

    - This notebook shows an example of a simple 3_inputs, 1_output NN, where a simple model is ussed to learn the function provided
    
    - Plots are shown to present the learning process, and the correspondent accuracy after testing
    
