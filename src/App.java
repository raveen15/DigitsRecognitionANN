public class App {
 	
	// Notes:
	// The first thing I did is to make the code a bit more dynamic, there was a clear problem with the backpropagation part,
	// but I wasn't sure if this was the only bug in the code. So to make sure I refactored the whole code to make it a bit more readable.
	// I made two other class: 
	// 1) StatUtil, which is a static class containing stats functions.
	// 2) Layer, which represent a layer with array on neurons inside them.
	// The last class was useful so to modularize the layers and not hardcode them.
	// The first class was useful to remove some code from the main method and make it less bulky.
	// Next thing I did is to remove some variable from the code and put them in the neuron class(min/maxWeight).
	// And to add some variables in the neurons class.
	// When the code was more readable I went to rewrite the forward propagation (which you got right).
	// I then rewrote the back propagation algorithm and this is where you messed up some stuff.
	// Here is a very good step by step explanation of every part of the backprop algorithm : https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
	
	// One thing to note is that the biases are still not implemented as I didn't have enough time to include them. It isn't a problem
	// for the XOR function, but it might be if you are trying to learn a function requiring some degrees of translation.

	// Variable Declaration
   
    // Layers
    static Layer[] layers; // My changes
    
    // Training data
    static TrainingData[] tDataSet; // My changes
   
    // Main Method
    public static void main(String[] args) {
    	// My changes
        // Set the Min and Max weight value for all Neurons
    	Neuron.setRangeWeight(-1,1);
    	
    	// Create the layers
    	// Notes: One thing you didn't code right is that neurons in a layer
    	// need to have number of weights corresponding to the previous layer
    	// which means that the first hidden layer need to have 2 weights per neuron and 6 neurons
    	layers = new Layer[3];
    	layers[0] = null; // Input Layer 0,45
    	layers[1] = new Layer(45,5); // Hidden Layer 45,5
    	layers[2] = new Layer(5,10); // Output Layer 5,10
        
    	// Create the training data
    	CreateTrainingData();
    	
        // System.out.println("============");
        // System.out.println("Output before training");
        // System.out.println("============");
        // for(int i = 0; i < tDataSet.length; i++) {
        //     System.out.println("=====" + i + "======");
        //     for(int j = 0; j < 10; j++){
        //         forward(tDataSet[i].data);
        //         System.out.println(j + ": " + layers[2].neurons[j].value);
        //     }
        // }
       
        train(1000000, 0.1f);

        System.out.println("============");
        System.out.println("Output after training");
        System.out.println("============");
        for(int i = 0; i < tDataSet.length; i++) {
            System.out.println("=====" + i + "======");
            for(int j = 0; j < 10; j++){
                forward(tDataSet[i].data);
                System.out.println(j + ": " + layers[2].neurons[j].value);
            }
        }
        float[] inputZeroTestOne = {0,1,0,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,1,1,1,0};
        forward(inputZeroTestOne);
        System.out.println("\n===Zero Test With Noise===");
        for(int j = 0; j < 10; j++){
            System.out.println(j + ": " + layers[2].neurons[j].value);
        }

        float[] inputOneTestOne = {0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,0};
        forward(inputOneTestOne);
        System.out.println("===One Test With Noise===");
        for(int j = 0; j < 10; j++){
            System.out.println(j + ": " + layers[2].neurons[j].value);
        }

        float[] inputTwoTestOne = {0,1,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,1,1};
        forward(inputTwoTestOne);
        System.out.println("===Two Test With Noise===");
        for(int j = 0; j < 10; j++){
            System.out.println(j + ": " + layers[2].neurons[j].value);
        }

        float[] inputFourTestOne = {0,0,0,1,0,0,0,1,1,0,0,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,1,1};
        forward(inputFourTestOne);
        System.out.println("===Four Test With Noise===");
        for(int j = 0; j < 10; j++){
            System.out.println(j + ": " + layers[2].neurons[j].value);
        }
    }
   
    public static void CreateTrainingData() {
        float[] input0 = {0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0};

        float[] input1 = {0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0};

        float[] input2 = {0,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,1,1,1};

        float[] input3 = {0,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0};

        float[] input4 = {0,0,0,1,0,0,0,1,1,0,0,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,0,0,0,1,0,0,0,0,1,0};
        
        float[] input5 = {1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0};
        
        float[] input6 = {0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0};

        float[] input7 = {1,1,1,1,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0};

        float[] input8 = {0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0};

        float[] input9 = {0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0};

        float[] expectedOutput0 = new float[] {1,0,0,0,0,0,0,0,0,0};
        float[] expectedOutput1 = new float[] {0,1,0,0,0,0,0,0,0,0};
        float[] expectedOutput2 = new float[] {0,0,1,0,0,0,0,0,0,0};
        float[] expectedOutput3 = new float[] {0,0,0,1,0,0,0,0,0,0};
        float[] expectedOutput4 = new float[] {0,0,0,0,1,0,0,0,0,0};
        float[] expectedOutput5 = new float[] {0,0,0,0,0,1,0,0,0,0};
        float[] expectedOutput6 = new float[] {0,0,0,0,0,0,1,0,0,0};
        float[] expectedOutput7 = new float[] {0,0,0,0,0,0,0,1,0,0};
        float[] expectedOutput8 = new float[] {0,0,0,0,0,0,0,0,1,0};
        float[] expectedOutput9 = new float[] {0,0,0,0,0,0,0,0,0,1};
        
        
        // My changes (using an array for the data sets)
        tDataSet = new TrainingData[10];
        tDataSet[0] = new TrainingData(input0, expectedOutput0);
        tDataSet[1] = new TrainingData(input1, expectedOutput1);
        tDataSet[2] = new TrainingData(input2, expectedOutput2);
        tDataSet[3] = new TrainingData(input3, expectedOutput3);
        tDataSet[4] = new TrainingData(input4, expectedOutput4); 
        tDataSet[5] = new TrainingData(input5, expectedOutput5);
        tDataSet[6] = new TrainingData(input6, expectedOutput6);
        tDataSet[7] = new TrainingData(input7, expectedOutput7);
        tDataSet[8] = new TrainingData(input8, expectedOutput8);
        tDataSet[9] = new TrainingData(input9, expectedOutput9);       
    }
    
    public static void forward(float[] inputs) {
    	// First bring the inputs into the input layer layers[0]
    	layers[0] = new Layer(inputs);
    	
        for(int i = 1; i < layers.length; i++) {
        	for(int j = 0; j < layers[i].neurons.length; j++) {
        		float sum = 0;
        		for(int k = 0; k < layers[i-1].neurons.length; k++) {
        			sum += layers[i-1].neurons[k].value*layers[i].neurons[j].weights[k];
        		}
        		sum += layers[i].neurons[j].bias;
        		layers[i].neurons[j].value = StatUtil.Sigmoid(sum);
        	}
        } 	
    }
    
    // This part is heavily inspired from the website in the first note.
    // The idea is that you calculate a gradient and cache the updated weights in the neurons.
    // When ALL the neurons new weight have been calculated we refresh the neurons.
    // Meaning we do the following:
    // Calculate the output layer weights, calculate the hidden layer weight then update all the weights
    public static void backward(float learning_rate,TrainingData tData) {
    	
    	int number_layers = layers.length;
    	int out_index = number_layers-1;
    	
    	// Update the output layers 
    	// For each output
    	for(int i = 0; i < layers[out_index].neurons.length; i++) {
    		// and for each of their weights
    		float output = layers[out_index].neurons[i].value;
    		float target = tData.expectedOutput[i];
    		float derivative = output-target;
    		float delta = derivative*(output*(1-output));
    		layers[out_index].neurons[i].gradient = delta;
    		for(int j = 0; j < layers[out_index].neurons[i].weights.length;j++) { 
    			float previous_output = layers[out_index-1].neurons[j].value;
    			float error = delta*previous_output;
    			layers[out_index].neurons[i].cache_weights[j] = layers[out_index].neurons[i].weights[j] - learning_rate*error;
    		}
    	}
    	
    	//Update all the subsequent hidden layers
    	for(int i = out_index-1; i > 0; i--) {
    		// For all neurons in that layers
    		for(int j = 0; j < layers[i].neurons.length; j++) {
    			float output = layers[i].neurons[j].value;
    			float gradient_sum = sumGradient(j,i+1);
    			float delta = (gradient_sum)*(output*(1-output));
    			layers[i].neurons[j].gradient = delta;
    			// And for all their weights
    			for(int k = 0; k < layers[i].neurons[j].weights.length; k++) {
    				float previous_output = layers[i-1].neurons[k].value;
    				float error = delta*previous_output;
    				layers[i].neurons[j].cache_weights[k] = layers[i].neurons[j].weights[k] - learning_rate*error;
    			}
    		}
    	}
    	
    	// Here we do another pass where we update all the weights
    	for(int i = 0; i< layers.length;i++) {
    		for(int j = 0; j < layers[i].neurons.length;j++) {
    			layers[i].neurons[j].update_weight();
    		}
    	}
    	
    }
    
    // This function sums up all the gradient connecting a given neuron in a given layer
    public static float sumGradient(int n_index,int l_index) {
    	float gradient_sum = 0;
    	Layer current_layer = layers[l_index];
    	for(int i = 0; i < current_layer.neurons.length; i++) {
    		Neuron current_neuron = current_layer.neurons[i];
    		gradient_sum += current_neuron.weights[n_index]*current_neuron.gradient;
    	}
    	return gradient_sum;
    }
 
    
    // This function is used to train being forward and backward.
    public static void train(int training_iterations,float learning_rate) {
    	for(int i = 0; i < training_iterations; i++) {
    		for(int j = 0; j < tDataSet.length; j++) {
                forward(tDataSet[j].data);
    			backward(learning_rate,tDataSet[j]);
    		}
    	}
    }
}