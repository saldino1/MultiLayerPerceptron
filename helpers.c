#include <stdlib.h>
#include <math.h>

// Struct for Node
typedef struct Node {
    int bias;
    int weightArrSize;
    double* weights;
    double output;
    double error;
} Node;

// Node Weights initialization
void nodeInit(Node* node, int numInputs) {
    node->weightArrSize = numInputs + 1;
    node->weights = (double*)calloc(node->weightArrSize, sizeof(double*));
    for(int i = 0; i < node->weightArrSize; i++){
        node->weights[i] = ((double)rand() / RAND_MAX);
    }
    node->bias = 1;
}

// Generate Inputs 
int* generateInputs (int numInputs) {
    int* inputs = (int*)malloc(numInputs * sizeof(int*));
    for(int i = 0; i < numInputs; i ++) {
        inputs[i] = rand() % 2;
    }
    return inputs;
}

// Activation Function DEPRECIATED
int signFunction (double nodeOutput) {
    if(nodeOutput > 0) {
        return 1;
    }
    return 0;
}

// Multilayer Activation Func
double sigmoidFunction (double nodeOutput) {
    return (1.0/(1 + exp(-nodeOutput)));
}

// Weight Summation
double weightSummation (Node* node, int* inputs) {
    double sum = node->weights[0] * node->bias;
    for(int i = 1; i < node->weightArrSize; i ++) {
        sum += node->weights[i] * inputs[i-1]; // adjust for weights [0] being bias
    }
    return sum;
}

// Error Calculation
int errorCalc (int correct, int guessed) {
    return (correct - guessed);
}

// New Weight Calculation
double weightUpdate (int error, double oldWeight, int input, double learningRate) {
    return oldWeight + (error * input * learningRate);
}

double outputCalc (Node* node, int* inputs) {
    return sigmoidFunction(weightSummation(node, inputs));
}

// -----------------NN GOALS------------------
int andFunction (int *inputs) {
    if(inputs [0] == 1 && inputs[1] == 1){
        return 1;
    }
    return 0;
}

int orFunction (int *inputs) {
    if(inputs[0] == 1 || inputs[1] == 1){\
        return 1;
    }
    return 0;
}

int xorFunction (int *inputs) {
    if((orFunction(inputs)) && !(andFunction(inputs))){
        return 1;
    }
    return 0;
}
