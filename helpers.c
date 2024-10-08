#include <stdlib.h>
#include <stdio.h>
#include <math.h>

const double LEARNING_RATE = 0.2;

// Struct for Node
typedef struct Node {
    int bias;
    int weightArrSize;
    double* weights;
    double output;
    double* error;
} Node;

// Node Weights initialization
void nodeInit(Node* node, int numInputs) {
    node->weightArrSize = numInputs + 1;
    node->weights = (double*)calloc(node->weightArrSize, sizeof(double));
    node->error = (double*)calloc(numInputs, sizeof(double));
    for(int i = 0; i < node->weightArrSize; i++){
        node->weights[i] = ((double)rand() / RAND_MAX);
    }
    node->bias = 1;
}

// Generate Inputs 
int* generateInputs (int numInputs) {
    int* inputs = (int*)malloc(numInputs * sizeof(int));
    for(int i = 0; i < numInputs; i ++) {
        inputs[i] = rand() % 2;
    }
    return inputs;
}

// Multilayer Activation Func
double sigmoidFunction (double nodeOutput) {
    return (1.0/(1 + exp(-nodeOutput)));
}

int signFunction (double nodeOutput) {
    if(nodeOutput > 0.5) {
        return 1;
    }
    return 0;
}

// Weight Summation
double weightSummation (Node* node, double* inputs) {
    double sum = node->weights[0] * node->bias;
    for(int i = 1; i < node->weightArrSize; i ++) {
        sum += node->weights[i] * inputs[i-1]; // adjust for weights [0] being bias
    }
    return sum;
}

// Error Calculation
double baseErrorCalc (int correct, int guessed) {
    return (correct - guessed);
}

// New Weight Calculation
double weightUpdate (int error, double oldWeight, int input, double learningRate) {
    return oldWeight + (error * input * learningRate);
}

double sumWeight (Node* node) {
    double total = 0;
    for (int i = 0; i < node->weightArrSize; i++) {
        total += node->weights[i];
    }
    return total;
}

// Calculate Error at a given Node in the net
void errorCalc (Node* node, int numInputs, double errorAhead) {
    //printf("Error at Node: %f\n", errorAhead * (weight/sumWeight));
    for(int i = 0; i < numInputs; i++){
        printf("Error at Node: %f\n",errorAhead * (node->weights[i+1]/sumWeight(node)));
        node->error[i] = errorAhead * (node->weights[i+1]/sumWeight(node));
    }
    //node->error = errorAhead * (weight/sumWeight);
}

// New Weight Calc - multilayer
void weightUpdateNode (Node* node, Node* inputs[]) {
    node->weights[0] += node->error[0] * 1 * LEARNING_RATE;
    for(int i = 1; i < node->weightArrSize; i++) {
        printf("Tweaked by: %f\n",node->error[i-1] * inputs[i-1]->output * LEARNING_RATE);
        node->weights[i] += node->error[i-1] * inputs[i-1]->output * LEARNING_RATE;
        printf("New Weight: %f\n", node->weights[i]);
    }
}

double outputCalc (Node* node, double* inputs) {
    return sigmoidFunction(weightSummation(node, inputs));
}

double* hiddenLayerOutputs(Node* inputNodes[], int numNodes){
    double* inputs = (double*)malloc(sizeof(double) * numNodes);
    for(int i = 0; i < numNodes; i++){
        inputs[i] = inputNodes[i]->output;
    }
    return inputs;
}

double outputCalcNodeBased (Node* node, Node* inputNodes[], int numNodes) {
    double* hiddenLayerOut = hiddenLayerOutputs(inputNodes, numNodes);
    double ans = sigmoidFunction(weightSummation(node, hiddenLayerOut));
    free(hiddenLayerOut);
    return ans;
}


// -----------------NN GOALS------------------
int andFunction (int *inputs) {
    return inputs [0] == 1 && inputs[1] == 1;
}

int orFunction (int *inputs) {
    return inputs[0] == 1 || inputs[1] == 1;
}

int xorFunction (Node *inputs[], int numNodes) {
    int* inputsInt = (int*)malloc(sizeof(int) * numNodes);
    for(int i = 0; i < numNodes; i ++) {
        //printf("xor: %i\n", (int)(inputs[i]->output));
        inputsInt[i] = (int)(inputs[i]->output);
    }
    if((orFunction(inputsInt)) && !(andFunction(inputsInt))){
        free(inputsInt);
        return 1;
    }
    free(inputsInt);
    return 0;
}
