#include "helpers.c"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main () {
    const int iterations = 10;
    const int inputLayerSize = 2;
    const int hiddenLayerSize = 2;
    const int outputLayerSize = 1;
    int countCorrect = 0;

    Node* inputLayer[inputLayerSize];
    Node* hiddenLayer[hiddenLayerSize];
    Node* outputLayer[outputLayerSize];

    srand(time(NULL));

    for(int i = 0; i < inputLayerSize; i ++) {
        inputLayer[i] = (Node*)malloc(sizeof(Node));
    }
    for(int i = 0; i < hiddenLayerSize; i ++) {
        hiddenLayer[i] = (Node*)malloc(sizeof(Node));
        nodeInit(hiddenLayer[i], inputLayerSize);
    }
    for(int i = 0; i < outputLayerSize; i ++) {
        outputLayer[i] = (Node*)malloc(sizeof(Node));
        nodeInit(outputLayer[i], hiddenLayerSize);
    }
  
    // Begin feedforward process
    printf("Neural Net Starts Now\n");
    for(int i = 0; i < iterations; i++) {
        int* inputsList = generateInputs(inputLayerSize);
        printf("Input of: ");
        for(int j = 0; j < inputLayerSize; j++) {
            inputLayer[j]->output = inputsList[j];
            printf("%.1f ", inputLayer[j]->output);
        }
        printf("\n");
        for(int j = 0; j < hiddenLayerSize; j++){
            hiddenLayer[j]->output = outputCalcNodeBased(hiddenLayer[j], inputLayer, inputLayerSize);
        }
        for(int j = 0; j < outputLayerSize; j++){
            outputLayer[j]->output = signFunction(outputCalcNodeBased(outputLayer[j], hiddenLayer, hiddenLayerSize));
            printf("Output from Node #%i of : %0.2f\n", j, outputLayer[j]->output);
        }

        
        //Training time baby
        double totalError = baseErrorCalc(xorFunction(inputLayer, inputLayerSize), outputLayer[0]->output);
        printf("Error: %0.1f\n", totalError);
        if(xorFunction(inputLayer, inputLayerSize) == outputLayer[0]->output && iterations > (2*iterations/3)) {
            countCorrect++;
            printf("Correct\n");
        }
        for(int j = 0; j < outputLayerSize; j++){
            errorCalc(outputLayer[j], totalError, 1, 1);
        }
        for(int j = 0; j < hiddenLayerSize; j++){
            errorCalc(hiddenLayer[j], outputLayer[0]->error, outputLayer[0]->weights[j+1],sumWeight(outputLayer[0]));
        }
        for(int j = 0; j < inputLayerSize; j++){
            //errorCalc(inputLayer[j],)
        }

        weightUpdateNode(outputLayer[0],hiddenLayer);
        for(int j = 0; j < hiddenLayerSize; j++){
            weightUpdateNode(hiddenLayer[j], inputLayer);
        }
        free(inputsList);
    }
    printf("Percent correct = %.2f\n", (countCorrect/(double)iterations) * 100);

    
    for(int i = 0; i < hiddenLayerSize; i ++) {
        free(inputLayer[i]);
        free(hiddenLayer[i]->weights);
        free(hiddenLayer[i]);
    }
    for(int i = 0; i < outputLayerSize; i ++) {
        free(outputLayer[i]->weights);
        free(outputLayer[i]);
    }

    return 0;
}