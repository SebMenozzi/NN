#include <iostream>
#include <math.h>
#include "matrix.h"

#define nb_neurons_hidden_layer 3
#define epoch 1
#define learning_rate 0.8

// GLOBAL
const int training_data[4][2] = {
    { 1, 0 },
    { 1, 1 },
    { 0, 1 },
    { 0, 0 }
};
const int anwser_data[4] = { 1, 0, 1, 0 };

int target;

Matrix input_to_hidden_weights(nb_neurons_hidden_layer, 2);
Matrix input_to_hidden_bias(nb_neurons_hidden_layer, 1);
Matrix input_layer(2, 1);

Matrix hidden_to_output_weights(1, nb_neurons_hidden_layer);
Matrix hidden_to_output_bias(1, 1);
Matrix hidden_layer(nb_neurons_hidden_layer, 1);

Matrix output_layer(1, 1);

float error_rate; // for debug

// UTILS

double random_between_neg_1_pos_1() {
    int randNumber = rand() % 2; // 0 or 1
    double result = rand() / (RAND_MAX + 1.0); // between 0 and 1
    return (randNumber == 1) ? result : -1 * result;
}

void generate_random_input_to_hidden_weights() {
    for (size_t row = 0; row < nb_neurons_hidden_layer; row++) {
        for (size_t column = 0; column < 2; column++)
        {
            input_to_hidden_weights.setValue(row, column, random_between_neg_1_pos_1());
        }
    }
}

void generate_random_hidden_to_output_weights() {
    for (size_t row = 0; row < nb_neurons_hidden_layer; row++)
    {
        hidden_to_output_weights.setValue(0, row, random_between_neg_1_pos_1());
    }
}

// activation function
float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

// derivative of activation function
float sigmoid_prime(float x)
{
    return x * (1.0 - x);
}

Matrix applySigmoid(Matrix a) {
  for (size_t column = 0; column < a.getHeight(); column++) {
    for (size_t row = 0; row < a.getWidth(); row++) {
      a.setValue(column, row, sigmoid(a.getValue(column, row)));
    }
  }
  return a;
}

// FORWARD PROPAGATION

void forward_propagate() {
    hidden_layer = applySigmoid(input_to_hidden_weights * input_layer + input_to_hidden_bias);
    output_layer = applySigmoid(hidden_to_output_weights * hidden_layer + hidden_to_output_bias);
}
/*
// BACKWARD PROPAGATION
void back_propagate() {
    float error = output_layer - target;
    error_rate = pow(error, 2);

    for (int i = 0; i < nb_neurons_hidden_layer; i++) {
        for (int j = 0; j < 2; j++) {
            float deltaw1 = error *
                            sigmoid_prime(output_layer) *
                            hidden_to_output_weights[i] *
                            sigmoid_prime(hidden_layer[i]) *
                            input_layer[j];
            // new input weights
            input_to_hidden_weights[i][j] -= deltaw1 * learning_rate;
        }

        float deltaw2 = error *
                        sigmoid_prime(output_layer) *
                        hidden_layer[i];
        // new hidden weights
        hidden_to_output_weights[i] -= deltaw2 * learning_rate;

        float deltab1 = error *
                        sigmoid_prime(output_layer) *
                        hidden_to_output_weights[i] *
                        sigmoid_prime(hidden_layer[i]);
        // new input bias
        input_to_hidden_bias[i] -= deltab1 * learning_rate;

        float deltab2 = error *
                        sigmoid_prime(output_layer);
        // new hidden bias
        hidden_to_output_bias -= deltab2 * learning_rate;
    }
}

void display_result() {
    input_layer.display();

    std::cout << "(" << input_layer.getValue(0, 0) << ", " << input_layer.getValue(1, 0) << ") : expected: " << target << " (error:" << output_layer.getValue(0, 0) << ")" << std::endl << std::endl;
}
*/
int main() {
    generate_random_input_to_hidden_weights();
    generate_random_hidden_to_output_weights();

    // train the network
    for (int i = 0; i < epoch; i++) {
        for (int inputs = 0; inputs < 1; inputs++) {
            input_layer.setValue(0, 0, training_data[inputs][0]);
            input_layer.setValue(1, 0, training_data[inputs][1]);
            target = anwser_data[inputs];

            forward_propagate();
            //back_propagate();
            //display_result();
        }
    }
    return 0;
}
