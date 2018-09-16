#include <iostream>
#include <math.h>
#include "matrix.h"

// nombre de neurones en hidden layer
#define nb_neurons_hidden_layer 3
// nombre d'itération
#define epoch 2000
// capacité d'apprentissage
#define learning_rate 0.7

// GLOBAL
// donnée entrante d'input pour XOR
const int training_data[4][2] = {
    { 1, 0 }, // 1 XOR 1 = 1
    { 1, 1 }, // 0
    { 0, 1 }, // 1
    { 0, 0 } // 0
};
// résultat attendu
const int anwser_data[4] = { 1, 0, 1, 0 };

// objectif
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

// donne un float entre -1 et 1
double random_between_neg_1_pos_1() {
    int randNumber = rand() % 2; // 0 or 1
    double result = rand() / (RAND_MAX + 1.0); // between 0 and 1
    return (randNumber == 1) ? result : -1 * result;
}

// met des poids aléatoires de input vers hidden layer
void generate_random_input_to_hidden_weights() {
    for (size_t row = 0; row < nb_neurons_hidden_layer; row++) {
        for (size_t column = 0; column < 2; column++)
        {
            input_to_hidden_weights.setValue(row, column, random_between_neg_1_pos_1());
        }
    }
}

// met des poids aléatoires de hidden vers output layer
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

// applique la fonction sigmoide pour chaque case d'une matrice
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
// BACKWARD PROPAGATION
void back_propagate() {
    float error = output_layer.getValue(0, 0) - target;
    error_rate = pow(error, 2);

    for (int row = 0; row < nb_neurons_hidden_layer; row++) {
        for (int column = 0; column < 2; column++) {
            float deltaw1 = error *
                            sigmoid_prime(output_layer.getValue(0, 0)) *
                            hidden_to_output_weights.getValue(0, row) *
                            sigmoid_prime(hidden_layer.getValue(row, 0)) *
                            input_layer.getValue(column, 0);
            // new input weights
            float new_input_to_hidden_weights = input_to_hidden_weights.getValue(row, column) - deltaw1 * learning_rate;
            input_to_hidden_weights.setValue(row, column, new_input_to_hidden_weights);
        }

        float deltaw2 = error *
                        sigmoid_prime(output_layer.getValue(0, 0)) *
                        hidden_layer.getValue(row, 0);
        // new hidden weights
        float new_hidden_to_output_weights = hidden_to_output_weights.getValue(0, row) - deltaw2 * learning_rate;
        hidden_to_output_weights.setValue(0, row, new_hidden_to_output_weights);

        float deltab1 = error *
                        sigmoid_prime(output_layer.getValue(0, 0)) *
                        hidden_to_output_weights.getValue(0, row) *
                        sigmoid_prime(hidden_layer.getValue(row, 0));
        // new input bias
        float new_input_to_hidden_bias = input_to_hidden_bias.getValue(row, 0) - deltab1 * learning_rate;
        input_to_hidden_bias.setValue(row, 0, new_input_to_hidden_bias);

        float deltab2 = error *
                        sigmoid_prime(output_layer.getValue(0, 0));

        // new hidden bias
        float new_hidden_to_output_bias = hidden_to_output_bias.getValue(0, 0) - deltab2 * learning_rate;
        hidden_to_output_bias.setValue(0, 0, new_hidden_to_output_bias);
    }
}

void display_result() {
    std::cout << "(" << input_layer.getValue(0, 0) << ", " << input_layer.getValue(1, 0) << ") : expected: " << target << " (error:" << output_layer.getValue(0, 0) << ")" << std::endl;
}

int main() {
    generate_random_input_to_hidden_weights();
    generate_random_hidden_to_output_weights();

    // pour chaque itération (2000)
    for (int i = 0; i < epoch; i++) {
        // pour chaque données à apprendre ( (0, 0) , (1, 0) , (0, 1) , (1, 1) pour XOR)
        for (int inputs = 0; inputs < 4; inputs++) {
            // on initialize le premier neurone de input layer
            input_layer.setValue(0, 0, training_data[inputs][0]);
            // on initialize le second neurone de input layer
            input_layer.setValue(1, 0, training_data[inputs][1]);
            // on initialize la valeur target correspond au résulat attendu par rapport aux neurones de input layer (0 ou 1 pour XOR)
            target = anwser_data[inputs];

            forward_propagate();
            back_propagate();
            display_result();
        }
    }

    // TEST
    input_layer.setValue(0, 0, 0.01); // Entree 1
    input_layer.setValue(1, 0, 0.99); // Entree 2
    // Calcule
    forward_propagate();
    // Affichage
    std::cout << "Sortie : " << output_layer.getValue(0, 0) << std::endl;
    // Calcule
    forward_propagate();
    // Affichage
    std::cout << "Sortie : " << output_layer.getValue(0, 0) << std::endl;

    return 0;
}
