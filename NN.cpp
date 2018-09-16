#include "NN.h"

NN::NN(std::vector<size_t> nbNeurones)
{
  // By default, sigmoid is used
  (void) nbNeurones;


}

void NN::setInputs(std::vector<float> inputs)
{
  (void) inputs;
}

std::vector<float> NN::getOutputs(void)
{
  std::vector<float> outputs;

  outputs.push_back(0.85);

  return outputs;
}
