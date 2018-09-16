#ifndef NN_H_INCLUDED
#define NN_H_INCLUDED

#include <vector>

class NN
{
private:


public:
  NN(std::vector<size_t> nbNeurones);
  void setInputs(std::vector<float> inputs);
  std::vector<float> getOutputs(void);

};

#endif // NN_H_INCLUDED
