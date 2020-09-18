// main.cpp
#include <iostream>

#include "tensor.hpp"

int main(void) {
  Shape sh{4, 5};

  std::cout << sh << std::endl;
  std::cout << sh.rank() << std::endl;
  std::cout << sh.size() << std::endl;
  Shape x = sh;
  Shape y(sh);

  // Tensor<int> t(std::move(sh));

  Tensor<int> t2{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };

  // std::cout << t << std::endl;
   std::cout << t2 << std::endl;

  return 0;
}