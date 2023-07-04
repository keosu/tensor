// main.cpp
#include "tensor.hpp"

#include <iostream>

using namespace yi;

void test() {
  Tensor<int> aa = {1, 2, 3, 4, 5};

  Tensor<int> a = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {5, 6, 7, 8},
  };
  Tensor<int> b = {{
                       {1, 2},
                       {5, 6},
                   },{
                       {1, 2},
                       {5, 6},
                   },
                   {
                       {1, 2},
                       {5, 6},
                   }};
  auto f = [](auto vec) {
    for (auto i : vec) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  };

  f(aa.shape());
  f(a.shape());
  f(b.shape());
}

int main(void) {
  test();
  return 0;
}
