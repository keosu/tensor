// main.cpp
#include "tensor.hpp"

#include <iostream>

using namespace ts;

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
                   },
                   {
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

  std::cout << aa;
  std::cout << a;
  std::cout << a.add(1).mul(2);
  std::cout << b;

  auto t = Tensor<int>(Shape{3, 4, 5});
  std::cout << t;
  auto t2 = one<int>(Shape{3, 4, 5});
  std::cout << t2;
}

int main(void) {
  test();
  return 0;
}
