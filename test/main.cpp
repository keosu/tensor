// main.cpp
#include <iostream>

#include "tensor.hpp"

int main(void) {
  nested_initializer_list_t<int,2> l{
      {1, 2, 3,5},
      {4, 5, 6,3},
      {7, 8, 9,4},
  };
  auto ss = deduce_shape<std::vector<int>>(l);
  auto ss2 = deduce_shape<Shape>(l);
  Shape sh{4, 5};

  std::cout << sh << std::endl;
  std::cout << sh.rank() << std::endl;
  std::cout << sh.size() << std::endl;
  Shape x = sh;
  Shape y(sh);

  std::cout << "x,y " << x << " " << y << std::endl;

  // Tensor<int> t(std::move(sh));

  Tensor<int> t1{ {
      {1, 2, 3, 123},
      {4, 5, 6, 123},
      {7, 8, 9, 123},
  }
  };

    Tensor<int> t2{ 1,2,4,5 };


  std::cout << t2 << std::endl;
   std::cout << t1 << std::endl;
   std::cout << t1.shape() << std::endl;

  return 0;
}