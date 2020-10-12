// main.cpp
#include <iostream>

#include "tensor.hpp"

using namespace st;

void test_innertype() {
  nested_initializer_list_t<int, 2> l{
      {1, 2, 3, 5},
      {4, 5, 6, 3},
      {7, 8, 9, 4},
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
}

void test_ele_accessment() {
  Tensor<int> t1{{
      {1, 2, 3, 123},
      {4, 5, 6, 123},
      {7, 8, 9, 123},
  }};

  auto shape = t1.shape();
  std::cout << "t1 " << t1 << std::endl;

  for (auto i = 0; i < t1.shape().size(); i++) {
    std::cout << t1[i] << "\t";
  }
  std::cout << std::endl;

  for (auto i = t1.begin();i != t1.end(); i++) {
    std::cout << *i << "\t";
  }
  std::cout << std::endl;
  
  for (auto i = 0; i < shape[0]; i++) {
    for (auto j = 0; j < shape[1]; j++) {
      for (auto k = 0; k < shape[2]; k++) { 
        std::cout << t1({i, j, k}) << " ";
      }
      std::cout << std::endl;
    }
  }
}

void test_basic() {
  Tensor<int> t1{{
      {1, 2, 3, 123},
      {4, 5, 6, 123},
      {7, 8, 9, 123},
  }};

  Tensor<int> t2{1, 2, 4, 5};

  Shape sh{4, 5};
  auto ze = st::zero<int>({7, 4});
  auto one = st::one<int>({3, 4, 5});

  auto eye = st::eye<int>(5);

  std::cout << ze << std::endl;
  std::cout << one << std::endl;
  std::cout << eye << std::endl;
  std::cout << eye({3, 3}) << std::endl;
}

void test_matmul() {
    Tensor<int> a{
      {1, 2, 3, 5},
      {4, 5, 6, 5},
      {7, 8, 9, 5},
      {10, 11, 12, 5}
  };
  auto x = st::one<int>({4,4});
  std::cout << a << std::endl;
  std::cout << x << std::endl;
  auto m = matmul(a, x);
  std::cout << m << std::endl;
}


void test() {
    Tensor<int> a{
      {1, 2, 3, 4},
      {5, 6, 7, 8}, 
  };

  std::cout << a << std::endl;
  a.add(3); 
  std::cout << a << std::endl;
  a.transpose(); 
  std::cout << a << std::endl;
  a.reshape(Shape{1,8}); 
  std::cout << a << std::endl;
  a.reshape(Shape{8, 1}); 
  std::cout << a << std::endl;
}

int main(void) {
  // test_ele_accessment();
  // test_matmul();
  test();
  return 0;
}