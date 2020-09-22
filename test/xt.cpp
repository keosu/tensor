#include <NumCpp.hpp>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <any>
#include <array>

 
void test_xtensor() {
  xt::xarray<double> arr1{{1.0, 2.0, 3.0}, {2.0, 5.0, 7.0}, {2.0, 5.0, 7.0}};

  xt::xarray<double> arr2{5.0, 6.0, 7.0};

  xt::xarray<double> res = xt::view(arr1, 1) + arr2;

  std::cout << arr1;

  xt::xarray<double> arr3{{{1,2,3,4},{1,2,3}},{{1,2}}}; //{{{{{{ 1,2 }}}}}}; 
  std::cout << arr3;
}

void test_numcpp() {
  nc::NdArray<int> a = {{1, 2}, {3, 4}, {5, 6}};
  auto b = nc::linspace<int>(1, 10, 5);

  auto c = nc::eye<float>(4);

  nc::Shape s{1, 2};
  auto e(s);
  auto d = s;

  std::cout << a << std::endl;
}
 

int main(int argc, char **argv) {
  test_xtensor();
  // test_numcpp(); 

  return 0;
}