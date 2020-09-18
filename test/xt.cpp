#include <NumCpp.hpp>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <any>
#include <array>

// types have not member type value_type
template <typename T, typename = void>
struct has_value_type : std::false_type {};
// types have member type value_type
template <typename T>
struct has_value_type<T, std::void_t<typename T::value_type>> : std::true_type {
};

// return nested times as 0 for types without member type value_type
template <typename T>
constexpr std::enable_if_t<!has_value_type<T>::value, size_t>
get_nested_times() {
  return 0;
}
// return nested times as 1 plus times got on the nested type recursively
template <typename T>
constexpr std::enable_if_t<has_value_type<T>::value, size_t>
get_nested_times() {
  return 1 + get_nested_times<typename T::value_type>();
}

template <typename List>
auto some_function(const List &list) {
  // N is the number of times the list is nested.
  constexpr auto N = get_nested_times<List>();
  // std::array<size_t, N> arr;
  return N;
}

void test_xtensor() {
  xt::xarray<double> arr1{{1.0, 2.0, 3.0}, {2.0, 5.0, 7.0}, {2.0, 5.0, 7.0}};

  xt::xarray<double> arr2{5.0, 6.0, 7.0};

  xt::xarray<double> res = xt::view(arr1, 1) + arr2;

  // std::cout << arr1;

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





// template <typename T, std::size_t... T_dimensions>
// class MultidimensionalArray {
//   static_assert(sizeof...(T_dimensions) > 0,
//                 "At least one dimension needs to be defined.");

//  public:
//   static constexpr std::size_t size() { 
//     return 123; 
//      }

//   // Single curly braces syntax.
//   MultidimensionalArray(typename NestedInitializerLists<T, 1>::type values) {
//     initialize<size()>(values);
//   }

//   using NestedInitializerLists =
//       NestedInitializerListsT<T, sizeof...(T_dimensions)>;

//   // Nested curly braces syntax.
//   MultidimensionalArray(NestedInitializerLists values) {
//     initialize<T_dimensions...>(values);
//   }

//  private:
//   template <std::size_t... T_shape, typename T_NestedInitializerLists>
//   void initialize(T_NestedInitializerLists values) {
//     auto iterator = _data.begin();
//     NestedInitializerListsProcessor<T, T_shape...>::process(
//         values, [&iterator](T value) { *(iterator++) = value; });
//   }

//   std::array<T, size()> _data;
// };

template <typename T, std::size_t T_levels>
struct NestedInitializerLists {
  using type = std::initializer_list<
      typename NestedInitializerLists<T, T_levels - 1>::type>;
};

template <typename T>
struct NestedInitializerLists<T, 0> {
  using type = T;
};

template <typename T, std::size_t I>
using NestedInitializerListsT = typename NestedInitializerLists<T, I>::type; 

int main(int argc, char **argv) {
  // test_xtensor();
  // test_numcpp();

NestedInitializerListsT<float, 3> values = {
    {
        {0, 1},
        {2, 3}
    },
    {
        {4, 5},
        {6, 7}
    }
};

for (auto secondLevel : values) {
    for (auto thirdLevel : secondLevel) {
        for (auto value : thirdLevel) {
            std::cout << value << " ";
        }
    }
}
 

  return 0;
}