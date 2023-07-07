#pragma once

#include <exception>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

namespace ts {

using Shape = std::vector<int>;

template <class T>
class Tensor {
 public:
  Tensor(Shape shp) {
    shp_.swap(shp);
    data_.resize(size());
  }

  Tensor(std::initializer_list<T> lst) {
    shp_.emplace_back(lst.size());
    data_.resize(lst.size());
    std::copy(lst.begin(), lst.end(), data_.begin());
  }
  Tensor(std::initializer_list<Tensor<T>> lst) {
    auto dim0 = lst.begin()->shape()[0];
    for (auto t = lst.begin() + 1; t != lst.end(); t++) {
      if (t->shape()[0] != dim0) {
        throw("dim should be the same");
      }
    }
    data_.resize(lst.begin()->size() * lst.size());
    auto dst = data_.begin();
    for (auto t = lst.begin(); t != lst.end(); t++) {
      std::copy(t->cbegin(), t->cend(), dst);
      dst += lst.begin()->size();
    }

    shp_.emplace_back(lst.size());
    for (auto d : lst.begin()->shape()) shp_.emplace_back(d);
  }

  auto &shape() const { return shp_; }
  auto &data() const { return data_; }
  auto size() const { return std::accumulate(shp_.begin(), shp_.end(), 1, std::multiplies<int>()); }
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto cbegin() const { return data_.begin(); }
  auto cend() const { return data_.end(); }

  /**
   * add
   */
  auto &add(T val) {
    for (auto &ele : data_) ele += val;
    return *this;
  }
  auto &operator+(T val) { return add(val); }

  /**
   * multiply
   */
  auto &mul(T val) {
    for (auto &ele : data_) ele *= val;
    return *this;
  }
  auto &operator*(T val) { return mul(val); }

  /**
   * sub
   */
  auto &sub(T val) { return add(-val); }
  auto &operator-(T val) { return add(-val); }

  /**
   * div
   */
  auto &div(T val) {
    for (auto &ele : data_) ele /= val;
    return *this;
  }
  auto &operator/(T val) { return div(val); }

  /**
   * reshape tensor
   */
  void reshape(const std::vector<int> &sh) {
    auto new_size = std::accumulate(sh.begin(), sh.end(), 1, std::multiplies<int>());
    auto old_size = std::accumulate(shp_.begin(), shp_.end(), 1, std::multiplies<int>());
    if (new_size != old_size) throw(std::invalid_argument("invalid shape"));

    shp_.clear();
    for (auto d : sh) shp_.emplace_back(d);
  }

  friend std::ostream &operator<<(std::ostream &out, const Tensor &t) {
    auto &shape = t.shape();
    auto &data = t.data();
    out << "Tensor (";
    std::copy(shape.begin(), shape.end(), std::ostream_iterator<T>(out, ", "));
    out << ")\n";

    std::copy(data.begin(), data.end(), std::ostream_iterator<T>(out, ", "));
    out << "\n";

    return out;
  }

 private:
  std::vector<T> data_;
  std::vector<int> shp_;
};

/**
 * generate an all-zero tensor with given shape
 */
template <typename T>
Tensor<T> zero(Shape shp) {
  auto t = Tensor<T>(shp);
  std::fill(t.begin(), t.end(), 0);
  return t;
}
/**
 * generate an all-one tensor with given shape
 */
template <typename T>
Tensor<T> one(Shape shp) {
  auto t = Tensor<T>(shp);
  std::fill(t.begin(), t.end(), 1);
  return t;
}

/**
 * generate identity matrix
 */
template <typename T>
Tensor<T> eye(int n) {
  auto t = Tensor<T>(Shape{n, n});
  for (int i = 0; i < n * n; i += n + 1) t[i] = 1;
  return t;
}

/**
 * generate identity matrix
 */
template <typename T>
Tensor<T> range(int n) {
  auto t = Tensor<T>(Shape{n});
  for (int i = 0; i < n; i++) t[i] = 1;
  return t;
}

}  // namespace ts
