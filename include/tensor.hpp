#pragma once

#include <exception>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <vector>
 

/***************************
 * nested_initializer_list *
 ***************************/
template <class T, std::size_t I>
struct nested_initializer_list  {
    using type = std::initializer_list<typename nested_initializer_list<T, I - 1>::type>;
}; 
template <class T>
struct nested_initializer_list<T, 0> {
    using type = T;
};

template <class T, std::size_t I>
using nested_initializer_list_t = typename nested_initializer_list<T, I>::type; 

/******************************
 * nested_copy implementation *
 ******************************/ 
template <class T, class S>
inline void nested_copy(T&& iter, const S& s)  {
    *iter++ = s;
} 
template <class T, class S>
inline void nested_copy(T&& iter, std::initializer_list<S> s) {
    for (auto it = s.begin(); it != s.end(); ++it) {
        nested_copy(std::forward<T>(iter), *it);
    }
}


/**
 * TreeNode class
 */
template <typename T>
struct TreeNode {
  bool if_leaf;
  T i;
  std::vector<TreeNode> v;

  TreeNode(T i_) : if_leaf(true), i(i_) {}
  TreeNode(std::initializer_list<TreeNode> v_) : if_leaf(false), v(v_) {}
};


/**
 * Shape class
 */
class Shape {
 public:
  Shape() { std::cout << "==> Shape empty constructor" << std::endl; }
  Shape(const Shape &sh) {
    rank_ = sh.rank();
    size_ = sh.size();
    dim_ = new int[sh.rank()];
  }
  Shape &operator=(const Shape &sh) {
    rank_ = sh.rank();
    size_ = sh.size();
    dim_ = new int[sh.rank()];
  }

  Shape(std::initializer_list<int> list) {
    rank_ = list.size();
    dim_ = new int[rank_];
    size_t cnt = 0;
    size_ = 1;
    for (auto item : list) {
      if (item <= 0)
        throw(std::invalid_argument("shape size should greater than 0"));
      dim_[cnt++] = item;
      size_ *= item;
    }
  }
  ~Shape() { delete[] dim_; }

  int operator[](int index) {
    if (index < 0 || index > rank_)
      throw(std::out_of_range("invalid shape index"));
    return dim_[index];
  }

  friend std::ostream &operator<<(std::ostream &out, Shape &shape) {
    out << "Shape(";
    for (auto i = 0; i < shape.rank(); i++) out << shape[i] << ", ";
    out << ")";
    return out;
  }
  int rank() const { return rank_; }
  int size() const { return size_; }

 private:
  size_t rank_;
  size_t size_;
  int *dim_;
};

/**
 * Tensor class
 */
template <typename T>
class Tensor {
 public:
  Tensor(Shape &&sh) {
    shape_ = sh;
    data_ = new T[sh.size() + 1];
  }
  Tensor(const std::initializer_list<std::initializer_list<T> > &list)
      : shape_({(int)list.size(), (int)(list.begin()->size())}) {
    data_ = new T[shape_.size() + 1];
    size_t cnt = 0;
    for (auto l1 : list) {
      for (auto item : l1) {
        data_[cnt++] = item;
      }
    }
  }
  ~Tensor() { delete[] data_; }

  Shape &shape() { return shape_; }
  T &operator[](int index) {
    if (index < 0 || index > shape_.size())
      throw(std::out_of_range("Tensor index out of range " +
                              std::to_string(index)));
    return data_[index];
  }

  friend std::ostream &operator<<(std::ostream &out, Tensor &tensor) {
    out << "Tensor[\n";
    auto dim2 = tensor.shape()[1];
    for (auto i = 0; i < tensor.shape()[0]; i++) {
      out << "  [";
      std::copy(&tensor[dim2 * i], &tensor[dim2 * (i + 1)],
                std::ostream_iterator<T>(out, ", "));
      out << "]\n";
    }
    out << "]";
    return out;
  }

 protected:
  T *data_;
  Shape shape_;
};
