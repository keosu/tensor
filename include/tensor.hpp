#pragma once

#include <exception>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

namespace yi {

template <class T>
class Tensor {
 public:
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

  auto shape() const { return shp_; }
  auto size() const { return std::accumulate(shp_.begin(), shp_.end(), 1, std::multiplies<int>()); }
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto cbegin() const { return data_.begin(); }
  auto cend() const { return data_.end(); }

 private:
  std::vector<T> data_;
  std::vector<int> shp_;
};
}  // namespace yi

namespace st {

/*********************
 * initializer_shape *
 *********************/
template <std::size_t I>
struct initializer_shape_impl {
  template <class T>
  static constexpr std::size_t value(T t) {
    return t.size() == 0 ? 0 : initializer_shape_impl<I - 1>::value(*t.begin());
  }
};

template <>
struct initializer_shape_impl<0> {
  template <class T>
  static constexpr std::size_t value(T t) {
    return t.size();
  }
};

template <class R, class U, std::size_t... I>
constexpr R initializer_shape(U t, std::index_sequence<I...>) {
  using size_type = typename R::value_type;
  return {size_type(initializer_shape_impl<I>::value(t))...};
}
/*********************
 * initializer_depth *
 *********************/
template <class U>
struct initializer_depth {
  static constexpr std::size_t value = 0;
};

template <class T>
struct initializer_depth<std::initializer_list<T>> {
  static constexpr std::size_t value = 1 + initializer_depth<T>::value;
};

/*********************
 * initializer_shape *
 *********************/
template <class R, class T>
constexpr R deduce_shape(T t) {
  return initializer_shape<R, decltype(t)>(
      t, std::make_index_sequence<initializer_depth<decltype(t)>::value>());
}

/***************************
 * nested_initializer_list *
 ***************************/
template <class T, std::size_t I>
struct nested_initializer_list {
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
inline void nested_copy(T &&iter, const S &s) {
  *iter++ = s;
}
template <class T, class S>
inline void nested_copy(T &&iter, std::initializer_list<S> s) {
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

//-------------------------------------------------------------------
// Tensor iterator
//-------------------------------------------------------------------
template <typename T>
class TensorIterator
    : public std::iterator<std::random_access_iterator_tag, T, ptrdiff_t, T *, T &> {
 public:
  TensorIterator(T *ptr = nullptr) { ptr = ptr; }
  TensorIterator(const TensorIterator<T> &iter) = default;
  ~TensorIterator() {}

  TensorIterator<T> &operator=(const TensorIterator<T> &iter) = default;
  TensorIterator<T> &operator=(T *ptr) {
    ptr = ptr;
    return (*this);
  }

  operator bool() const {
    if (ptr)
      return true;
    else
      return false;
  }

  bool operator==(const TensorIterator<T> &iter) const { return (ptr == iter.getConstPtr()); }
  bool operator!=(const TensorIterator<T> &iter) const { return (ptr != iter.getConstPtr()); }

  TensorIterator<T> &operator+=(const ptrdiff_t &movement) {
    ptr += movement;
    return (*this);
  }
  TensorIterator<T> &operator-=(const ptrdiff_t &movement) {
    ptr -= movement;
    return (*this);
  }
  TensorIterator<T> &operator++() {
    ++ptr;
    return (*this);
  }
  TensorIterator<T> &operator--() {
    --ptr;
    return (*this);
  }
  TensorIterator<T> operator++(int) {
    auto temp(*this);
    ++ptr;
    return temp;
  }
  TensorIterator<T> operator--(int) {
    auto temp(*this);
    --ptr;
    return temp;
  }
  TensorIterator<T> operator+(const ptrdiff_t &movement) {
    auto oldPtr = ptr;
    ptr += movement;
    auto temp(*this);
    ptr = oldPtr;
    return temp;
  }
  TensorIterator<T> operator-(const ptrdiff_t &movement) {
    auto oldPtr = ptr;
    ptr -= movement;
    auto temp(*this);
    ptr = oldPtr;
    return temp;
  }

  ptrdiff_t operator-(const TensorIterator<T> &iter) {
    return std::distance(iter.getPtr(), this->getPtr());
  }

  T &operator*() { return *ptr; }
  const T &operator*() const { return *ptr; }
  T *operator->() { return ptr; }

  T *getPtr() const { return ptr; }
  const T *getConstPtr() const { return ptr; }

 protected:
  T *ptr;
};

/**
 * Shape class
 */
class Shape {
 public:
  using value_type = int;
  Shape() = default;  // scalar
  Shape(const Shape &sh) {
    dim_.resize(sh.rank());
    std::vector<int> vec = sh.get();
    dim_.swap(vec);
  }
  Shape &operator=(const Shape &sh) {
    dim_.resize(sh.rank());
    std::vector<int> vec = sh.get();
    dim_.swap(vec);
    return *this;
  }

  Shape(std::initializer_list<int> list) {
    for (auto item : list) {
      if (item <= 0) throw(std::invalid_argument("shape size should greater than 0"));
      dim_.emplace_back(item);
    }
  }
  int &operator[](int index) {
    if (index < -rank() || index >= rank()) throw(std::out_of_range("invalid shape index"));
    if (index < 0) index += rank();
    return dim_[index];
  }

  std::vector<int> get() const { return dim_; }

  void set(std::vector<int> &&vec) { dim_.swap(vec); }

  friend std::ostream &operator<<(std::ostream &out, Shape &shape) {
    auto &dim = shape.dim_;
    out << "Shape(";
    for (auto i = 0; i < dim.size(); i++) out << shape[i] << ", ";
    out << ")";
    return out;
  }
  int rank() const { return dim_.size(); }
  int size() const { return std::accumulate(dim_.begin(), dim_.end(), 1, std::multiplies<int>()); }

 private:
  std::vector<int> dim_;
};

/**
 * Tensor class
 */
template <typename T>
class Tensor {
 public:
  Tensor(Shape &sh) : shape_(sh) { data_.resize(sh.size(), 0); }
  Tensor(Shape &&sh) : shape_(sh) { data_.resize(sh.size(), 0); }

  template <class U>
  void inline init(U list) {
    shape_ = deduce_shape<Shape>(list);
    data_.resize(shape_.size(), 0);
    nested_copy(data_.begin(), list);
  }
  Tensor(const nested_initializer_list_t<T, 0> list) { init(list); }
  Tensor(const nested_initializer_list_t<T, 1> list) { init(list); }
  Tensor(const nested_initializer_list_t<T, 2> list) { init(list); }
  Tensor(const nested_initializer_list_t<T, 3> list) { init(list); }
  Tensor(const nested_initializer_list_t<T, 4> list) { init(list); }
  Tensor(const nested_initializer_list_t<T, 5> list) { init(list); }

  Shape &shape() { return shape_; }
  T &operator[](int index) {
    if (index < -shape_.size() || index >= shape_.size())
      throw(std::out_of_range("Tensor index out of range " + std::to_string(index)));
    if (index < 0) index = shape_.size() + index;
    return data_[index];
  }
  T &operator()(std::initializer_list<int> lst) {
    if (lst.size() != 1 && lst.size() != shape_.rank())
      throw(std::invalid_argument("list size should be equal to 1 or the shape rank"));

    if (lst.size() == 1) {
      int index = *lst.begin();
      if (index < -shape_.size() || index >= shape_.size())
        throw(std::out_of_range("Tensor index out of range " + std::to_string(index)));
      if (index < 0) index = shape_.size() + index;
      return data_[index];

    } else {
      int dim = 0;
      int index = 0;
      for (auto v = lst.begin(); v != lst.end(); v++) {
        if (*v >= shape_[dim] || *v < -shape_[dim])
          throw(std::out_of_range("Tensor index out of range "));
        int tmp = *v >= 0 ? *v : shape_[dim] + *v;
        index = index * ((dim == 0) ? 0 : shape_[dim]) + tmp;
        dim++;
      }
      return data_[index];
    }
  }

  /**
   * elementwise add
   */
  auto &add(T val) {
    for (auto &ele : data_) ele += val;
    return *this;
  }
  auto &operator+(T val) { return add(val); }

  /**
   * elementwise multiply
   */
  auto &mul(T val) {
    for (auto &ele : data_) ele *= val;
    return *this;
  }
  auto &operator*(T val) { return mul(val); }
  /**
   * reshape tensor
   */
  void reshape(const Shape &sh) {
    if (shape_.size() != sh.size()) throw(std::invalid_argument("invalid shape"));
    shape_ = sh;
  }

  /**
   * transpose a 2-d tensor
   */
  void transpose() {
    if (shape_.rank() != 2) throw(std::invalid_argument("transpose only available for 2-d tensor"));
    std::vector<T> vec(data_.size());
    for (auto i = 0; i < data_.size(); i++) {
      int x1 = i / shape_[0], y1 = i % shape_[0];
      int idx = y1 * shape_[1] + x1;
      vec[i] = data_[idx];
    }
    std::copy(vec.begin(), vec.end(), data_.begin());
    std::swap(shape_[0], shape_[1]);
  }
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }

  friend std::ostream &operator<<(std::ostream &out, const Tensor &t) {
    auto &tensor = const_cast<Tensor &>(t);
    auto &shape = tensor.shape_;
    auto &data = tensor.data_;
    out << "Tensor: " << shape << " [\n";
    auto p = data.begin();
    while (p != data.end()) {
      out << "  ";
      std::copy(p, p + shape[-1], std::ostream_iterator<T>(out, ", "));
      out << "\n";
      p += shape[-1];
    }
    out << "]";
    return out;
  }

 protected:
  std::vector<T> data_;
  Shape shape_;
};
};  // end of namespace st

namespace st {
/**
 * generate an all-zero tensor with given shape
 */
template <typename T>
Tensor<T> zero(Shape &&sh) {
  return Tensor<T>(sh);
}
/**
 * generate an all-one tensor with given shape
 */
template <typename T>
Tensor<T> one(Shape &&sh) {
  auto t = Tensor<T>(sh);
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
 * matmul of two 2-d tensors
 */
template <typename T>
Tensor<T> matmul(Tensor<T> &a, Tensor<T> &b, bool transpose = false) {
  if (a.shape().rank() != 2 || b.shape().rank() != 2)
    throw(std::invalid_argument("Only matrix is supported"));
  // if (a.shape()[1] != b.shape()[transpose?1:0]);
  //   throw(std::invalid_argument("Invalid matrix demension"));

  Shape sh{a.shape()[0], b.shape()[transpose ? 0 : 1]};
  Tensor<T> t{sh};
  for (auto i = 0; i < t.shape()[0]; i++) {
    for (auto j = 0; j < t.shape()[1]; j++) {
      T sum{0};
      for (auto k = 0; k < a.shape()[1]; k++) {
        sum += a({i, k}) * b({k, j});
      }
      t({i, j}) = sum;
    }
  }
  return t;
}

/**
 * Dot product of two arrays
 * For 2-D arrays it is equivalent to matrix multiplication, and for 1-D arrays to inner product of
 * vectors.
 */
template <typename T>
Tensor<T> dot(Tensor<T> &a, Tensor<T> &b, bool transpose = false) {
  if (a.shape().rank() == 1 && b.shape().rank() == 1) {
    auto dotProduct = std::inner_product(a.begin(), a.end(), b.begin(), 0);
    return Tensor<T>{dotProduct};
  } else if (a.shape().rank() == 2 && b.shape().rank() == 2) {
    return matmul(a, b);
  } else {
    throw(std::invalid_argument("Only 1-D or 2-D array are supported"));
  }
}

};  // end of namespace st
