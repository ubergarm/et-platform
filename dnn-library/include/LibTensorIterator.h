/*-------------------------------------------------------------------------
 * Copyright (C) 2022, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

template<typename ElemTy> class Handle;


template<typename ElemTy>
class HandleIterator {
public:
  // type definitions
  using iterator_category = std::forward_iterator_tag;
  using value_type = ElemTy;
  using difference_type = std::ptrdiff_t;
  using pointer = ElemTy*;
  using reference = ElemTy&;
  using HandleT = Handle<ElemTy>;

  HandleIterator() = delete;
  HandleIterator(const HandleIterator& x) = default;

  HandleIterator(HandleIterator&& x)
    : dims_(x.dims_)
    , strides_(x.strides_)
    , ndims_(x.ndims_)
    , ptr_(x.ptr_)
    , offset_(x.offset_)
    , coords_(x.coords_)
    , wrap_(x.wrap_) {
  }

  HandleIterator(HandleT& h, const dim_t offset)
    : dims_(h.tensor_->dims())
    , strides_(h.tensor_->strides())
    , ndims_(h.tensor_->ndims())
    , ptr_(h.tensor_->template getRawDataPointer<ElemTy>())
    , offset_(offset)
    , coords_(h.tensor_->offset2Coord(offset))
    , wrap_(buildWrap()) {
    // offset2Coord can return coordinates pointing to padding: adjust to point to first element without offset
    for (size_t i = ndims_ - 1; i > 0; --i) {
      if (coords_[i] >= dims_[i]) {
        for (size_t j = ndims_ - 1; j >= i; --j) {
          offset_ += strides_[j - 1] - strides_[j] * coords_[j];
          coords_[j] = 0;
          coords_[j - 1]++;
        }
      }
    }
  }

  HandleIterator(HandleT& h, const dim_array_t& coords)
    : dims_(h.tensor_->dims())
    , strides_(h.tensor_->strides())
    , ndims_(h.tensor_->ndims())
    , ptr_(h.tensor_->template getRawDataPointer<ElemTy>())
    , offset_(h.getElementPtr(coords))
    , coords_(coords)
    , wrap_(buildWrap()) {
  }

  HandleIterator(HandleT& h)
    : dims_(h.tensor_->dims())
    , strides_(h.tensor_->strides())
    , ndims_(h.tensor_->ndims())
    , ptr_(h.tensor_->template getRawDataPointer<ElemTy>())
    , wrap_(buildWrap()) {
  } // assuming offset=0, coords=0

  HandleIterator(HandleT& h, dim_t offset, const dim_array_t& coords)
    : dims_(h.tensor_->dims())
    , strides_(h.tensor_->strides())
    , ndims_(h.tensor_->ndims())
    , ptr_(h.tensor_->template getRawDataPointer<ElemTy>())
    , offset_(offset)
    , coords_(coords)
    , wrap_(buildWrap()) {
  }

  // increment
  HandleIterator& operator++( ) {
    offset_ += strides_[ndims_ - 1];
    coords_[ndims_ - 1]++;
    wrap1(ndims_ - 1);
    return *this;
  }

  HandleIterator& step(size_t dim){
    if ( dim == size_t(-1)){
      offset_ += strides_[0] * dims_[0];
      return *this;
    }
    assert(dim < ndims_);
    coords_[dim]++;
    offset_ += strides_[dim];
    wrap1(dim);
    return *this;
  }

  // increment.. consider only valid elements
  HandleIterator& operator+=(difference_type x) {
    offset_ += x * strides_[ndims_ - 1];
    coords_[ndims_ - 1] += x;
    wrapN(ndims_ - 1);
    return *this;
  }
  
  reference operator[](difference_type n) { return ptr_[offset_ + n]; }
  reference operator*( ) { return ptr_[offset_]; }
  
  bool operator==(const HandleIterator &x){ return offset_ == x.offset_;}
  bool operator!=(const HandleIterator &x){ return offset_ != x.offset_;}
  bool operator<(const HandleIterator &x) { return offset_ < x.offset_;}
  difference_type operator-(const HandleIterator& x) {return offset_ - x.offset_;}
  HandleIterator operator+(difference_type x) {
    return HandleIterator (offset_+ x);
  }

  static HandleIterator begin(HandleT& t) {
    return t.begin();
  }
  static HandleIterator end(HandleT& t) {
    return t.end();
  }

  dim_t offset() const { return offset_; }
  const dim_array_t & coords() const { return coords_;}
  
private:
  const dim_array_t &dims_;
  const dim_array_t &strides_;
  const size_t ndims_;
  ElemTy * const ptr_;
  
  dim_t offset_ = 0;
  dim_array_t coords_ = {0};
  using wrap_t = std::array<dim_t, max_tensor_dimensions - 1>;
  const wrap_t wrap_;

  wrap_t buildWrap() {
    wrap_t w;
    for(size_t i = 0; i < ndims_ - 1;i++) {
      w[i] = strides_[i] - strides_[i + 1] * dims_[i + 1];
    }
    return w;
  }

  void wrap1(size_t dim){
    if (dim == 0) return;
    if (coords_[dim] >= dims_[dim]){
      offset_ += wrap_[dim - 1];
      coords_[dim]=0;
      coords_[dim-1]++;
    }
    wrap1(dim-1);
  }
    
  void wrapN(size_t dim){
    if (dim == 0) return;
    if (coords_[dim] >= dims_[dim]){
      offset_ -= coords_[dim] * strides_[dim];
      dim_t carry = coords_[dim] / dims_[dim];
      coords_[dim] = coords_[dim] % dims_[dim];
      coords_[dim - 1] += carry;
      offset_ += coords_[dim] * strides_[dim] + carry * strides_[dim - 1];
    }
    wrapN(dim - 1);
  }
};
  
