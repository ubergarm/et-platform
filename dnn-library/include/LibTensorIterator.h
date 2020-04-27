
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
  
  HandleIterator() = delete;
  HandleIterator(const HandleIterator& x) = default;
  
  HandleIterator(Handle<ElemTy>& h, const size_t offset) :
    dims_(h.tensor_->dims()),
    strides_(h.tensor_->strides()),
    ndims_(h.tensor_->ndims()),
    ptr_(h.tensor_->template getRawDataPointer <ElemTy>()),
    offset_(offset),
    coords_(h.tensor_->offset2Coord(offset)),
    wrap_(buildWrap())
  {
    // offset2Coord can return coordinates pointing to padding: adjust to point to first element without offset
    for(size_t i = ndims_-1; i > 0 ; --i) {
      if (coords_[i] >= dims_[i]) {
        offset_+=strides_[i-1] - strides_[i] * coords_[i];
        coords_[i] = 0;
        coords_[i-1]++;
      }
    }
  }
  
  HandleIterator(Handle<ElemTy>& h, const dim_array_t &coords) :
    dims_(h.tensor_->dims()),
    strides_(h.tensor_->strides()),
    ndims_(h.tensor_->ndims()),
    ptr_(h.tensor_->template getRawDataPointer<ElemTy>()),
    offset_(h.getElementPtr(coords)),
    coords_(coords),
    wrap_(buildWrap()) { }
  
  HandleIterator(Handle<ElemTy>& h) :
    dims_(h.tensor_->dims()),
    strides_(h.tensor_->strides()),
    ndims_(h.tensor_->ndims()),
    ptr_(h.tensor_->template getRawDataPointer<ElemTy>()),
    wrap_(buildWrap())  { } // assuming offset=0, coords=0
  
  HandleIterator(Handle<ElemTy> &h, size_t offset, const dim_array_t &coords) :
    dims_(h.tensor_->dims()),
    strides_(h.tensor_->strides()),
    ndims_(h.tensor_->ndims()),
    ptr_(h.tensor_->template getRawDataPointer<ElemTy>()),
    offset_(offset),
    coords_(coords),
    wrap_(buildWrap())  {}
  
  // increment
  HandleIterator& operator++( ) {
    assert(strides_[ndims_-1] == 1);
    offset_++;
    coords_[ndims_ -1]++;
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
  
  // increment.. if going past a dimension, points to start of next dimension
  HandleIterator& operator+=(difference_type x) {
    offset_ +=x;
    assert(strides_[ndims_-1] == 1);
    coords_[ndims_-1]+=x;
    wrapN(ndims_-1);
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
  
  static HandleIterator begin(Handle<ElemTy>& t) { return t.begin();}
  static HandleIterator end(Handle<ElemTy>& t) { return t.end(); }
  
  size_t offset() const { return offset_; }
  const dim_array_t & coords() const { return coords_;}
  
private:
  const dim_array_t &dims_;
  const dim_array_t &strides_;
  const size_t ndims_;
  ElemTy * const ptr_;
  
  size_t offset_ = 0;
  dim_array_t coords_ = {0};
  const dim_array_t wrap_;
  
  dim_array_t buildWrap(){
    dim_array_t w = {0};
    for(size_t i = 0; i < ndims_ - 1;i++) {
      w[i +1 ] = strides_[i] - strides_[i+1] * dims_[i+1];
    }
    return w;
  }

  void wrap1(size_t dim){
    if (dim == 0) return;
    if (coords_[dim] >= dims_[dim]){
      offset_+=wrap_[dim];
      coords_[dim]=0;
      coords_[dim-1]++;
    }
    wrap1(dim-1);
  }
    
  void wrapN(size_t dim){
    if (dim == 0) return;
    if (coords_[dim] >= dims_[dim]){
      offset_+= strides_[dim-1] - coords_[dim];
      coords_[dim]=0;
      coords_[dim-1]++;
    }
    wrap1(dim-1);
  }
};
  
