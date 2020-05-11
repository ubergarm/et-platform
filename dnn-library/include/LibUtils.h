#ifndef _LIB_UTILS_H_
#define _LIB_UTILS_H_

#include "LibTypes.h"
#include <utility>
#include "Float16.h"

namespace dnn_lib{



    /*@brief aliases, forward declarations, and misc constants
   */
  constexpr unsigned max_tensor_dimensions = 6;

  constexpr ElemKind IndexElemKind =
    (sizeof(dim_t) == 4) ? ElemKind::Int32ITy : ElemKind::Int64ITy;
  
  struct Type;
  
  using TypeRef = const Type *;
  
  using float16_t = float16;

  template <class ElemTy> class Handle;
  
  template<ElemKind elK>
  struct elemKind2elemTy {
    using type =
      typename std::conditional<elK == ElemKind::FloatTy, float,
       typename std::conditional<elK == ElemKind::Float16Ty, float16,
        typename std::conditional<elK == ElemKind::Int8QTy, int8_t,
         typename std::conditional<elK == ElemKind::UInt8QTy, uint8_t,
          typename std::conditional<elK == ElemKind::Int16QTy, int16_t,
           typename std::conditional<elK == ElemKind::Int32QTy, int32_t,
            typename std::conditional<elK == ElemKind::Int32ITy, int32_t,
             typename std::conditional<elK == ElemKind::Int64ITy, int64_t,
              typename std::conditional<elK == ElemKind::UInt8FusedQTy, uint8_t,
               typename std::conditional<elK == ElemKind::UInt8FusedFP16QTy, uint8_t,
                typename std::conditional<elK == ElemKind::UInt4FusedFP16QTy, uint8_t,
                 typename std::conditional<elK == ElemKind::BoolTy, bool,
                      void // void is the default value, if no elKind matches
            >::type >::type >::type >::type >::type> ::type
      >::type >::type >::type >::type >::type>::type;
    
  //@TODO static_assert(!std::is_same<type, void>::value);
  };

  using dim_array_t = std::array<dim_t, max_tensor_dimensions>;
  using sdim_array_t = std::array<sdim_t, max_tensor_dimensions>;
 

  
  // TODO: add comments/documentation


  template<size_t ndims = max_tensor_dimensions, size_t first = 0>
  struct dims_loop{

    // loop with strides from 1 tensor
    template<typename func_t, size_t N>
    static inline void run(const std::array<dim_t, N> &dims, const std::array<dim_t,N> &strides, func_t fnc, dim_t base = 0 ){
      static_assert(N<=ndims);
      for ( dim_t i = 0 ; i < dims[first]; i++)
        dims_loop<ndims, first+1>::run (dims, strides, fnc, base + strides[first] * i );
    }

    // loop with strides from 2 tensors
    template<typename func_t, size_t N, size_t ST1, size_t ST2>
    static inline void run(const std::array<dim_t, N> &dims,
                        const std::array<dim_t, ST1> &strides1,
                        const std::array<dim_t, ST2> &strides2,
                        func_t fnc, dim_t base1 = 0, dim_t base2 = 0 ){
      static_assert(N<=ndims);
      static_assert(ST1>=N && ST2 >=N);
      for ( dim_t i = 0 ; i < dims[first]; i++)
        dims_loop<ndims, first+1>::run (dims, strides1, strides2,  fnc,
                                        base1 + strides1[first] * i,
                                        base2 + strides2[first] * i);
    }
    

#if 0
    // loop with strides from 2 tensors with initial and end coordinates
    template<typename func_t, size_t N, size_t ST1, size_t ST2>
    static inline void run(const std::array<dim_t, N> &dims,
                           const std::array<dim_t,ST1> &strides1, const std::array<dim_t,ST2> &strides2,
                           const std::array<dim_t, N> &start,  const std::array<dim_t, N> &end, 
                           func_t fnc, dim_t base1 = 0, dim_t base2 = 0,
                           bool firstStep = true, bool lastStep = false){

      // loop until the end of the dimension for all the outer dimensions
      dim_t count = first == 0 || lastStep ? end[first] : // stop when reaching the target dimension
        dims[first];  // and complete all other iterations
      dim_t ini= firstStep ? start[first]  : 0;
      if ( ini > count) return; //nothing else to do
      for ( dim_t i = ini ; i < count; i++) {
        dims_loop<ndims, first+1>::run (dims, strides1, strides2,
                                        start, dims, fnc,
                                        base1 + strides1[first] * i,
                                        base2 + strides2[first] * i,
                                        firstStep, false);
        firstStep = false;
      }

      
      // and remaining portions
      if ((first == 0 || lastStep) && first != N-1) {
        dims_loop<ndims, first+1>::template run(dims, strides1, strides2,
                                                start, end, fnc,
                                                base1 + strides1[first] * end[first],
                                                base2 + strides2[first] * end[first],
                                                firstStep, true);
      }
    }
#else
    // loop with strides from 2 tensors with initial and end coordinates
    template<typename func_t, size_t N, size_t ST1, size_t ST2>
    static inline void run(const std::array<dim_t, N> &dims,
                           const std::array<dim_t,ST1> &strides1, const std::array<dim_t,ST2> &strides2,
                           const std::array<dim_t, N> &start,  const dim_t end, 
                           func_t fnc, dim_t base1 = 0, dim_t base2 = 0,
                           bool firstStep = true, bool lastStep = false){
      
      dim_t ini= firstStep ? start[first]  : 0;

      for ( dim_t i = ini ; i < dims[first] && base1 + strides1[first] *i < end; i++){
        dims_loop<ndims, first+1>::run (dims, strides1, strides2,
                                        start, end, fnc,
                                        base1 + strides1[first] * i,
                                        base2 + strides2[first] * i,
                                        firstStep, false);
        firstStep = false;
      }
    }
#endif
  };
  
  template<size_t last_dim>
  struct dims_loop<last_dim, last_dim>{
    
    // loop with strides from 1 tensor
    template<typename func_t, size_t N>
    static inline void run(const std::array<dim_t, N> &, const std::array<dim_t, N> &, func_t fnc, dim_t base =0 ){
      fnc(base);
    }
    
    
    // loop with strides from 2 tensors
    template<typename func_t, size_t N, size_t ST1, size_t ST2>
    static inline void run(const std::array<dim_t, N> &,
                           const std::array<dim_t, ST1> &, const std::array<dim_t, ST2> &,
                           func_t fnc, dim_t base1 = 0, dim_t base2 = 0 ){
      fnc(base1, base2);
    }


#if 0
    // loop with strides from 2 tensors with initial and end coordinates
    template<typename func_t, size_t N, size_t ST1, size_t ST2>
    static inline void run(const std::array<dim_t, N> &dims,
                           const std::array<dim_t,ST1> &strides1, const std::array<dim_t,ST2> &strides2,
                           const std::array<dim_t, N> &start,  const std::array<dim_t, N> &end, 
                           func_t fnc, dim_t base1 = 0, dim_t base2 = 0, bool first_step = true, bool last_step = false ){
      fnc(base1, base2);
    }
#else
        // loop with strides from 2 tensors with initial and end coordinates
    template<typename func_t, size_t N, size_t ST1, size_t ST2>
    static inline void run(const std::array<dim_t, N> &dims,
                           const std::array<dim_t,ST1> &strides1, const std::array<dim_t,ST2> &strides2,
                           const std::array<dim_t, N> &start,  const dim_t end, 
                           func_t fnc, dim_t base1 = 0, dim_t base2 = 0, bool first_step = true, bool last_step = false ){
      fnc(base1, base2);
    }
#endif
    
  };



    /*@brief Construct std::array of certain dimensions from another one with
    difference size, padding if necessary
  */
  template<typename T, size_t SIZE, T padding>
  class pad_array {
  public:
    template<size_t N>
    static constexpr std::array<T, SIZE> create(const std::array<T,N> v) {
      return _create(v, std::make_index_sequence<SIZE> {} );
    }
  private:
    template<size_t N, size_t ... dims>
    static constexpr std::array<T,SIZE>  _create(const std::array<T,N> v, std::index_sequence<dims...> ) {
      return { getEl(v, dims)... };
    }
    template<size_t N>
    static constexpr T getEl(const std::array<T,N> v, const size_t idx) {
      return idx >= N ? padding : v[idx];
    }
  };

#define make_dims  pad_array<dim_t, max_tensor_dimensions, 1>::create
#define make_strides  pad_array<dim_t, max_tensor_dimensions, 0>::create


#define assume(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)


}
#endif
