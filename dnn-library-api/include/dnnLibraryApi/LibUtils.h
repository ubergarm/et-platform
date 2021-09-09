#ifndef _LIB_UTILS_H_
#define _LIB_UTILS_H_

#include "LibTypes.h"
#include <utility>

namespace dnn_lib {
/*@brief aliases, forward declarations, and misc constants
 */
constexpr unsigned max_tensor_dimensions = 6;

using dim_array_t = std::array<dim_t, max_tensor_dimensions>;
using sdim_array_t = std::array<sdim_t, max_tensor_dimensions>;

/*@brief Construct std::array of certain dimensions from another one with
 difference size, padding if necessary
 */
template <typename T, size_t SIZE, T padding> class pad_array {
public:
  template <size_t N> static constexpr std::array<T, SIZE> create(const std::array<T, N>& v) {
    return _create(v, std::make_index_sequence<SIZE>{});
  }

private:
  template <size_t N, size_t... dims>
  static constexpr std::array<T, SIZE> _create(const std::array<T, N>& v, std::index_sequence<dims...>) {
    return {getEl<dims>(v)...};
  }
  template <size_t idx, size_t N> static constexpr T getEl(const std::array<T, N>& v) {
    return idx >= N ? padding : v[idx];
  }
};

#define make_dims pad_array<dim_t, max_tensor_dimensions, 1>::create
#define make_strides pad_array<dim_t, max_tensor_dimensions, 0>::create

} // namespace dnn_lib

#endif
