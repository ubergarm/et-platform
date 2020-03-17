#ifndef GEN_INSTANCES_H
#define GEN_INSTANCES_H

#define GEN_INSTANCES_INSTANCES(template, functionName, op, ...)               \
  template void functionName<float, op>(__VA_ARGS__);                          \
  template void functionName<float16, op>(__VA_ARGS__);                        \
  template void functionName<int8_t, op>(__VA_ARGS__);                         \
  template void functionName<int64_t, op>(__VA_ARGS__);

// The logic operators only support integral values
#define GEN_INSTANCES_INSTANCES_LOGIC(template, functionName, op, ...)    \
  template void functionName<int8_t, op>(__VA_ARGS__);                         \
  template void functionName<int64_t, op>(__VA_ARGS__);

#define GEN_INSTANCES_OP(template, functionName, ...)                          \
  template void functionName<float>(__VA_ARGS__);                              \
  template void functionName<float16>(__VA_ARGS__);                            \
  template void functionName<int8_t>(__VA_ARGS__);                             \
  template void functionName<int64_t>(__VA_ARGS__);                            \
  template void functionName<int32_t>(__VA_ARGS__);                            \
  template void functionName<int16_t>(__VA_ARGS__);

#define GEN_INSTANCES_1TYPEFP(template, functionName, ...)                     \
  template void functionName<float>(__VA_ARGS__);                              \
  template void functionName<float16>(__VA_ARGS__);

#define GEN_INSTANCES_3TYPE(template, functionName, op, ...)                   \
  template void functionName<float, float, float, op>(__VA_ARGS__);            \
  template void functionName<float16, float16, float16, op>(__VA_ARGS__);      \
  template void functionName<int8_t, int8_t, int8_t, op>(__VA_ARGS__);         \
  template void functionName<uint8_t, int8_t, int8_t, op>(__VA_ARGS__);        \
  template void functionName<int8_t, uint8_t, int8_t, op>(__VA_ARGS__);        \
  template void functionName<int8_t, int8_t, uint8_t, op>(__VA_ARGS__);        \
  template void functionName<uint8_t, uint8_t, int8_t, op>(__VA_ARGS__);       \
  template void functionName<uint8_t, int8_t, uint8_t, op>(__VA_ARGS__);       \
  template void functionName<int8_t, uint8_t, uint8_t, op>(__VA_ARGS__);       \
  template void functionName<uint8_t, uint8_t, uint8_t, op>(__VA_ARGS__);      \
  template void functionName<int16_t, int16_t, int16_t, op>(__VA_ARGS__);      \
  template void functionName<int64_t, int64_t, int64_t, op>(__VA_ARGS__);

#define GEN_INSTANCES_2TYPE(template, functionName, op, ...)                   \
  template void functionName<float, float, op>(__VA_ARGS__);                   \
  template void functionName<float16, float16, op>(__VA_ARGS__);               \
  template void functionName<int8_t, int8_t, op>(__VA_ARGS__);                 \
  template void functionName<uint8_t, int8_t, op>(__VA_ARGS__);                \
  template void functionName<int8_t, uint8_t, op>(__VA_ARGS__);                \
  template void functionName<uint8_t, uint8_t, op>(__VA_ARGS__);               \
  template void functionName<int64_t, int64_t, op>(__VA_ARGS__);

#define GEN_INSTANCES_3TYPE_OP(template, functionName, ...)                    \
  template void functionName<float, float, float>(__VA_ARGS__);                \
  template void functionName<float16, float16, float16>(__VA_ARGS__);          \
  template void functionName<int8_t, int8_t, int8_t>(__VA_ARGS__);             \
  template void functionName<uint8_t, int8_t, int8_t>(__VA_ARGS__);            \
  template void functionName<int8_t, uint8_t, int8_t>(__VA_ARGS__);            \
  template void functionName<int8_t, int8_t, uint8_t>(__VA_ARGS__);            \
  template void functionName<uint8_t, uint8_t, int8_t>(__VA_ARGS__);           \
  template void functionName<uint8_t, int8_t, uint8_t>(__VA_ARGS__);           \
  template void functionName<int8_t, uint8_t, uint8_t>(__VA_ARGS__);           \
  template void functionName<uint8_t, uint8_t, uint8_t>(__VA_ARGS__);          \
  template void functionName<int16_t, int16_t, int16_t>(__VA_ARGS__);          \
  template void functionName<int64_t, int64_t, int64_t>(__VA_ARGS__);

#define GEN_INSTANCES_2TYPE_OP(template, functionName, ...)                    \
  template void functionName<float, float>(__VA_ARGS__);                       \
  template void functionName<float16, float16>(__VA_ARGS__);                   \
  template void functionName<int8_t, int8_t>(__VA_ARGS__);                     \
  template void functionName<uint8_t, int8_t>(__VA_ARGS__);                    \
  template void functionName<int8_t, uint8_t>(__VA_ARGS__);                    \
  template void functionName<uint8_t, uint8_t>(__VA_ARGS__);                   \
  template void functionName<int64_t, int64_t>(__VA_ARGS__);

#define GEN_INSTANCES_INTONLY_OP(template, functionName, ...)                  \
  template void functionName<int64_t>(__VA_ARGS__);                            \
  template void functionName<int32_t>(__VA_ARGS__);

#define GEN_INSTANCES_QUANT(template, functionName, ...)                       \
  template void functionName<int8_t>(__VA_ARGS__);                             \
  template void functionName<int16_t>(__VA_ARGS__);                            \
  template void functionName<int32_t>(__VA_ARGS__);

#define GEN_INSTANCES_OP_INDEX(template, functionName, ...)                    \
  template void functionName<float, int64_t>(__VA_ARGS__);                     \
  template void functionName<float16, int64_t>(__VA_ARGS__);                   \
  template void functionName<int8_t, int64_t>(__VA_ARGS__);                    \
  template void functionName<int64_t, int64_t>(__VA_ARGS__);                   \
  template void functionName<int32_t, int64_t>(__VA_ARGS__);                   \
  template void functionName<float, int32_t>(__VA_ARGS__);                     \
  template void functionName<float16, int32_t>(__VA_ARGS__);                   \
  template void functionName<int8_t, int32_t>(__VA_ARGS__);                    \
  template void functionName<int64_t, int32_t>(__VA_ARGS__);                   \
  template void functionName<int32_t, int32_t>(__VA_ARGS__);

#define GEN_INSTANCES_CONVERT(template, functionName, ...)                     \
  template void functionName<float, int64_t>(__VA_ARGS__);                     \
  template void functionName<float, float16>(__VA_ARGS__);                     \
  template void functionName<float, float>(__VA_ARGS__);                       \
  template void functionName<float16, float>(__VA_ARGS__);                     \
  template void functionName<float16, float16>(__VA_ARGS__);                   \
  template void functionName<int64_t, float>(__VA_ARGS__);                     \
  template void functionName<int64_t, int64_t>(__VA_ARGS__);

#define GEN_INSTANCES_INT8_FUN(functionName, ...)                              \
  void functionNameInt8(__VA_ARGS__);

#define GEN_INSTANCES_FRQSLWS_V(template, functionName, ...)                   \
  template void functionName<true,  true,  true>(__VA_ARGS__);                 \
  template void functionName<true,  true,  false>(__VA_ARGS__);                \
  template void functionName<true,  false, true>(__VA_ARGS__);                 \
  template void functionName<false, true,  false>(__VA_ARGS__);                \
  template void functionName<false, false, true>(__VA_ARGS__);

#define GEN_INSTANCES_RQSLWS_V(template, functionName, ...)                   \
  template void functionName<true,  true>(__VA_ARGS__);                       \
  template void functionName<true,  false>(__VA_ARGS__);                      \
  template void functionName<false, true>(__VA_ARGS__);                       \
  template void functionName<false, false>(__VA_ARGS__);

#define GEN_INSTANCES(functionName, op, ...)                                          \
  template <typename srcType, typename opType>                                        \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_INSTANCES(extern template, functionName, op, __VA_ARGS__)

#define GEN_OP(functionName, ...)                                                     \
  template <typename srcType>                                                         \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_OP(extern template, functionName, __VA_ARGS__)

#define GEN_1TYPEFP(functionName, ...)                                                \
  template <typename dstType>                                                         \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_1TYPEFP(extern template, functionName, __VA_ARGS__)

#define GEN_3TYPE(functionName, op, ...)                                              \
  template <typename src1Type, typename src2Type, typename dstType, typename opType>  \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_3TYPE(extern template, functionName, op, __VA_ARGS__)

#define GEN_2TYPE(functionName, op, ...)                                              \
  template <typename src1Type, typename dstType, typename opType>                     \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_2TYPE(extern template, functionName, op, __VA_ARGS__)

#define GEN_3TYPE_OP(functionName, ...)                                               \
  template <typename src1Type, typename src2Type, typename dstType>                   \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_3TYPE_OP(extern template, functionName, __VA_ARGS__)

#define GEN_2TYPE_OP(functionName, ...)                                               \
  template <typename src1Type, typename dstType>                                      \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_2TYPE_OP(extern template, functionName, __VA_ARGS__)

#define GEN_INTONLY_OP(functionName, ...)                                             \
  template <typename srcType>                                                         \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_INTONLY_OP(extern template, functionName, __VA_ARGS__)

#define GEN_QUANT(functionName, ...)                                                  \
  template <typename srcType>                                                         \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_QUANT(extern template, functionName, __VA_ARGS__)

#define GEN_OP_INDEX(functionName, ...)                                               \
  template <typename srcType, typename indexType>                                     \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_OP_INDEX(extern template, functionName, __VA_ARGS__)

#define GEN_CONVERT(functionName, ...)                                                \
  template <typename srcType, typename dstType>                                       \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_CONVERT(extern template, functionName, __VA_ARGS__)

#define GEN_INT8_FUN(functionName, ...)                                               \
  void functionNameInt8(__VA_ARGS__);

#define GEN_FRQSLWS_V(functionName, ...)                                              \
  template <bool Weighted = true, bool Float32Dst = true, bool FLoat16Dst = false>    \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_FRQSLWS_V(extern template, functionName, __VA_ARGS__)

#define GEN_RQSLWS_V(functionName, ...)                                               \
  template <bool Int8Src = false, bool FLoat16Dst = false>                            \
  void functionName(__VA_ARGS__);                                                     \
  GEN_INSTANCES_RQSLWS_V(extern template, functionName, __VA_ARGS__)

#endif
