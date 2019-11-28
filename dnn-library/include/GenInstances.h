#ifndef GEN_INSTANCES_H
#define GEN_INSTANCES_H

#define GEN_INSTANCES_INSTANCES(template, functionName, op, ...)               \
  template void functionName<float, op>(__VA_ARGS__);                          \
  template void functionName<float16, op>(__VA_ARGS__);                        \
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

#endif
