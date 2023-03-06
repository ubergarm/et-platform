/*-------------------------------------------------------------------------
 * Copyright (C) 2022, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies adn
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef DNNLIB_OPERATORS_H
#define DNNLIB_OPERATORS_H

class Add {};
class Sub {};
class Mul {};
class Div {};
class Max {};
class Min {};
class CmpEQ {};
class CmpNEQ {};
class CmpLTE {};
class CmpLT {};
class Select {};
class Pow {};
class ElementLog {};
class ElementExp {};
class ElementErf {};
class ElementIsNaN {};
class ElementNeg {};
class Tanh {};
class Sin {};
class Cos {};
class Sigmoid {};
class And {}; // bit-wise
class Or {}; // bit-wise
class Xor {}; // bit-wise
class LogicalNot {};

#endif //  DNNLIB_OPERATORS_H

