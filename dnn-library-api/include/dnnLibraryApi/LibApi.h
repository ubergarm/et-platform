/*-------------------------------------------------------------------------
 * Copyright (C) 2021, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

/*! \brief DNN Library API definition
 *
 * The DNN Library API defines a set of functions to retrieve information for the different operators present in the
 * library. These set of functions are used by the application on top to know what are the different parameters of
 * the operators, as well as other important information like what operands are left at which hierarchy level.
 */

#ifndef _LIB_API_H_
#define _LIB_API_H_

#include "dnnLibraryApi/LibTensor.h"
#include "dnnLibraryApi/LibTypes.h"

#include <string>
#include <vector>

namespace dnn_lib {

/*!
 * getInstrConfig fills a struct with relevant information of one operator implemented by the Lib
 * \param[in] operatorName is the name of the operator that the consumer is requesting
 * \param[out] instConfig contains all the relevant information for the operator of interest
 * \returns true if the operator name was found in the database
 */
bool getInstrConfig(const std::string& operatorName, instrConfig& instConfig);

/*!
 * getInstrNumCycles returns the expected number of execution cycles required to execute a specific operator on an
 * ETSOC device. It requires knowing the operator type, number of assigned minions as well as the shapes of the
 * different operands
 * \param[in] operatorName is the name of the operator that the consumer is requesting
 * \param[in] assignedMinions is the total amount of compute resources assigned to the execution of the operator
 * \param[in] operands is a vector with the different input and output operands of the operator
 * \returns the total expected number of cycles required to execute the operator
 */
size_t getInstrNumCycles(const std::string& operatorName, size_t assignedMinions,
                         const std::vector<LibTensor*>& operands);

/*!
 * getInstrNumCycles returns the expected number of execution cycles required to execute a specific operator on an
 * ETSOC device. It requires knowing the operator type, number of assigned minions as well as the shapes of the
 * different operands
 * \param[in] operatorName is the name of the operator that the consumer is requesting
 * \param[in] assignedMinions is the total amount of compute resources assigned to the execution of the operator
 * \param[in] operands is a vector with the different input and output operands of the operator
 * \returns the total expected number of cycles required to execute the operator
 */
size_t getInstrNumCycles(const std::string& operatorName, size_t assignedMinions, const std::vector<Tensor>& operands);

/*!
 * getImplementation returns the best implementation of a specific operator that the Library can execute based on the
 * shapes of the different tensors passed as arguments.
 * \param[in] operatorName is the name of the operator that the consumer is requesting
 * \param[in] outOperands is a vector with the different output tensors of the operation
 * \param[in] inOperands is a vector with the different input tensors of the operation
 * \returns the selected implementation ID based on the different tensor constraints
 */
size_t getImplementation(const std::string& operatorName, const std::vector<Tensor>& outOperands,
                         const std::vector<Tensor>& inOperands);

} // end namespace dnn_lib

#endif
