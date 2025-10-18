/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
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

// Local
#include "dnnLibraryApi/LibTypes.h"

// STD
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

/*!
 * getMemberName returns a string with the member name.
 * \param[in] mb is the enum of the member of interest
 * \returns the member name in string format
 */
std::string getMemberName(InstrMembers mb);

/*!
 * getMemberType returns a string with the member type (bool, int, int64_t, ...).
 * \param[in] mb is the enum of the member of interest
 * \returns the member type in string format
 */
std::string getMemberType(InstrMembers mb);

/*!
 * getMemberScalar returns a boolean containing if the member is a scalar value
 * \param[in] mb is the enum of the member of interest
 * \returns if the member is a scalar or not
 */
bool getMemberScalar(InstrMembers mb);

/*!
 * getGenericOperatorIncludes returns generic includes needed for this op
 * \param[in] operator to be queried
 * \returns generic include collection
 */
std::vector<std::string> getGenericOperatorIncludes(const std::string& operatorName);

/*!
 * getSpecificOperatorIncludes returns specific includes needed for this op
 * \param[in] operator to be queried
 * \returns specific include collection
 */
std::vector<std::string> getSpecificOperatorIncludes(const std::string& operatorName);

} // end namespace dnn_lib

#endif
