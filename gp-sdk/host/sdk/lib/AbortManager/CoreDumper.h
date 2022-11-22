//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------

#ifndef CORE_DUMPER_H
#define CORE_DUMPER_H

#include "GPSDKElf.h"
#include "runtime/IRuntime.h"

#include <ostream>
#include <string>

class AbortManager;

class CoreDumper {
public:
  // This is a static class
  CoreDumper() = delete;

  // Dump a core for the kernel launch associated to the given event identifier
  static void dumpCore(AbortManager& abortManager, rt::IRuntime* runtime, rt::EventId eventId,
                       const rt::StreamError& error);

private:
  static auto constexpr DEFAULT_PROCESS_NAME = "kernels.elf";
  static constexpr size_t ALIGNMENT = 256;

  static constexpr uint64_t CONTEXT_TYPE_COMPUTE_HANG = 0;
  static constexpr uint64_t CONTEXT_TYPE_USER_MODE_EXCEPTION = 1;
  static constexpr uint64_t CONTEXT_TYPE_SYSTEM_ABORT = 2;
  static constexpr uint64_t CONTEXT_TYPE_SELF_ABORT = 3;
  static constexpr uint64_t CONTEXT_TYPE_KERNEL_EXECUTION_ERROR = 4;
  static constexpr uint64_t TOTAL_CONTEXT_TYPES = 5;

  // Linux process states
  enum class ProcessState { R = 0, S, D, T, Z, W };

  struct NoteData {
    // Note fragments of the main thread
    GPSDKElf::MainThreadNote mainThread_;
    // Note fragments of all other threads
    std::vector<GPSDKElf::PRStatusNote> threads_;

    // Size in disk
    size_t size() const {
      return GPSDKElf::MainThreadNote::size_ + GPSDKElf::PRStatusNote::size_ * threads_.size();
    }

    // Apply a function to each member/element
    template <typename FUNC, typename... Args> void apply(FUNC& f, Args... args) const {
      mainThread_.apply(f, args...);
      for (auto const& thread : threads_) {
        thread.apply(f, args...);
      }
    }
  };

  // Fill an output stream with as many zeros as specified
  static void fillOStreamGap(std::ostream& out, size_t size);

  // Copy a C++ string to a C-style char array taking the ending if it does not
  // fit
  static void copyStringEnding(char* dest, std::string const& source, size_t limit);

  // Generates the siginfo note
  static GPSDKElf::PRSigInfoNote getSigInfoNote([[maybe_unused]] const rt::StreamError& error,
                                                const rt::ErrorContext& context);
  // Generates the process status note
  static GPSDKElf::PRStatusNote getPRStatusNote(const rt::StreamError& error, const rt::ErrorContext& context);
  // Generates the process info note
  static GPSDKElf::PRPSInfoNote getPSInfoNote([[maybe_unused]] const rt::StreamError& error,
                                              const rt::ErrorContext& context, std::string const& processName);

  // Returns the positions of the context vector that contain a valid entry
  static std::vector<size_t> getValidErrorContextIndices(const rt::StreamError& error);

  // Generates the note data of the core dump
  static NoteData createNoteData(const rt::StreamError& error, std::vector<size_t> const& contextIndices,
                                 std::string const& processName);

  // Helper class to dump structs piecewise (to skip compiler introduced
  // padding)
  struct Dumper {
    std::ostream& out_;

    explicit Dumper(std::ostream& out)
      : out_(out) {
    }

    template <typename T> void operator()(T const& object) const {
      out_.write((char const*)&object, sizeof(object));
    }
  };

  // Returns an offset corrected to be aligned to ALIGN
  static size_t align(size_t offset);
};

#endif //CORE_DUMPER_H
