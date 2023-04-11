//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------

#include "CoreDumper.h"
#include "AbortManager.h"
#include "GPSDKElf.h"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <tuple>

# include <unistd.h>

// Standard RISC-V exception mcause values

// Instruction address misaligned
static constexpr uint64_t RISCV_INSTR_MISSALIGN_MCAUSE = 0;
// Instruction access fault
static constexpr uint64_t RISCV_INSTR_FAULT_MCAUSE = 1;
// Illegal instruction
static constexpr uint64_t RISCV_INSTR_ILLEGAL_MCAUSE = 2;
// Load address misaligned
static constexpr uint64_t RISCV_LOAD_MISSALIGN_MCAUSE = 4;
// Load access fault
static constexpr uint64_t RISCV_LOAD_FAULT_MCAUSE = 5;
// Store/AMO address misaligned
static constexpr uint64_t RISCV_STORE_MISSALIGN_MCAUSE = 6;
// Store/AMO access fault
static constexpr uint64_t RISCV_STORE_FAULT_MCAUSE = 7;

// ELF machine ID which may not be in the linux headers
#ifndef EM_RISCV
#define EM_RISCV 243
#endif

// Fill an output stream with as many zeros as specified
void CoreDumper::fillOStreamGap(std::ostream& out, size_t size) {
  assert((int64_t)size >= 0);

  for (size_t i = 0; i < size; i++) {
    out << '\0';
  }
}

// Copy a C++ string to a C-style char array taking the ending if it does not
// fit
void CoreDumper::copyStringEnding(char* dest, std::string const& source, size_t limit) {
  auto length = source.size();

  if (length > limit) {
    size_t offset = length - limit;
    for (size_t i = 0; i < limit; i++) {
      dest[i] = source[i + offset];
    }
  } else {
    for (size_t i = 0; i < length; i++) {
      dest[i] = source[i];
    }
  }
}

// Generates the siginfo note
GPSDKElf::PRSigInfoNote CoreDumper::getSigInfoNote([[maybe_unused]] const rt::StreamError& error,
                                                   const rt::ErrorContext& context) {
  GPSDKElf::PRSigInfoNote note;

  // Since this is a struct containing unions, here we initialize it to 0
  memset(&note.sigInfo_, 0, sizeof(note.sigInfo_));

  bool isInterrupt = context.mcause_ & (1UL << 63);
  uint64_t cleanMCause = context.mcause_ & ~(1UL << 63);

  note.sigInfo_.si_pid = context.hartId_;
  if (isInterrupt) {
    note.sigInfo_.si_signo = SIGINT;
  } else {
    switch (cleanMCause) {
    case RISCV_INSTR_MISSALIGN_MCAUSE:
      note.sigInfo_.si_signo = SIGBUS;
      note.sigInfo_.si_code = BUS_ADRALN;
      break;
    case RISCV_INSTR_FAULT_MCAUSE:
      note.sigInfo_.si_signo = SIGSEGV;
      note.sigInfo_.si_code = SEGV_ACCERR;
      break;
    case RISCV_INSTR_ILLEGAL_MCAUSE:
      note.sigInfo_.si_signo = SIGILL;
      note.sigInfo_.si_code = ILL_ILLOPC;
      break;
    case RISCV_LOAD_MISSALIGN_MCAUSE:
      note.sigInfo_.si_signo = SIGBUS;
      note.sigInfo_.si_code = BUS_ADRALN;
      break;
    case RISCV_LOAD_FAULT_MCAUSE:
      note.sigInfo_.si_signo = SIGSEGV;
      note.sigInfo_.si_code = SEGV_ACCERR;
      break;
    case RISCV_STORE_MISSALIGN_MCAUSE:
      note.sigInfo_.si_signo = SIGBUS;
      note.sigInfo_.si_code = BUS_ADRALN;
      break;
    case RISCV_STORE_FAULT_MCAUSE:
      note.sigInfo_.si_signo = SIGSEGV;
      note.sigInfo_.si_code = SEGV_ACCERR;
      break;
    default:
      // NOTE: we assign SIGPIPE for things that make no sense
      note.sigInfo_.si_signo = SIGPIPE;
      break;
    }
    note.sigInfo_.si_addr = (void*)context.mtval_;
  }

  return note;
}

// Generates the process status note
GPSDKElf::PRStatusNote CoreDumper::getPRStatusNote(const rt::StreamError& error, const rt::ErrorContext& context) {
  GPSDKElf::PRStatusNote note;

  // Build the ELF siginfo from what would be a system siginfo note
  {
    auto siginfoNote = getSigInfoNote(error, context);
    note.status_.pr_info.si_signo = siginfoNote.sigInfo_.si_signo;
    note.status_.pr_info.si_code = siginfoNote.sigInfo_.si_code;
    note.status_.pr_info.si_errno = 0;
    note.status_.pr_cursig = 0;
    note.status_.pr_sigpend = 0;
    note.status_.pr_sighold = 0;
    note.status_.pr_pid = context.hartId_;
    note.status_.pr_ppid = 0;
    note.status_.pr_pgrp = 0;
    note.status_.pr_sid = 0;
    // NOTE: note.pr_utime, note.pr_stime, note.pr_cu_time and note.pr_cstime
    // are set to 0 by default
  }

  note.status_.pr_reg.pc = context.mepc_;
  for (int i = 0; i < 31; i++) {
    note.status_.pr_reg.xregs[i] = context.gpr_[i];
  }

  note.status_.pr_fpvalid = 0;

  return note;
}

// Generates the process info note
GPSDKElf::PRPSInfoNote CoreDumper::getPSInfoNote([[maybe_unused]] const rt::StreamError& error,
                                                 const rt::ErrorContext& context, std::string const& processName) {
  GPSDKElf::PRPSInfoNote note;

  // Zero it to simplify copying strings
  memset(&note.psInfo_, 0, sizeof(note.psInfo_));

  note.psInfo_.pr_state = (char)ProcessState::R;
  note.psInfo_.pr_sname = 'R';
  note.psInfo_.pr_zombie = 0;
  note.psInfo_.pr_nice = 0;
  note.psInfo_.pr_flag = PF_DUMPCORE;
  note.psInfo_.pr_uid = getuid();
  note.psInfo_.pr_gid = getgid();
  note.psInfo_.pr_pid = context.hartId_;
  note.psInfo_.pr_ppid = 0;
  note.psInfo_.pr_pgrp = 0;
  note.psInfo_.pr_sid = 0;

  // Set up the process name
  copyStringEnding(note.psInfo_.pr_fname, processName, sizeof(note.psInfo_.pr_fname));
  copyStringEnding(note.psInfo_.pr_psargs, processName, sizeof(note.psInfo_.pr_psargs));

  return note;
}

// Returns the positions of the context vector that contain a valid entry
std::vector<size_t> CoreDumper::getValidErrorContextIndices(const rt::StreamError& error) {
  std::vector<size_t> result;

  auto& contexts = error.errorContext_.value();
  for (size_t i = 1; i < contexts.size(); i++) {
    if (contexts[i].type_ < TOTAL_CONTEXT_TYPES) {
      result.push_back(i);
    }
  }

  return result;
}

// Generates the note data of the core dump
CoreDumper::NoteData CoreDumper::createNoteData(const rt::StreamError& error, std::vector<size_t> const& contextIndices,
                                                std::string const& processName) {
  assert(not contextIndices.empty());

  auto& contexts = error.errorContext_.value();
  auto const& first = contexts[contextIndices[0]];

  NoteData data;
  data.mainThread_.prStatus_ = getPRStatusNote(error, first);
  data.mainThread_.psInfo_ = getPSInfoNote(error, first, processName);
  data.mainThread_.sigInfo_ = getSigInfoNote(error, first);

  for (size_t i = 1; i < contextIndices.size(); i++) {
    auto index = contextIndices[i];
    data.threads_.emplace_back(getPRStatusNote(error, contexts[index]));
  }

  return data;
}

// Returns an offset corrected to be aligned to ALIGN
size_t CoreDumper::align(size_t offset) {
  if (offset % ALIGNMENT != 0) {
    offset += ALIGNMENT - (offset % ALIGNMENT);
  }

  return offset;
}

// Dump a core for the kernel launch associated to the given event identifier
void CoreDumper::dumpCore(AbortManager& abortManager, rt::IRuntime* runtime, rt::EventId eventId,
                          const rt::StreamError& error) {
  AbortManager::Handler abortHandler(abortManager);

  if (not error.errorContext_.has_value()) {
    std::cout << "[INFO] Not dumping a core without context.\n";
    std::cout << "[WARNING] Device error (no core dump possible)\n";
    return;
  }

  if (abortHandler.hasMultipleKernels()) {
    std::cout << "[INFO] Multiple kernels loaded. The core dump will not include the "
                 "stack since it may have been overwritten.\n";
  }

  if (not abortHandler.hasCode()) {
    std::cout << "[INFO] Not dumping a core without any kernel loaded.\n";
    std::cout << "[WARNING] Device error (no core dump possible)\n";
    return;
  }

  auto contextIndices = getValidErrorContextIndices(error);
  if (contextIndices.empty()) {
    std::cout << "[INFO] Could not find any initialized device error context.\n";
    std::cout << "[WARNING] Device error (no core dump possible)\n";
    return;
  }

  auto kernelLaunchDeviceId = abortHandler.getKernelLaunchDevice(eventId);
  if (not kernelLaunchDeviceId.has_value()) {
    // Not a kernel launch error, so there is no core to generate
    return;
  }

  // Device for which the core is to be dumped
  auto deviceId = kernelLaunchDeviceId.value();

  // This stream will be used to copy the data from the device so that is can be
  // dumped into the core file
  auto copyBackStream = runtime->createStream(deviceId);

  auto const& allocatedDeviceMemory = abortHandler.getAllocatedDeviceMemoryMap(deviceId);

  // Find out the name of the kernel ELF file
  std::string processName = DEFAULT_PROCESS_NAME;
  for (auto& [deviceAddress, allocationInfo] : allocatedDeviceMemory) {
    if ((allocationInfo.useType_ == AbortManager::DeviceMemoryUse::CODE) and (allocationInfo.path_.size() != 0)) {
      // Check if the exception PC is in this memory region
      for (auto const& context : error.errorContext_.value()) {
        if ((deviceAddress <= (std::byte*)context.mepc_) and
            ((std::byte*)context.mepc_ < deviceAddress + allocationInfo.size_)) {
          processName = allocationInfo.path_;
          break;
        }
      }
    }
  }

  std::string coreFileName = "core." + std::to_string(getpid()) + ".etsoc." + std::to_string((int)error.device_) + "." +
                             std::to_string((int)eventId);
  std::ofstream out(coreFileName.c_str());
  if (out.bad()) {
    std::cout << "[ERROR] Error creating '" << coreFileName
               << "' device core dump file.\n";
    std::cout << "[WARNING] Device error (no core dump possible)\n";
    runtime->destroyStream(copyBackStream);
    return;
  }
  size_t fileOffset = 0;

  auto numSegments = /* notes */ 1 + allocatedDeviceMemory.size();

  // Initialize and write the ELF header
  {
    GPSDKElf::Header header;
    header.e_type = ET_CORE;
    header.e_machine = EM_RISCV;
    header.e_phnum = numSegments;

    out.write((const char*)&header, sizeof(header));
    fileOffset += sizeof(header);
  }

  std::vector<GPSDKElf::SegmentHeader> segmentHeaders(numSegments);

  // Data starts after the segment headers
  auto dataOffset = sizeof(GPSDKElf::Header) + sizeof(GPSDKElf::SegmentHeader) * numSegments;
  dataOffset = align(dataOffset);

  // Initialize the notes segment header
  GPSDKElf::SegmentHeader& notesHeader = segmentHeaders[0];
  auto note = createNoteData(error, contextIndices, processName);
  notesHeader.p_type = PT_NOTE;
  notesHeader.p_offset = dataOffset;
  notesHeader.p_filesz = note.size();
  notesHeader.p_align = 4;

  dataOffset = dataOffset + note.size();
  dataOffset = align(dataOffset);

  // Initialize the code and data segment headers
  size_t segmentIndex = 1;
  for (auto& [deviceAddress, allocationInfo] : allocatedDeviceMemory) {
    auto& segmentHeader = segmentHeaders[segmentIndex];

    segmentHeader.p_type = PT_LOAD;

    switch (allocationInfo.useType_) {
    case AbortManager::DeviceMemoryUse::CODE:
      segmentHeader.p_flags = (PF_X | PF_R);
      break;
    case AbortManager::DeviceMemoryUse::DATA:
    case AbortManager::DeviceMemoryUse::STACK:
      segmentHeader.p_flags = (PF_R | PF_W);
      break;
    default:
      assert(false);
      break;
    }

    segmentHeader.p_offset = dataOffset;
    segmentHeader.p_vaddr = (uint64_t)deviceAddress;
    segmentHeader.p_paddr = (uint64_t)deviceAddress;
    segmentHeader.p_filesz = allocationInfo.size_;
    segmentHeader.p_memsz = allocationInfo.size_;
    segmentHeader.p_align = ALIGNMENT;

    dataOffset = dataOffset + allocationInfo.size_;
    dataOffset = align(dataOffset);
    segmentIndex++;
  }

  // Write the segment headers
  out.write((const char*)segmentHeaders.data(), sizeof(*segmentHeaders.data()) * segmentHeaders.size());
  fileOffset += sizeof(*segmentHeaders.data()) * segmentHeaders.size();

  // Enforce alignment in the output file
  fillOStreamGap(out, segmentHeaders[0].p_offset - fileOffset);
  fileOffset = segmentHeaders[0].p_offset;

  // Dump the note segment
  {
    Dumper dumper(out);
    note.apply(dumper);
    fileOffset += note.size();
  }

  // Dump allocated device memory regions
  segmentIndex = 1;
  for (auto& [deviceAddress, allocationInfo] : allocatedDeviceMemory) {
    // Space for a copy in the host
    auto hostAddress = std::make_unique<std::byte[]>(allocationInfo.size_);

    // Copy from device to host
    auto copyEventId =
      runtime->memcpyDeviceToHost(copyBackStream, deviceAddress, hostAddress.get(), allocationInfo.size_,
                                  /* barrier */ false);

    // Enforce alignment in the output file
    fillOStreamGap(out, segmentHeaders[segmentIndex].p_offset - fileOffset);
    fileOffset = segmentHeaders[segmentIndex].p_offset;

    // Wait for the copy to finish
    bool success = runtime->waitForEvent(copyEventId);
    if (not success) {
      std::cout << "[ERROR] Timed out copying core dump data from device.\n";
      runtime->destroyStream(copyBackStream);
      out.close();
      return;
    }

    // Write the segment
    out.write((const char*)hostAddress.get(), allocationInfo.size_);
    fileOffset += allocationInfo.size_;

    segmentIndex++;
  }

  out.close();

  runtime->destroyStream(copyBackStream);

  std::cout << "\n[WARNING] Device error (core dumped " << coreFileName << ")\n";
}
