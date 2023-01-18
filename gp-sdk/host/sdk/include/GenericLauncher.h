//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------
#ifndef GENERIC_LAUNCHER_H
#define GENERIC_LAUNCHER_H
#include <device-layer/IDeviceLayer.h>
#include <runtime/IRuntime.h>

#include <hostUtils/logging/Logger.h>

#include <string>
#include <vector>

/**
 * \enum Mode
 * \brief list of execution mode types.
 **/
enum class Mode {
  PCIE,   /*!< Launch kernel in the PCIE connected accelerator. */
  SYSEMU, /*!< Launch kernel in the SYSEMU simulator. */
  FAKE,   /*!< Launch kernel in a FAKE device (dummy loopback) that does nothing. Used for testing purposes. */
  LAST
};

static inline std::string to_string(Mode m) {
  const std::vector<std::string> modes = {"silicon", "sysemu", "fake"};
  return m >= Mode::LAST ? "unknown" : modes[int(m)];
}

static inline Mode modeFromString(std::string m) {
  std::map<std::string, Mode> modeFromStr = {{"sysemu", Mode::SYSEMU}, {"fake", Mode::FAKE}, {"silicon", Mode::PCIE}};
  return modeFromStr.count(m) ? modeFromStr[m] : Mode::LAST;
}

/**
 * Structure to configure the type and number of devices on which the kernel will be launched.
 * \brief Configures type and number of devices.
 */
struct Config {
  void dump() const {
    std::cout << "Selected options:\n";
    std::cout << "  Device type: " << to_string(mode_) << "\n";
    std::cout << "  Device count: " << numDevices_ << "\n";
  }

  Mode mode_{Mode::SYSEMU};
  size_t numDevices_{1};
};

/**
 * Creates a GenericLauncher object, provides methods to load and execute kernels in ETSoC-1 devices.
 * Kernels are compiled RISC-V ETSoC-1 compatible binary files in Extensible Linkable Format (ELF).
 * \brief GenericLauncher class to load and run kernels in a device.
 */
class GenericLauncher {
public:
  GenericLauncher() = delete;
  /**
   * Creates a new GenericLauncher object with a config.
   * \brief Main constructor.
   * \param config specifies device type and device count configuration used in the kernel execution
   */
  GenericLauncher(const Config& config)
    : config_(config){};

  /**
   * Initializes the PCIE, SYSEMU or FAKE device interface where the kernel will run.
   * \brief Initializes the execution device.
   */
  void initialize(); // setup

  /**
   * Destroys the launcher.
   * \brief Destroys the launcher.
   */
  void tearDown();

  /**
   * Dumps the selected kernel event trace into a formatted text file.
   * If multiple instances of the same kernel are launched,
   * \p fileIdx allows add a distinctive numeric Id to the file name.
   * \brief Dumps a kernel trace
   * \param fileIdx numeric index appended to the trace file name
   * \param KernelId id of the kernel to dump
   */
  void dumpTracesToFile(uint64_t fileIdx = 0, rt::KernelId kernelId = (rt::KernelId)(-1));

  /**
   * Starts the execution of the loaded kernel on the device. Host and device code execute asynchronously until
   * waitKernelCompletion() is called. \brief Launches the kernel on the device. \param kernelId id of the kernel to
   * launch. \param params launch parameters. \param shireMask mask with the shires that will execute the kernel.
   */
  template <typename TParams>
  void kernelLaunch(rt::KernelId kernelId, TParams * params, uint64_t shireMask = 0xffffffff) {
    doKernelLaunch(kernelId, (std::byte *)params, sizeof(TParams), shireMask);
  }

  /**
   * Starts the execution of the loaded kernel on the device. Host and device code execute asynchronously until
   * waitKernelCompletion() is called. \brief Launches the kernel on the device without providing parameters. \param
   * kernelId id of the kernel to launch. \param shireMask mask with the shires that will execute the kernel.
   */
  void kernelLaunch(rt::KernelId kernelId, uint64_t shireMask = 0xffffffff) {
    doKernelLaunch(kernelId, nullptr, 0, shireMask);
  }

  /**
   * Blocks host side progress until the kernel on the device completes its execution.
   * If the device kernel does not complete in the allocated \p timeout seconds, it will be aborted raising an error.
   * \brief Waits for device kernel completion.
   * \param timeout max number of seconds to wait before timing out
   */
  void waitKernelCompletion(std::chrono::seconds timeout);

  /**
   * Loads an ETSoC-1 RISC-V binary in ELF format located at \p kernelName filepath.
   * \brief Loads a kernel on the device.
   * \param kernelName path to the kernel file (elf format) to load
   * \return returns a kernelId
   */
  rt::KernelId loadKernel(const std::string& kernelName, uint32_t deviceIdx = 0);

  /**
   * Unloads the kernel specified by \p kernelId from the device.
   * \brief Unloads a kernel from the device.
   * \param kernelId id of the kernel to unload
   */
  void unLoadKernel(rt::KernelId kernelId);
  
  std::atomic<uint64_t> kernelError_ = 0; // Number of kernels that reported an error
  std::atomic<uint64_t> kernelAbort_ = 0; // Number of kernels aborted

private:
  std::vector<std::byte> readFile(const std::string& path);
  // just to enable glog-logger on runtime.
  logging::LoggerDefault loggerDefault_;
  void doKernelLaunch(rt::KernelId, std::byte * params, size_t size, uint64_t shireMask);
  
protected:
  const Config& config_;
  std::unique_ptr<dev::IDeviceLayer> deviceLayer_;
  rt::RuntimePtr runtime_;
  std::vector<rt::DeviceId> devices_;
  std::vector<rt::StreamId> defaultStreams_;
  std::vector<rt::StreamId> traceStreams_;

  std::byte* traceDeviceBuffer_;
  // TODO: multi-dev design.
  size_t devIdx_ = 0;
};

#endif // GENERIC_LAUNCHER_H
