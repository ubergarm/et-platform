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
#include <AbortManager.h>
#include <device-layer/IDeviceLayer.h>
#include <runtime/IRuntime.h>

#include <hostUtils/logging/Logger.h>

#include <string>
#include <vector>

#if __has_include("filesystem")
#include <filesystem>
#elif __has_include("experimental/filesystem")
#include <experimental/filesystem>
namespace std {
namespace filesystem = std::experimental::filesystem;
}
#endif
namespace fs = std::filesystem;

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
   * \param argc number of cmd-line arguments to parse
   * \param argv pointer to arguments
   * \param strictArs (defaults to true) if true, unknown parameters error the app, so it
   * assumes we only receive the parameters of interest.  if false, the check is relaxed (This would allow reparsing
   * previously parsed command-lines for example).
   */
  GenericLauncher(const Config& config, int argc, char** argv, bool strictArgs = true)
    : config_(config) {

    parse_args(argc, argv, strictArgs);
  };

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
   * \brief Dumps a kernel trace
   * \param fileIdx numeric index appended to the trace file name
   * \param KernelId id of the kernel to dump
   * \param deviceIdx device target to work with
   */
  void dumpTracesToFile(uint64_t fileIdx = 0, rt::KernelId kernelId = (rt::KernelId)(-1), uint32_t deviceIdx = 0);

  // /**
  //  * Starts the execution of the loaded kernel on the device. Host and device code execute asynchronously until
  //  * waitKernelCompletion() is called.
  //  * \brief Launches the kernel on the device.
  //  * \param kernelId id of the kernel to launch.
  //  * \param numThreads number of threads, must be a multiple of 32.
  //  * \param params launch parameters.
  //  * \param shireMask mask with the shires that will execute the kernel.
  //  */
  // template <typename TParams>
  // void kernelLaunch(rt::KernelId kernelId, int32_t numThreads, TParams * params, int32_t numThreadsPerCore = 1,
  // uint32_t deviceIdx = 0, uint64_t shireMask = 0xffffffff) {
  //   // params->env.numThreads = numThreads;
  //   // // Compute shireMask based on numThreads. This should be decided by the runtime in the future.
  //   // uint64_t activeShires = (((numThreads / numThreadsPerCore) + 31) / 32);
  //   // uint64_t smask = (1UL << activeShires) - 1UL;
  //   params->env.shireMask = shireMask;
  //   doKernelLaunch(kernelId, (std::byte *) params, sizeof(TParams), smask, deviceIdx);
  // }

  /**
   * Starts the execution of the loaded kernel on the device. Host and device code execute asynchronously until
   * waitKernelCompletion() is called.
   * \brief Launches the kernel on the device. \param kernelId id of the kernel to
   * launch.
   * \param params launch parameters.
   * \param shireMask mask with the shires that will execute the kernel.
   * \param deviceIdx device target to work with
   */
  template <typename TParams>
  void kernelLaunch(rt::KernelId kernelId, TParams* params, uint32_t deviceIdx = 0, uint64_t shireMask = 0xffffffff) {
    doKernelLaunch(kernelId, (std::byte*)params, sizeof(TParams), shireMask, deviceIdx);
  }

  /**
   * Starts the execution of the loaded kernel on the device. Host and device code execute asynchronously until
   * waitKernelCompletion() is called.
   * \brief Launches the kernel on the device without providing parameters.
   * \param kernelId id of the kernel to launch.
   * \param shireMask mask with the shires that will execute the kernel.
   * \param deviceIdx device target to work with
   */
  void kernelLaunch(rt::KernelId kernelId, uint32_t deviceIdx = 0, uint64_t shireMask = 0xffffffff) {
    doKernelLaunch(kernelId, nullptr, 0, shireMask, deviceIdx);
  }

  /**
   * Blocks host side progress until the kernel on the device completes its execution.
   * If the device kernel does not complete in the allocated \p timeout seconds, it will be aborted raising an error.
   * \brief Waits for device kernel completion.
   * \param timeout max number of seconds to wait before timing out
   * \param deviceIdx device target to work with
   */
  void waitKernelCompletion(std::chrono::seconds timeout, uint32_t deviceIdx = 0);

  /**
   * Loads an ETSoC-1 RISC-V binary in ELF format located at \p kernelName filepath.
   * \brief Loads a kernel on the device.
   * \param kernelName path to the kernel file (elf format) to load
   * \param deviceIdx device target to work with
   * \return returns a kernelId
   */
  rt::KernelId loadKernel(const std::string& kernelName, uint32_t deviceIdx = 0);

  /**
   * Unloads the kernel specified by \p kernelId from the device.
   * \brief Unloads a kernel from the device.
   * \param kernelId id of the kernel to unload
   */
  void unLoadKernel(rt::KernelId kernelId);

  /**
   * Get the number of PCIE devices detected
   * \brief It has real value after initialize call
   * \return numDev_
   */
  uint32_t getNumDevices(void) {
    return numDev_;
  }

  std::atomic<uint64_t> kernelError_ = 0; // Number of kernels that reported an error
  std::atomic<uint64_t> kernelAbort_ = 0; // Number of kernels aborted

  // static inline constexpr const char* help_msg =
  constexpr static const char* help_msg =
    "  '', --enableCoreDump          Write perfetto trace to a file instead\n"
    "  '', --simulator_params        Hyperparameters to pass to simulator, overrides default values\n";

private:
  std::vector<std::byte> readFile(const std::string& path);
  // just to enable glog-logger on runtime.
  logging::LoggerDefault loggerDefault_;
  void doKernelLaunch(rt::KernelId, std::byte* params, size_t size, uint64_t shireMask, uint32_t deviceIdx);
  void reportUserException(const rt::StreamError& error) const;
  void createRuntime(bool enableCoreDump, rt::Options options);
  void resetRuntime(bool enableCoreDump);
  rt::IRuntime* getRuntime(bool enableCoreDump);
  AbortManager abortManager_;

  void parse_args(int argc, char* argv[], bool strict);

  // parameters expected
  fs::path gp_sdk_device_installdir_;
  std::string simulator_params_;
  bool enableCoreDump_ = false;

protected:
  const Config& config_;
  std::unique_ptr<dev::IDeviceLayer> deviceLayer_;
  rt::RuntimePtr runtime_;
  rt::RuntimePtr runtimeBase_;
  std::vector<rt::DeviceId> devices_;
  std::vector<rt::StreamId> defaultStreams_;
  std::vector<rt::StreamId> traceStreams_;

  std::vector<std::byte*> traceDeviceBuffer_;
  uint32_t numDev_ = 0;
  size_t devIdx_ = 0;
};

#endif // GENERIC_LAUNCHER_H
