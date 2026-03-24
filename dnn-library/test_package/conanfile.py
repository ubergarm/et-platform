from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain
import os
import glob

class DnnLibraryApiTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["HEADER_ONLY"] = self.options["dnnLibrary"].header_only
        tc.variables["WITH_DEVICE_HEADERS"] = self.options["dnnLibrary"].with_device_headers and self.settings.arch == "rv64" # only build device-headers test code if cross-compiling to RISC-V
        tc.variables["WITH_HOST_HEADERS"] = self.options["dnnLibrary"].with_host_headers
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
    
    def test(self):
        pass
