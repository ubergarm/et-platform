from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain
import os
import glob

class DnnLibraryApiTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
    
    def test(self):
        bin_path = os.path.join("bin", "test_package")
        self.run(bin_path, run_environment=True)
        

