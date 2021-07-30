from conans import ConanFile, tools
from conans.errors import ConanInvalidConfiguration
from conan.tools.cmake import CMakeToolchain, CMake
import os

class DnnLibraryConan(ConanFile):
    name = "dnnLibrary"
    version = "0.1.0"
    license = "Esperanto Technologies"
    author = "Pau Farre <pau.farre@esperantotech.com>" # recipe author
    url = "https://gitlab.esperanto.ai/software/dnn-library"
    description = "<Description of DnnLibrary here>"
    topics = ("dnnLibrary", "dnn", "neuralizer")

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "fPIC": [True, False]
    }
    default_options = {
        "fPIC": True
    }
    generators = "cmake_find_package_multi"

    exports_sources = [ "CMakeLists.txt", "include/*", "scripts/*", "src/*", "dnnLibraryConfig.cmake.in" ]

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC
    
    def configure(self):
        if self.settings.compiler.get_safe("cppstd"):
            tools.check_min_cppstd(self, 17)

    def requirements(self):
        self.requires("device-minion-runtime/0.0.1")

    def validate(self):
        if self.settings.arch != "rv64":
            raise ConanInvalidConfiguration("arch not supported")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_INSTALL_LIBDIR"] = "lib"
        tc.variables["CMAKE_VERBOSE_MAKEFILE"] = True
        tc.variables["ENABLE_WARNINGS_AS_ERRORS"] = False
        tc.variables["ENABLE_DEPRECATED"] = False
        tc.generate()
    
    _cmake = None
    def _configure_cmake(self):
        if not self._cmake:
            cmake = CMake(self)
            cmake.configure()
            self._cmake = cmake
        return self._cmake
    
    def build(self):
        cmake = self._configure_cmake()
        cmake.build()
    
    def package(self):
        cmake = self._configure_cmake()
        cmake.install()
        tools.rmdir(os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)

