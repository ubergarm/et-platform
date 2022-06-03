from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain,CMakeDeps
from conans import tools
from conans.errors import ConanInvalidConfiguration
import os
import re

class DnnLibraryApiConan(ConanFile):
    name = "dnnLibraryApi"
    version = "0.2.0"
    url = "https://gitlab.esperanto.ai/software/dnn-library-api.git"
    license = "Esperanto Technologies"

    description = "DnnLibrary Host API"
    topics = ("dnnLibraryApi", "dnnLibrary", "neuralizer")
    settings = "os", "arch", "compiler", "build_type"

    scm = {
        "type": "git",
        "url": "git@gitlab.esperanto.ai:software/dnn-library-api.git",
        "revision": "auto",
    }
    generators = "CMakeDeps"

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "enable_warnings_as_errors": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "enable_warnings_as_errors": False,
    }

    exports_sources = [ "CMakeLists.txt", "include/*", "src/*", "dnnLibraryApiConfig.cmake.in" ]

    python_requires = "conan-common/[>=0.5.0 <1.0.0]"

    def set_version(self):
        self.version = self.python_requires["conan-common"].module.get_version_from_cmake_project(self, "dnnLibraryApi")       

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def validate(self):
        if self.settings.os != "Linux":
            raise ConanInvalidConfiguration("dnnLibraryApi is only supported on Linux")

        check_req_min_cppstd = self.python_requires["conan-common"].module.check_req_min_cppstd
        check_req_min_cppstd(self, "17")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["ENABLE_WARNINGS_AS_ERRORS"] = self.options.enable_warnings_as_errors
        tc.variables["CMAKE_INSTALL_LIBDIR"] = "lib"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        tools.rmdir(os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
