from conans import ConanFile, tools
from conan.tools.cmake import CMake, CMakeToolchain
from conans.errors import ConanInvalidConfiguration
import os


class DnnLibraryApiConan(ConanFile):
    name = "dnnLibraryApi"
    version = "0.1.0"
    license = "Esperanto Technologies"
    author = "Pau Farre <pau.farre@esperantotech.com>" # recipe author
    url = "https://gitlab.esperanto.ai/software/dnn-library-api"
    description = "DnnLibrary Host API"
    topics = ("dnnLibraryApi", "dnnLibrary", "neuralizer")

    settings = "os", "arch", "compiler", "build_type"
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

    generators = "cmake_find_package_multi"

    exports_sources = [ "CMakeLists.txt", "include/*", "src/*", "dnnLibraryApiConfig.cmake.in" ]

    build_requires = "cmake-modules/[>=0.4.1 <1.0.0]"
    python_requires = "conan-common/[>=0.1.0 <1.0.0]"

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
        tc.variables["CMAKE_MODULE_PATH"] = os.path.join(self.deps_cpp_info["cmake-modules"].rootpath, "cmake")
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
