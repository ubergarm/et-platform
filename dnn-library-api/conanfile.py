from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain,CMakeDeps
from conans import tools
from conans.errors import ConanInvalidConfiguration
import os


class DnnLibraryApiConan(ConanFile):
    name = "dnnLibraryApi"
    url = "https://gitlab.esperanto.ai/software/dnn-library-api.git"
    description = "DnnLibrary Host API"
    topics = ("dnnLibraryApi", "dnnLibrary", "neuralizer")
    license = "Esperanto Technologies"

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

    scm = {
        "type": "git",
        "url": "git@gitlab.esperanto.ai:software/dnn-library-api.git",
        "revision": "auto",
    }
    generators = "CMakeDeps"


    python_requires = "conan-common/[>=1.1.0 <2.0.0]"

    def set_version(self):
        get_version = self.python_requires["conan-common"].module.get_version
        self.version = get_version(self, self.name)

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def requirements(self):
        self.requires("dnnLibrary/[>=0.2.0 <1.0.0]", private=True)
    
    def validate(self):
        check_req_min_cppstd = self.python_requires["conan-common"].module.check_req_min_cppstd
        check_req_min_cppstd(self, "17")

        if self.settings.os != "Linux":
            raise ConanInvalidConfiguration("dnnLibraryApi is only supported on Linux")

        dnn_library = self.dependencies["dnnLibrary"]
        dnn_library_flag = "with_host_headers"
        if not dnn_library.options.get_safe(dnn_library_flag):
            raise ConanInvalidConfiguration("{0} requires {1} package with '-o {1}:{2}'".format(self.name, "dnnLibrary", dnn_library_flag))
    
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
