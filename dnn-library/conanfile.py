from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake
from conan.tools.layout import cmake_layout
from conans import tools
from conans.errors import ConanInvalidConfiguration
import os


class DnnLibraryConan(ConanFile):
    name = "dnnLibrary"
    version = "0.2.0"
    license = "Esperanto Technologies"
    author = "Pau Farre <pau.farre@esperantotech.com>" # recipe author
    url = "https://gitlab.esperanto.ai/software/dnn-library"
    description = "<Description of DnnLibrary here>"
    topics = ("dnnLibrary", "dnn", "neuralizer")

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "fPIC": [True, False],
        "warnings_as_errors": [True, False]
    }
    default_options = {
        "fPIC": True,
        "warnings_as_errors": False
    }

    scm = {
        "type": "git",
        "url": "git@gitlab.esperanto.ai:software/dnn-library.git",
        "revision": "auto",
    }
    generators = "CMakeDeps"

    python_requires = "conan-common/[>=0.5.0 <1.0.0]"

    def set_version(self):
        self.version = self.python_requires["conan-common"].module.get_version_from_cmake_project(self, self.name)
    
    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC
    
    def requirements(self):
        self.requires("et-common-libs/0.0.5")

    def validate(self):        
        if self.settings.arch != "rv64":
            raise ConanInvalidConfiguration("Cross-compiling to arch {} is not supported".format(self.settings.arch))

        check_req_min_cppstd = self.python_requires["conan-common"].module.check_req_min_cppstd
        check_req_min_cppstd(self, "17")

        et_common_libs = self.dependencies["et-common-libs"]
        # et-common-libs must be compiled with these components
        for flag in ["with_cm_umode"]:
            if not et_common_libs.options.get_safe(flag):
                raise ConanInvalidConfiguration("{0} requires {1} package with '-o {1}:{2}'".format(self.name, "et-common-libs", flag))

    def layout(self):
        cmake_layout(self)
        self.folders.source = "."
    
    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_INSTALL_LIBDIR"] = "lib"
        tc.variables["ENABLE_WARNINGS_AS_ERRORS"] = self.options.warnings_as_errors
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

