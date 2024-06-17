from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.files import collect_libs, rmdir
import os


class DnnLibraryApiConan(ConanFile):
    name = "dnnLibraryApi"
    url = "git@gitlab.com:esperantotech/software/dnn-library-api.git"
    homepage = "https://gitlab.com/esperantotech/software/dnn-library-api"
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

    python_requires = "conan-common/[>=1.1.0 <2.0.0]"

    def set_version(self):
        get_version = self.python_requires["conan-common"].module.get_version
        self.version = get_version(self, self.name)

    def export(self):
        register_scm_coordinates = self.python_requires["conan-common"].module.register_scm_coordinates
        register_scm_coordinates(self)

    def export_sources(self):
        copy_sources_if_scm_dirty = self.python_requires["conan-common"].module.copy_sources_if_scm_dirty
        copy_sources_if_scm_dirty(self)
    
    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC
    
    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def requirements(self):
        self.requires("dnnLibrary/0.10.0")
    
    def validate(self):
        check_req_min_cppstd = self.python_requires["conan-common"].module.check_req_min_cppstd
        check_req_min_cppstd(self, "17")

        if self.settings.os != "Linux":
            raise ConanInvalidConfiguration("dnnLibraryApi is only supported on Linux")

        dnn_library = self.dependencies["dnnLibrary"]
        dnn_library_flag = "with_host_headers"
        if not dnn_library.options.get_safe(dnn_library_flag):
            raise ConanInvalidConfiguration("{0} requires {1} package with '-o {1}:{2}'".format(self.name, "dnnLibrary", dnn_library_flag))
    
    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["ENABLE_WARNINGS_AS_ERRORS"] = self.options.enable_warnings_as_errors
        tc.variables["CMAKE_INSTALL_LIBDIR"] = "lib"
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def source(self):
        get_sources_if_scm_pristine = self.python_requires["conan-common"].module.get_sources_if_scm_pristine
        get_sources_if_scm_pristine(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.libs = collect_libs(self)
