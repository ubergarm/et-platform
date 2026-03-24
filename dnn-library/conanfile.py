from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.cmake import CMakeToolchain, CMakeDeps, CMake
from conan.tools.files import collect_libs, rmdir
from conan.tools.layout import cmake_layout
import os
import shutil


class DnnLibraryConan(ConanFile):
    name = "dnnLibrary"
    license = "Esperanto Technologies"
    author = "Pau Farre <pau.farre@esperantotech.com>" # recipe author
    url = "git@gitlab.com:esperantotech/software/dnn-library.git"
    homepage = "https://gitlab.com/esperantotech/software/dnn-library"
    description = "<Description of DnnLibrary here>"
    topics = ("dnnLibrary", "dnn", "neuralizer")

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "fPIC": [True, False],
        "warnings_as_errors": [True, False],
        "header_only": [True, False],
        "with_device_headers": [True, False],
        "with_host_headers": [True, False],
    }
    default_options = {
        "fPIC": True,
        "warnings_as_errors": False,
        "header_only": True,
        "with_device_headers": False,
        "with_host_headers": True
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
        if self.options.header_only:
            self.options.rm_safe('fPIC')
            self.options.rm_safe('warnings_as_errors')
    
    def requirements(self):
        if self.options.with_device_headers:
            self.requires("et-common-libs/[>=0.0.5 <1.0.0]")

    def package_id(self):
        if self.options.header_only:
            del self.info.settings.arch
            del self.info.settings.build_type
            del self.info.settings.compiler
            del self.info.settings.os
    
    def validate(self):
        if self.options.header_only:
            return
        
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
        if not self.options.header_only:
            cmake_layout(self)
            self.folders.source = "."
    
    def source(self):
        get_sources_if_scm_pristine = self.python_requires["conan-common"].module.get_sources_if_scm_pristine
        get_sources_if_scm_pristine(self)

    def generate(self):
        if not self.options.header_only:
            deps = CMakeDeps(self)
            deps.generate()

            tc = CMakeToolchain(self)
            tc.variables["CMAKE_INSTALL_LIBDIR"] = "lib"
            tc.variables["ENABLE_WARNINGS_AS_ERRORS"] = self.options.warnings_as_errors
            tc.generate()
    
    def build(self):
        if not self.options.header_only:
            cmake = CMake(self)
            cmake.configure()
            cmake.build()

    def package(self):
        if self.options.header_only:
            host_headers_dst = os.path.join("include", "dnn_lib", "host_headers", "dnn_lib")

            if self.options.with_host_headers and not self.options.with_device_headers:
                self.copy("InstrTableGenerated.h", src=os.path.join(self.source_folder, "include", "host_headers"), dst=host_headers_dst)
            else:
                self.copy("*", src=os.path.join(self.source_folder, "include"), dst=os.path.join("include", "dnn_lib"), keep_path=False)
                src = os.path.join(self.package_folder, "include", "dnn_lib", "InstrTableGenerated.h")
                dst = os.path.join(self.package_folder, "include", "dnn_lib", "host_headers", "dnn_lib", "InstrTableGenerated.h")
                os.mkdir(os.path.join(self.package_folder, "include", "dnn_lib", "host_headers"))
                os.mkdir(os.path.join(self.package_folder, "include", "dnn_lib", "host_headers", "dnn_lib"))
                shutil.move(src, dst)
        else:
            cmake = CMake(self)
            cmake.install()
            
            rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.libs = collect_libs(self)

        if self.options.with_device_headers:
            self.cpp_info.components["dnnLibrary"].set_property("cmake_target_name", "dnnLibrary::dnn_lib")
            self.cpp_info.components["dnnLibrary"].includedirs = ["include"]
            if not self.options.header_only:
                self.cpp_info.components["dnnLibrary"].libdirs = ["lib"]
                self.cpp_info.components["dnnLibrary"].libs = ["dnn_lib"]
                self.cpp_info.components["dnnLibrary"].requires = ["et-common-libs::cm-umode"]

        if self.options.with_host_headers:
            self.cpp_info.components["dnn_lib_host_generated_headers"].set_property("cmake_target_name", "dnnLibrary::dnn_lib_host_generated_headers")
            self.cpp_info.components["dnn_lib_host_generated_headers"].includedirs = [os.path.join("include", "dnn_lib", "host_headers")]
            self.cpp_info.components["dnn_lib_host_generated_headers"].libs = []
            self.cpp_info.components["dnn_lib_host_generated_headers"].requires = []

