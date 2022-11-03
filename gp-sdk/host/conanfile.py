from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.env import VirtualBuildEnv
from conan.tools.files import update_conandata, load, get, copy, rm, rmdir
from conan.tools.microsoft import is_msvc
from conan.tools.scm import Git, Version
import os

required_conan_version = ">=1.52.0"

class GpSdkHostConan(ConanFile):
    name = "gp-sdk-host"
    url = "git@gitlab.esperanto.ai:software/gp-sdk.git"
    homepage = "https://gitlab.esperanto.ai/software/gp-sdk"
    description = ""
    license = "Esperanto Technologies"


    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
    }

    python_requires = "conan-common/[>=0.5.0 <1.0.0]"

    @property
    def _minimum_cpp_standard(self):
        return 17
    
    # in case the project requires C++14/17/20/... the minimum compiler version should be listed
    @property
    def _compilers_minimum_version(self):
        return {
            "gcc": "7",
            "Visual Studio": "15.7",
            "clang": "7",
            "apple-clang": "10",
        }
    
    def set_version(self):
        self.version = self.python_requires["conan-common"].module.get_version_from_cmake_project(self, self.name)
    
    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC
    
    def export(self):
        git = Git(self, self.recipe_folder)
        if not git.is_dirty():
            _, scm_commit = git.get_url_and_commit()
        else:
            scm_commit = None
        # we store the current url and commit in conandata.yml
        update_conandata(self, {"sources": {"commit": scm_commit, "url": self.url, "is_dirty": git.is_dirty()}})

    def configure(self):
        if self.options.shared:
            try:
                del self.options.fPIC # once removed by config_options, need try..except for a second del
            except Exception:
                pass
        try:
            del self.settings.compiler.libcxx # for plain C projects only
        except Exception:
            pass
        try:
            del self.settings.compiler.cppstd # for plain C projects only
        except Exception:
            pass

    def layout(self):
        self.folders.root = ".."
        self.folders.subproject = "host"
        cmake_layout(self)
    
    def requirements(self):
        self.requires("deviceApi/0.6.0")
        self.requires("deviceLayer/1.1.0")
        self.requires("runtime/0.6.0")
        self.requires("esperantoTrace/0.6.0")
    
        self.requires("gflags/2.2.2")
    
    def validate(self):
        # validate the minimum cpp standard supported. For C++ projects only
        if self.info.settings.compiler.cppstd:
            check_min_cppstd(self, self._minimum_cpp_standard)

        if self.settings.os != "Linux":
            self.output.warn("%s has only been tested under Linux. You're on your own" % self.name)
        minimum_version = self._compilers_minimum_version.get(str(self.info.settings.compiler), False)
        if minimum_version and Version(self.info.settings.compiler.version) < minimum_version:
            raise ConanInvalidConfiguration(f"{self.ref} requires C++{self._minimum_cpp_standard}, which your compiler does not support.")
        # in case it does not work in another configuration, it should validated here too
        if is_msvc(self) and self.info.options.shared:
            raise ConanInvalidConfiguration(f"{self.ref} can not be built as shared on Visual Studio and msvc.")
    
    def build_requirements(self):
        # if another tool than the compiler or CMake is required to build the project (pkgconf, bison, flex etc)
        # self.tool_requires("tool/x.y.z")
        pass
    
    def export_sources(self):
        git = Git(self, self.recipe_folder)
        
        if not git.is_dirty():
            # CLEAN BUILD
            scm_url, scm_commit = git.get_url_and_commit()
            git = Git(self, self.export_sources_folder)
            git.clone(url=scm_url, target=".")
            git.checkout(commit=scm_commit)
        else:
            # LOCAL BUILD from current folder
            source_folder = os.path.join(self.recipe_folder, "..")
            copy(self, "*", source_folder, self.export_sources_folder)
    
    def generate(self):
        # BUILD_SHARED_LIBS and POSITION_INDEPENDENT_CODE are automatically parsed when self.options.shared or self.options.fPIC exist
        tc = CMakeToolchain(self)
        tc.variables["USE_CONAN"] = True
        tc.variables["CMAKE_INSTALL_LIBDIR"] = "lib"
        tc.generate()

        # generates CMake config files for each 'requirement'
        deps = CMakeDeps(self)
        deps.generate()
    
        # In case there are dependencies listed on 'build_requirements', VirtualBuildEnv should be used
        # tc = VirtualBuildEnv(self)
        # tc.generate(scope="build")

    def _patch_sources(self):
        # In case sources need to be patched
        pass

    def build(self):
        self._patch_sources()
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def _cleanup_installdir(self):
        rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))
        rmdir(self, os.path.join(self.package_folder, "share"))
        rm(self, "*.la",  os.path.join(self.package_folder, "lib"))
        rm(self, "*.pdb", os.path.join(self.package_folder, "lib"))
        rm(self, "*.pdb", os.path.join(self.package_folder, "bin"))
    
    def package(self):
        copy(self, pattern="LICENSE", dst=os.path.join(self.package_folder, "licenses"), src=self.source_folder)
        cmake = CMake(self)
        cmake.install()
    
        self._cleanup_installdir()


    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "gp-sdk-host")
        self.cpp_info.set_property("cmake_target_name", "gp-sdk-host::gp-sdk-host")
        self.cpp_info.set_property("pkg_config_name", "gp-sdk-host")

        # If they are needed on Linux, m, pthread and dl are usually needed on FreeBSD too
        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.append("m")
            self.cpp_info.system_libs.append("pthread")
            self.cpp_info.system_libs.append("dl")
        
        # If libraries are added to 'gp-sdk-host' populate self.cpp_info.libs

        bin_folder = os.path.join(self.package_folder, "bin")
        # In case need to find packaged tools when building a package
        self.buildenv_info.append("PATH", bin_folder)
        # In case need to find packaged tools at runtime
        self.runenv_info.append("PATH", bin_folder)
