from conans import ConanFile, tools

class DnnLibraryApiConan(ConanFile):
    name = "dnnLibraryApi"
    version = "0.1.0"
    license = "Esperanto Technologies"
    author = "Pau Farre <pau.farre@esperantotech.com>" # recipe author
    url = "https://gitlab.esperanto.ai/software/dnn-library-api"
    description = "DnnLibrary Host API"
    topics = ("dnnLibraryApi", "dnnLibrary", "neuralizer")

    exports_sources = "include/*.h"
    no_copy_source = True

    def package(self):
        self.copy("*.h")

    def package_id(self):
        self.info.header_only()
