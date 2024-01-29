SOURCEDIR:=./src
DOCSDIR:=./docs
REPORTDIR:=./report
OUTPUTDIR:=./build
OBJECTDIR:=$(OUTPUTDIR)/obj
INCLUDEDIR:=./include
LIBDIR:=./lib

# brh
GLADSOURCEDIR:=./glad/src
GLADLIBDIR:=./glad/include
GLADOBJECTDIR:=$(OBJECTDIR)/glad

WARNINGS:=all extra pedantic

# A little hacky but what can we dooo
override DEBUGFLAGS:=-g3 -Og $(DEBUGFLAGS)

FFLAGS:=no-omit-frame-pointer
STANDARD:=c++20

# would need to be changed for windows or something ig
FFLAGS+=sanitize=address,undefined
EXTENSION:=

SOURCES:=$(wildcard $(SOURCEDIR)/*.cpp)
DEFAULTTARGETS:=$(patsubst $(SOURCEDIR)/%.cpp, $(OUTPUTDIR)/%$(EXTENSION), $(SOURCES))
LIBSOURCES:=$(wildcard $(LIBDIR)/*.cpp)
LIBHEADERS:=$(wildcard $(INCLUDEDIR)/*.hpp)
OBJECTS:=$(patsubst $(LIBDIR)/%.cpp, $(OBJECTDIR)/%.o, $(LIBSOURCES))

# kinda repetitve...
GLADSOURCES:=$(wildcard $(GLADSOURCEDIR)/*.c)
GLADOBJECTS:=$(patsubst $(GLADSOURCEDIR)/%.c, $(GLADOBJECTDIR)/%.o, $(GLADSOURCES))

LDFLAGS+=-lglfw -lGL -lX11 -lpthread -lXrandr -lXi -ldl
CCFLAGS+=-I$(INCLUDEDIR) -I$(GLADLIBDIR) $(WARNINGS:%=-W%) $(FFLAGS:%=-f%) $(DEBUGFLAGS)
CXXFLAGS+=$(CCFLAGS) -std=$(STANDARD)

.PHONY: all clean
.SECONDARY: $(GLADOBJECTS)
.DELETE_ON_ERROR:

ifndef VERBOSE
.SILENT:
endif

# https://stackoverflow.com/a/74742720
define print_step
	$(eval $@_MAINREQ = $(1))
	$(eval $@_TARGET = $(2))
	printf "=== %s -> %s ===\n" '${$@_MAINREQ}' '${$@_TARGET}'
endef

$(OBJECTDIR)/%.o: $(LIBDIR)/%.cpp $(LIBHEADERS)
	mkdir -pv $(OBJECTDIR)
	$(call print_step,$<,$@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(GLADOBJECTDIR)/%.o: $(GLADSOURCEDIR)/%.c
	mkdir -pv $(GLADOBJECTDIR)
	$(call print_step,$<,$@)
	$(CC) $(CCFLAGS) -Wno-pedantic -c $< -o $@

all: $(DEFAULTTARGETS)
$(OUTPUTDIR)/%$(EXTENSION): $(SOURCEDIR)/%.cpp $(OBJECTS) $(GLADOBJECTS) $(LIBHEADERS)
	mkdir -pv $(OUTPUTDIR)
	$(call print_step,$<,$@)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(GLADOBJECTS) $< -o $@ $(LDFLAGS)

# nukes are simpler than surgeries:
clean:
	rm -rfv $(OUTPUTDIR) $(GLADOBJECTDIR)
