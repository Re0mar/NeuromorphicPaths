# Builds every diagram under diagrams/ from its .d2 source into an .svg beside it.
# Same shape as the report Makefile in the Compiler Construction course: a DOCUMENTS-style
# variable, one pattern rule, and all / clean / cleanall / install targets.

DIAGRAMS ?= $(wildcard diagrams/*.d2)
SVGS := $(DIAGRAMS:.d2=.svg)

# dagre is d2's default and fits a left-to-right pipeline. elk gives tidier edges on
# diagrams with many crossings; switch per run with  make D2_LAYOUT=elk
D2_LAYOUT ?= dagre
D2_THEME ?= 0
D2_FLAGS ?= --layout=$(D2_LAYOUT) --theme=$(D2_THEME) --pad=40

# Where `make install` copies the finished diagrams. Overridable like the course Makefile's ${out}.
out ?= ../build/docs

.PHONY : all clean cleanall install fmt check

all : $(SVGS)

%.svg : %.d2
	d2 $(D2_FLAGS) $< $@

# d2 has no intermediates, so clean only removes stray previews from manual runs.
clean :
	rm -f diagrams/*.png diagrams/*.pdf diagrams/*.pptx

cleanall : clean
	rm -f $(SVGS)

install : $(SVGS)
	install -Dt $(out) $^

# Rewrites each .d2 in d2's canonical formatting.
fmt :
	d2 fmt $(DIAGRAMS)

# Compiles every .d2 to stdout and discards it. `d2 validate` only parses and passed a file
# that then failed to compile, so the gate has to compile. Cheap enough for a CI gate step.
check :
	@for f in $(DIAGRAMS); do d2 $(D2_FLAGS) $$f - --stdout-format svg > /dev/null || exit 1; done
