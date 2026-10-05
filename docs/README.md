# docs

Project documentation and the diagrams it uses. Prose is Markdown. Diagrams are written in
[D2](https://d2lang.com) and rendered to SVG with the Makefile here, the same way the
Compiler Construction report folder renders its PDF.

## Layout

```
docs/
  Makefile            renders diagrams/*.d2 to diagrams/*.svg
  README.md           this file
  diagrams/           one .d2 per diagram, its .svg committed beside it
  evaluation/         measured results the report draws on, each with the commands that reproduce it
  guides/             short how-tos for running things, written for teammates
  proposal/           the requirements and anything else sent to the professor
```

The laptop pipeline's own documentation is in [`../server/README.md`](../server/README.md).

## Building

Needs `d2` on the path. On Windows, `scoop install d2`. On Mac, `brew install d2`. On Linux,
the install script on d2lang.com. No Node, no Java, no browser.

```
cd docs
make            # render every diagram whose .d2 is newer than its .svg
make check      # compile every .d2 without writing the SVG
make fmt        # canonical formatting
make cleanall   # remove the rendered SVGs
make D2_LAYOUT=elk   # try the other layout engine for one run
```

## Conventions

- One diagram per `.d2` file, named for what it shows, in `diagrams/`.
- The rendered `.svg` is committed next to its source. GitHub renders Markdown without
  running make, so the SVG has to be in the repository for the image to show.
- Markdown references the SVG by relative path, for example `![Pipeline](diagrams/pipeline_components.svg)`.
- Edit the `.d2`, run `make`, commit both files together. Never hand-edit an SVG.
- Default theme, default fonts. Shape and label carry the meaning, not color.
- LF line endings, like everything else in the repository.
- SVG only. d2 can emit PNG, but that path downloads a headless browser on first use and
  fails on a locked-down network. Markdown and the report both take SVG directly.

## Why D2 and not Mermaid

Both would do. D2 is a single binary already installed on the machines we build on, it
renders offline, and it has a layout engine that handles left-to-right pipelines without
fighting. Mermaid's renderer needs Node and a headless browser. If someone prefers writing
Mermaid, GitHub renders fenced ` ```mermaid ` blocks inline in Markdown with no build step at
all, so use that for a quick inline sketch and D2 for anything the report or a document
depends on.
