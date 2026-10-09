"""
Drives the page in the Chrome installed on this laptop, headless, against a real web sink over TLS.

The page had no test runner before this. Everything it drew was checked by eye on a replay, which
meant every page change needed a person at a screen. These helpers open the page the sink serves
in a real browser, collect every JavaScript error it throws, read what it drew, and take
screenshots, so a page change is proved by the session that made it.

Playwright comes with the `browser` extra and is optional. Without it, or without an installed
Chrome or Edge, every browser test skips by name. The bundled Chromium Playwright can download is
not used: it has no H.264 decoder, so the live video could never be tested on it.
"""

# Standard library imports
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Third party imports
import pytest

# Local package imports
from nav.sinks.web import WebSink, _Slot

SCREENSHOT_DIRECTORY = Path(__file__).parent / "screenshots"
# Chrome first, Edge second. Both ship Google's H.264 decoder, and this laptop has both.
BROWSER_CHANNELS = ("chrome", "msedge")
# Lets the page's AudioContext resume without a tap, so the sound modes can be switched in a test.
LAUNCH_ARGUMENTS = ["--autoplay-policy=no-user-gesture-required"]
CONNECTED_TIMEOUT_MS = 5000


def launch_browser() -> Iterator[object]:
    """
    Start Playwright and the installed Chrome, headless, for the tests of one module. Skips without either.

    A generator, so a module-scoped fixture can `yield from` it and the browser closes with the module.
    """
    pytest.importorskip("playwright.sync_api", reason="the browser tests need the browser extra")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    playwright = sync_playwright().start()
    browser = None
    reasons = []
    for channel in BROWSER_CHANNELS:
        try:
            browser = playwright.chromium.launch(channel=channel, headless=True, args=LAUNCH_ARGUMENTS)
            break
        except PlaywrightError as not_installed:
            # The next channel may be installed. Only when none is does the test skip.
            reasons.append(f"{channel}: {str(not_installed).splitlines()[0]}")
    if browser is None:
        playwright.stop()
        pytest.skip("no installed Chrome or Edge for Playwright to drive. " + "; ".join(reasons))
    try:
        yield browser
    finally:
        browser.close()
        playwright.stop()


@dataclass
class PageSession:
    """One open page and everything it reported. close() closes its context."""

    page: object
    context: object
    errors: list[str] = field(default_factory=list)

    def close(self) -> None:
        self.context.close()


def open_page(
    browser: object,
    sink: WebSink,
    width: int = 1280,
    height: int = 800,
    fake_clock: bool = False,
    init_script: str | None = None,
    **context_options: object,
) -> PageSession:
    """
    Open the sink's page in a fresh context and wait until it reports the socket connected.

    :param fake_clock: Install Playwright's clock before the page loads, so a test can fast-forward
        the page's own timers (the 1.5 s stale rule) rather than wait them out.
    :param init_script: JavaScript run in the page before any of its own, for a test that takes
        a browser feature away or stands a stub in for one.
    :param context_options: Passed to `new_context`, for a device descriptor such as a phone.
    """
    options = {"ignore_https_errors": True, "viewport": {"width": width, "height": height}, **context_options}
    context = browser.new_context(**options)
    if init_script is not None:
        context.add_init_script(init_script)
    page = context.new_page()
    session = PageSession(page=page, context=context)

    def on_console(message) -> None:
        # Every console error counts, a failed resource load included. The page ships an empty
        # icon so the one load Chrome makes on its own, the favicon, cannot fail.
        if message.type == "error":
            session.errors.append(f"console.error: {message.text} ({message.location.get('url', '')})")

    page.on("console", on_console)
    page.on("pageerror", lambda error: session.errors.append(f"pageerror: {error}"))
    if fake_clock:
        page.clock.install()
    page.goto(f"https://127.0.0.1:{sink.port}/")
    page.wait_for_function(
        "() => document.getElementById('link').textContent.startsWith('Connected')",
        timeout=CONNECTED_TIMEOUT_MS,
    )
    return session


def canvas_pixels(page: object, element_id: str) -> int:
    """How many pixels on the canvas are not fully transparent."""
    return page.evaluate(
        """(id) => {
            const canvas = document.getElementById(id);
            const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
            let drawn = 0;
            for (let index = 3; index < data.length; index += 4) if (data[index] !== 0) drawn++;
            return drawn;
        }""",
        element_id,
    )


def canvas_signature(page: object, element_id: str) -> int:
    """
    A hash of everything on the canvas, so two drawings can be told apart without saying how they differ.

    Read twice, and the second read is the answer. Chrome rasterizes a canvas on the GPU until the
    first readback, then moves it to the CPU and rasterizes the same content again with slightly
    different antialiasing, so the first read of a canvas and every later one disagree while the
    pixels drawn are the same. After that first read every redraw is CPU-rasterized and stable, so
    a test that compares two drawings takes both after the canvas has been read once.
    """
    expression = """(id) => {
        const canvas = document.getElementById(id);
        const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
        let hash = 2166136261;
        for (let index = 0; index < data.length; index++) hash = Math.imul(hash ^ data[index], 16777619) >>> 0;
        return hash;
    }"""
    page.evaluate(expression, element_id)
    return page.evaluate(expression, element_id)


def arrow_pixels(page: object) -> int:
    return canvas_pixels(page, "arrow")


def snapshot(page: object, name: str) -> Path:
    """Write a full-page screenshot under tests/screenshots/, which git ignores, and return its path."""
    SCREENSHOT_DIRECTORY.mkdir(exist_ok=True)
    target = SCREENSHOT_DIRECTORY / f"{name}.png"
    page.screenshot(path=str(target), full_page=True)
    return target


def send_text(sink: WebSink, raw: str) -> None:
    """
    Queue a raw text frame for every browser, bypassing the message builders.

    The sink's own `_enqueue`, because there is no public way to send a frame the sink would never
    build, and the page's handling of one is what these tests are about.
    """
    sink._enqueue(_Slot.PATH, raw)


def send_binary(sink: WebSink, raw: bytes) -> None:
    """Queue raw bytes for every browser, the way a depth picture travels. Same reasoning as send_text."""
    sink._enqueue(_Slot.DEPTH_PNG, raw)
