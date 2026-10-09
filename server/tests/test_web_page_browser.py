"""
The page in a real browser: what it draws, says and does with what the laptop sends it.

Runs in the Chrome installed on the laptop through browser_harness, against a real web sink over
TLS. Skips by name without Playwright or a browser. Every test fails on any JavaScript error the
page throws, through the fixture's teardown, which is the one guard a session cannot have without
a browser.
"""

# Standard library imports
import dataclasses
import json
import re

# Third party imports
import numpy as np
import pytest

# Local package imports
from browser_harness import (
    PageSession,
    arrow_pixels,
    canvas_pixels,
    canvas_signature,
    launch_browser,
    open_page,
    send_text,
    snapshot,
)
from nav.planner.alarm import avoidance_surprise_bits_at
from nav.scene.config import SceneConfig
from nav.sinks.config import WebConfig
from nav.sinks.web import WebSink
from nav.sinks.web_messages import WebMessageKind
from nav.sources.scene_video import AccessUnit, AccessUnitAssembler, SceneVideoFeed, VideoDescription
from nav.types import DebugView, ObstaclePoint, ObstacleSet
from web_samples import sample_field, sample_path, sample_view

ELEMENT_TIMEOUT_MS = 4000
# Longer than the page's STALE_MS of 1500, fast-forwarded on the fake clock.
PAST_STALE_MS = 1600
PHONE_WIDTH = 412
DESKTOP_WIDTH = 1280


@pytest.fixture(scope="module")
def browser():
    yield from launch_browser()


@pytest.fixture
def sink():
    sink = WebSink(WebConfig(port=0), SceneConfig(), video_feed=None)
    sink.start()
    yield sink
    sink.close()


@pytest.fixture
def browser_page(browser, sink):
    session = open_page(browser, sink)
    yield session
    session.close()
    assert session.errors == [], "the page threw: " + "; ".join(session.errors)


def _wait_for_text(page, element_id: str, fragment: str) -> None:
    page.wait_for_function(
        "([id, fragment]) => document.getElementById(id).textContent.includes(fragment)",
        arg=[element_id, fragment],
        timeout=ELEMENT_TIMEOUT_MS,
    )


def _publish_everything(sink: WebSink) -> None:
    field, grid = sample_field()
    sink.publish_debug(sample_path(heading=0.3), field, grid, sample_view())


def test_the_page_is_a_secure_context_with_a_video_decoder_in_chrome(browser_page: PageSession) -> None:
    # The whole reason the page is HTTPS. WebCodecs only exists on a secure origin.
    assert browser_page.page.evaluate("window.isSecureContext") is True
    assert browser_page.page.evaluate("typeof VideoDecoder") == "function"


def test_a_path_updates_the_status_and_draws_the_arrow(browser_page: PageSession, sink: WebSink) -> None:
    page = browser_page.page
    blank = canvas_signature(page, "arrow")

    sink.publish(sample_path(heading=0.3))

    # 0.3 rad is 17.2 degrees, which the page rounds to 17 with its sign.
    _wait_for_text(page, "status", "Heading +17 degrees")
    assert arrow_pixels(page) > 0
    assert canvas_signature(page, "arrow") != blank, "the arrow at 17 degrees looks exactly like the arrow at 0"


def test_the_alarm_turns_the_page_red_and_back(browser_page: PageSession, sink: WebSink) -> None:
    page = browser_page.page

    sink.publish(sample_path(alarm=True))
    page.wait_for_function("() => document.body.classList.contains('alarm')", timeout=ELEMENT_TIMEOUT_MS)
    _wait_for_text(page, "status", "Obstacle close")

    sink.publish(sample_path(alarm=False))
    page.wait_for_function("() => !document.body.classList.contains('alarm')", timeout=ELEMENT_TIMEOUT_MS)


def test_a_plan_view_shows_the_panel_with_its_two_numbers(browser_page: PageSession, sink: WebSink) -> None:
    page = browser_page.page

    _publish_everything(sink)

    page.wait_for_selector("#plan-panel", state="visible", timeout=ELEMENT_TIMEOUT_MS)
    _wait_for_text(page, "information", "bits")
    assert page.text_content("#scene-bits").endswith(" bits")
    # The sample path has nothing in its corridor.
    assert page.text_content("#contact-time") == "nothing ahead"
    assert canvas_pixels(page, "plan") > 0


def test_the_time_to_collision_is_shown_in_seconds(browser_page: PageSession, sink: WebSink) -> None:
    page = browser_page.page
    field, grid = sample_field()
    one_second_away = dataclasses.replace(sample_path(), avoidance_surprise_bits=avoidance_surprise_bits_at(1.0))

    sink.publish_debug(one_second_away, field, grid, sample_view())

    _wait_for_text(page, "contact-time", " s")
    assert page.text_content("#contact-time") == "1.0 s"


def test_the_legend_starts_folded_and_the_panel_stays_short(browser_page: PageSession, sink: WebSink) -> None:
    page = browser_page.page
    _publish_everything(sink)
    page.wait_for_selector("#plan-panel", state="visible", timeout=ELEMENT_TIMEOUT_MS)
    folded_height = _box(page, "legend")["height"]

    assert not page.evaluate("() => document.getElementById('legend').open")
    page.click("#legend > summary")
    assert _box(page, "legend")["height"] > folded_height, "opening the legend shows its branches"


def test_unticking_field_changes_the_plan_drawing(browser_page: PageSession, sink: WebSink) -> None:
    page = browser_page.page
    _publish_everything(sink)
    page.wait_for_selector("#plan-panel", state="visible", timeout=ELEMENT_TIMEOUT_MS)
    _wait_for_text(page, "information", "bits")
    # The first readback moves the canvas off the GPU (see canvas_signature), so the baseline is a
    # drawing made after that: toggle once, which redraws, before taking it.
    canvas_signature(page, "plan")
    page.uncheck("#show-field")
    page.check("#show-field")
    with_field = canvas_signature(page, "plan")

    page.uncheck("#show-field")
    without_field = canvas_signature(page, "plan")
    page.check("#show-field")

    assert without_field != with_field, "unticking Field changed nothing on the plan"
    assert canvas_signature(page, "plan") == with_field, "ticking Field again did not restore the drawing"


def test_a_depth_picture_appears_when_sent(browser_page: PageSession, sink: WebSink) -> None:
    page = browser_page.page

    _publish_everything(sink)

    page.wait_for_selector("#depth", state="visible", timeout=ELEMENT_TIMEOUT_MS)
    assert page.get_attribute("#depth", "src").startswith("blob:")


def test_choosing_risk_brings_the_risk_picture_and_its_reading_line(browser_page: PageSession, sink: WebSink, monkeypatch: pytest.MonkeyPatch) -> None:
    import nav.sinks.web as web_module

    drawn = []
    real_render = web_module.render_risk_view

    def counting_render(*arguments):
        drawn.append(arguments)
        return real_render(*arguments)

    monkeypatch.setattr(web_module, "render_risk_view", counting_render)
    page = browser_page.page
    _publish_everything(sink)
    page.wait_for_selector("#depth", state="visible", timeout=ELEMENT_TIMEOUT_MS)
    depth_source = page.get_attribute("#depth", "src")
    assert not page.is_visible("#risk-legend")

    # Nothing more is published, so the risk picture can only come from the laptop redrawing the last view.
    page.select_option("#picture-mode", "risk")

    page.wait_for_function("(before) => document.getElementById('depth').src !== before", arg=depth_source, timeout=ELEMENT_TIMEOUT_MS)
    assert page.is_visible("#risk-legend")
    assert len(drawn) == 1


def _view_with_a_group() -> DebugView:
    # The 4 by 4 sample frame, f = 2 and center (2, 2). A group 2 m straight ahead lands on pixel
    # (2, 2), which the 160 times scale-up puts at (320, 320), and the level floor turns nothing.
    group = ObstaclePoint(0.0, 2.0, 1, 1.0, 0.01, None, None, False, np.array([0.0, 0.0, 2.0]))
    return dataclasses.replace(sample_view(), obstacles=ObstacleSet(1.0, (group,), 1))


def test_the_rings_are_drawn_over_the_picture_and_hide_with_their_box(browser_page: PageSession, sink: WebSink) -> None:
    page = browser_page.page
    field, grid = sample_field()
    sink.publish_debug(sample_path(), field, grid, _view_with_a_group())
    page.wait_for_selector("#rings", state="visible", timeout=ELEMENT_TIMEOUT_MS)
    page.wait_for_function("() => document.getElementById('rings').width > 0", timeout=ELEMENT_TIMEOUT_MS)

    assert canvas_pixels(page, "rings") > 0
    ring_box, picture_box = _box(page, "rings"), _box(page, "depth")
    assert all(abs(ring_box[side] - picture_box[side]) <= 1 for side in ("x", "y", "width", "height")), "the rings lie exactly over the picture"

    page.uncheck("#show-rings")
    assert not page.is_visible("#rings")
    page.check("#show-rings")
    assert page.is_visible("#rings")


def test_switching_sound_modes_throws_nothing_and_shows_the_picker_only_in_cancellation(browser_page: PageSession) -> None:
    page = browser_page.page

    for mode in ("alarm", "cancel", "off", "cancel"):
        page.select_option("#mode", mode)
        page.wait_for_timeout(100)
        assert page.is_visible("#music-label") == (mode == "cancel"), f"the music picker in mode {mode}"


def test_the_cue_goes_stale_after_1500_ms(browser, sink: WebSink) -> None:
    session = open_page(browser, sink, fake_clock=True)
    try:
        page = session.page
        page.select_option("#mode", "cancel")
        sink.publish(sample_path())
        _wait_for_text(page, "nc", "NC ON")

        page.clock.fast_forward(PAST_STALE_MS)

        _wait_for_text(page, "nc", "NC ?")
        sink.publish(sample_path())
        _wait_for_text(page, "nc", "NC ON")
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_a_closed_laptop_shows_connection_lost(browser, sink: WebSink) -> None:
    session = open_page(browser, sink)
    try:
        sink.close()

        _wait_for_text(session.page, "link", "Connection lost")
    finally:
        session.close()

    # The laptop is gone, so the page's reconnects fail, and Chrome logs each failed connection
    # as a console error. Those are the browser reporting the laptop's absence, not the page's
    # doing, and they are the one kind of console error this test allows.
    unexpected = [error for error in session.errors if "WebSocket connection" not in error]
    assert unexpected == [], unexpected


def test_a_malformed_text_frame_is_reported_and_the_next_path_still_draws(browser_page: PageSession, sink: WebSink) -> None:
    page = browser_page.page

    send_text(sink, "not json")
    _wait_for_text(page, "status", "Bad message")

    sink.publish(sample_path(heading=0.3))
    _wait_for_text(page, "status", "Heading +17 degrees")


def test_a_message_of_an_unknown_kind_is_ignored(browser_page: PageSession, sink: WebSink) -> None:
    page = browser_page.page
    sink.publish(sample_path(heading=0.3))
    _wait_for_text(page, "status", "Heading +17 degrees")
    before = canvas_signature(page, "arrow")

    send_text(sink, json.dumps({"kind": "later", "heading": 1.0}))
    page.wait_for_timeout(300)

    assert page.text_content("#status") == "Heading +17 degrees."
    assert canvas_signature(page, "arrow") == before


def test_the_collector_catches_a_page_error(browser, sink: WebSink) -> None:
    # The guard's own test. open_page directly, so the fixture's teardown does not fire on the
    # error this test plants on purpose.
    session = open_page(browser, sink)
    try:
        session.page.evaluate("() => { setTimeout(() => { throw new Error('planted'); }, 0); }")
        session.page.wait_for_timeout(200)
    finally:
        session.close()

    assert any("planted" in error for error in session.errors), session.errors


@pytest.mark.parametrize(("name", "width", "height"), [("baseline_desktop", DESKTOP_WIDTH, 800), ("baseline_phone", PHONE_WIDTH, 915)])
def test_a_baseline_screenshot_is_written(browser, sink: WebSink, name: str, width: int, height: int) -> None:
    session = open_page(browser, sink, width=width, height=height)
    try:
        _publish_everything(sink)
        session.page.wait_for_selector("#depth", state="visible", timeout=ELEMENT_TIMEOUT_MS)
        _wait_for_text(session.page, "information", "bits")
        target = snapshot(session.page, name)
    finally:
        session.close()

    assert target.is_file() and target.stat().st_size > 0
    assert session.errors == [], session.errors


# *******************************************
# The video panel
# *******************************************

UNIT_INTERVAL_SECONDS = 1.0 / 30.0
# A stub for the browser's decoder, so the page's own rules around it (the keyframe wait, the
# falling-behind drop, the rebuild on an error) can be driven without a real H.264 stream. It
# records every chunk's type, reports whatever queue size the test sets, and throws on a chunk of
# exactly CORRUPT_UNIT_BYTES bytes of data, which is how a test plants a bad unit.
CORRUPT_UNIT_BYTES = 7
STUB_DECODER_SCRIPT = f"""
window.__decoded = [];
window.__queueSize = 0;
window.__builds = 0;
class StubDecoder {{
  constructor(init) {{ this.state = "unconfigured"; this.init = init; window.__builds++; }}
  configure(config) {{ this.state = "configured"; this.codec = config.codec; }}
  get decodeQueueSize() {{ return window.__queueSize; }}
  decode(chunk) {{
    if (chunk.data.byteLength === {CORRUPT_UNIT_BYTES}) throw new Error("planted corrupt unit");
    window.__decoded.push(chunk.type);
  }}
  close() {{ this.state = "closed"; }}
}}
window.VideoDecoder = StubDecoder;
window.EncodedVideoChunk = class {{ constructor(init) {{ Object.assign(this, init); }} }};
"""


def _real_units(frame_count: int = 12):
    """A real H.264 stream as the glasses would send it, assembled into units. The first is a keyframe."""
    pytest.importorskip("av", reason="PyAV comes with the glasses extra")
    pytest.importorskip("pupil_labs.realtime_api", reason="the client comes with the glasses extra")
    from neon_captures import encode_h264

    stream = encode_h264(frame_count)
    assembler = AccessUnitAssembler(stream.parameter_sets)
    units = []
    for index, picture in enumerate(stream.pictures):
        finished = assembler.feed(picture, index * UNIT_INTERVAL_SECONDS)
        if finished is not None:
            units.append(finished)
    units.append(assembler.flush())
    assert units[0].keyframe, "libx264 starts on an IDR"
    return assembler.description, units


def _unit(index: int, keyframe: bool, size: int = 64) -> AccessUnit:
    return AccessUnit(timestamp_seconds=index * UNIT_INTERVAL_SECONDS, data=bytes([index % 256]) * size, keyframe=keyframe)


def _write_mp4(path):
    """A short H.264 MP4 the browser can play, written with PyAV."""
    pytest.importorskip("av", reason="PyAV comes with the glasses extra")
    import av
    import numpy as np

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    for index in range(20):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 12, dtype=np.uint8), format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture
def feed() -> SceneVideoFeed:
    return SceneVideoFeed()


@pytest.fixture
def video_sink(feed: SceneVideoFeed):
    sink = WebSink(WebConfig(port=0), SceneConfig(), video_feed=feed)
    sink.start()
    yield sink
    sink.close()


def _live_option_disabled(page) -> bool:
    return page.evaluate("() => document.querySelector('#video-mode option[value=\"live\"]').disabled")


def _decoded_count(page) -> int:
    text = page.text_content("#video-status")
    match = re.search(r"(\d+) decoded", text)
    return int(match.group(1)) if match else 0


def _wait_for_decoded(page, count: int) -> None:
    page.wait_for_function(
        "(count) => /(\\d+) decoded/.test(document.getElementById('video-status').textContent) && parseInt(document.getElementById('video-status').textContent.match(/(\\d+) decoded/)[1]) >= count",
        arg=count,
        timeout=ELEMENT_TIMEOUT_MS,
    )


def test_live_video_decodes_in_chrome_and_the_counter_counts(browser, video_sink: WebSink, feed: SceneVideoFeed) -> None:
    description, units = _real_units()
    feed.describe(description)
    session = open_page(browser, video_sink)
    try:
        page = session.page
        _wait_for_text(page, "video-status", "Waiting for the next keyframe")
        blank = canvas_pixels(page, "live")

        for unit in units:
            feed.offer(unit)
        # A decoder may hold the last frame back until the next unit arrives, so one short is fine.
        _wait_for_decoded(page, len(units) - 1)

        assert blank == 0
        assert canvas_pixels(page, "live") > 0, "a decoded frame was drawn on the live canvas"
        assert len(units) - 1 <= _decoded_count(page) <= len(units)
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_a_page_that_joins_mid_stream_draws_its_first_frame_on_the_next_keyframe(browser, video_sink: WebSink, feed: SceneVideoFeed) -> None:
    description, units = _real_units()
    keyframe, deltas = units[0], units[1:]
    feed.describe(description)
    session = open_page(browser, video_sink)
    try:
        page = session.page
        _wait_for_text(page, "video-status", "Waiting for the next keyframe")
        # Deltas first, which a browser that just joined cannot use. Then the keyframe and one more delta.
        for delta in deltas[:3]:
            feed.offer(delta)
        page.wait_for_timeout(300)
        before_keyframe = _decoded_count(page)
        feed.offer(keyframe)
        feed.offer(deltas[3])
        feed.offer(deltas[4])
        _wait_for_decoded(page, 2)

        assert before_keyframe == 0, "deltas before the first keyframe never reach the decoder"
        assert 2 <= _decoded_count(page) <= 3
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_switching_to_recording_plays_the_file_and_back_to_live_keeps_decoding(browser, feed: SceneVideoFeed, tmp_path) -> None:
    recording = _write_mp4(tmp_path / "demo.mp4")
    sink = WebSink(WebConfig(port=0, recording_path=str(recording)), SceneConfig(), video_feed=feed)
    sink.start()
    description, units = _real_units()
    feed.describe(description)
    session = open_page(browser, sink)
    try:
        page = session.page
        # The video socket opens from the path socket's connect, so the first unit, the only
        # keyframe, must not go out before the page says the description arrived.
        _wait_for_text(page, "video-status", "Waiting for the next keyframe")
        for unit in units[:4]:
            feed.offer(unit)
        _wait_for_decoded(page, 3)

        page.select_option("#video-mode", "recording")
        page.wait_for_selector("#recording", state="visible", timeout=ELEMENT_TIMEOUT_MS)
        _wait_for_text(page, "video-status", "Recording")
        page.wait_for_function("() => document.getElementById('recording').currentTime > 0.2", timeout=ELEMENT_TIMEOUT_MS)
        assert page.is_hidden("#live")
        for unit in units[4:8]:
            feed.offer(unit)

        page.select_option("#video-mode", "live")
        page.wait_for_selector("#live", state="visible", timeout=ELEMENT_TIMEOUT_MS)
        assert page.is_hidden("#recording")
        _wait_for_decoded(page, 7)
    finally:
        session.close()
        sink.close()
    assert session.errors == [], session.errors


def test_without_a_recording_the_option_says_so(browser_page: PageSession) -> None:
    page = browser_page.page

    page.select_option("#video-mode", "recording")

    _wait_for_text(page, "video-status", "No recording configured")
    assert page.is_visible("#live") and page.is_hidden("#recording"), "the canvas stays rather than a blank panel"


def test_a_run_without_scene_video_says_so_in_live_mode(browser_page: PageSession) -> None:
    _wait_for_text(browser_page.page, "video-status", "no scene video")
    assert _live_option_disabled(browser_page.page)


@pytest.mark.parametrize("sound_mode", ["off", "alarm", "cancel"])
def test_the_arcs_light_by_the_laptops_gains_in_every_mode(browser_page: PageSession, sink: WebSink, sound_mode: str) -> None:
    page = browser_page.page
    page.select_option("#mode", sound_mode)

    def lit(side: str) -> int:
        return page.evaluate("(side) => document.querySelectorAll('[id^=\"ear-' + side + '-\"].lit').length", side)

    def wait_for_arcs(left: int, right: int) -> None:
        page.wait_for_function(
            "([left, right]) => document.querySelectorAll('[id^=\"ear-left-\"].lit').length === left && document.querySelectorAll('[id^=\"ear-right-\"].lit').length === right",
            arg=[left, right],
            timeout=ELEMENT_TIMEOUT_MS,
        )

    sink.publish(sample_path(ear_gain_left=1.0, ear_gain_right=0.62))
    wait_for_arcs(5, 3)
    sink.publish(sample_path(ear_gain_left=0.10, ear_gain_right=1.0))
    wait_for_arcs(1, 5)
    sink.publish(sample_path(ear_gain_left=0.0, ear_gain_right=0.92))
    wait_for_arcs(0, 5)

    assert (lit("left"), lit("right")) == (0, 5)


def test_the_arcs_go_gray_when_the_laptop_goes_quiet(browser, sink: WebSink) -> None:
    session = open_page(browser, sink, fake_clock=True)
    try:
        page = session.page
        sink.publish(sample_path(ear_gain_left=1.0, ear_gain_right=1.0))
        page.wait_for_function("() => document.querySelectorAll('.arc.lit').length === 10", timeout=ELEMENT_TIMEOUT_MS)

        page.clock.fast_forward(PAST_STALE_MS)

        page.wait_for_function("() => document.querySelectorAll('.arc.lit').length === 0", timeout=ELEMENT_TIMEOUT_MS)
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_the_indicator_goes_off_with_the_alarm_and_back_on_after_two_seconds(browser, sink: WebSink) -> None:
    session = open_page(browser, sink, fake_clock=True)
    try:
        page = session.page
        sink.publish(sample_path(alarm=True))
        _wait_for_text(page, "nc", "NC OFF")

        # Quiet paths keep coming while the hold runs, as they do on a walk, so the stale rule
        # does not fire first. 1.5 s after the alarm cleared the indicator is still off.
        sink.publish(sample_path(alarm=False))
        for _ in range(3):
            page.wait_for_timeout(100)
            page.clock.fast_forward(500)
            sink.publish(sample_path(alarm=False))
        page.wait_for_timeout(100)
        assert page.text_content("#nc") == "NC OFF", "still off 1.5 s after the alarm cleared"

        page.clock.fast_forward(600)
        _wait_for_text(page, "nc", "NC ON")

        page.clock.fast_forward(PAST_STALE_MS)
        _wait_for_text(page, "nc", "NC ?")
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_the_indicator_is_visible_in_every_sound_mode(browser_page: PageSession) -> None:
    page = browser_page.page

    for sound_mode in ("off", "alarm", "cancel"):
        page.select_option("#mode", sound_mode)
        assert page.is_visible("#nc"), f"the indicator in mode {sound_mode}"


def test_hidden_floor_draws_in_the_second_gray(browser_page: PageSession, sink: WebSink) -> None:
    from nav.sinks.web_messages import plan_view_message, web_text_message

    page = browser_page.page
    field, grid = sample_field()
    # Every cell seen, so the hidden mark is the only thing that can change a cell's color.
    plain = {**plan_view_message(sample_path(), field, grid, sample_view(), SceneConfig()), "floor_seen": [[True] * len(grid) for _ in field]}
    send_text(sink, web_text_message(WebMessageKind.PLAN_VIEW, plain))
    page.wait_for_selector("#plan-panel", state="visible", timeout=ELEMENT_TIMEOUT_MS)
    _wait_for_text(page, "information", "bits")
    canvas_signature(page, "plan")  # The first readback, see canvas_signature.
    page.uncheck("#show-field")
    page.check("#show-field")
    without_hidden = canvas_signature(page, "plan")

    hidden = [[False] * len(grid) for _ in field]
    hidden[0][1] = True
    send_text(sink, web_text_message(WebMessageKind.PLAN_VIEW, {**plain, "floor_hidden": hidden}))
    page.wait_for_timeout(300)

    assert canvas_signature(page, "plan") != without_hidden, "a hidden cell drew in the same color as a seen one"


def test_without_a_decoder_live_is_disabled_and_says_why(browser, sink: WebSink) -> None:
    session = open_page(browser, sink, init_script="Object.defineProperty(window, 'VideoDecoder', { value: undefined });")
    try:
        page = session.page
        _wait_for_text(page, "video-status", "can't decode the live video")
        assert _live_option_disabled(page)
        sink.publish(sample_path(heading=0.3))
        _wait_for_text(page, "status", "Heading +17 degrees")
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_a_corrupt_unit_is_reported_and_video_resumes_at_the_next_keyframe(browser, video_sink: WebSink, feed: SceneVideoFeed) -> None:
    feed.describe(VideoDescription("avc1.42801f", (b"\x00\x00\x00\x01\x67sps", b"\x00\x00\x00\x01\x68pps")))
    session = open_page(browser, video_sink, init_script=STUB_DECODER_SCRIPT)
    try:
        page = session.page
        page.wait_for_function("() => window.__builds === 1", timeout=ELEMENT_TIMEOUT_MS)
        feed.offer(_unit(0, keyframe=True))
        page.wait_for_function("() => window.__decoded.length === 1", timeout=ELEMENT_TIMEOUT_MS)

        feed.offer(AccessUnit(timestamp_seconds=1 * UNIT_INTERVAL_SECONDS, data=b"x" * CORRUPT_UNIT_BYTES, keyframe=False))
        _wait_for_text(page, "video-status", "Decoder error")
        feed.offer(_unit(2, keyframe=False))
        feed.offer(_unit(3, keyframe=True))
        page.wait_for_function("() => window.__decoded.length === 2", timeout=ELEMENT_TIMEOUT_MS)

        assert page.evaluate("window.__decoded") == ["key", "key"], "the delta after the error never reached the rebuilt decoder"
        assert page.evaluate("window.__builds") == 2, "rebuilt once"
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_a_binary_frame_shorter_than_the_header_is_ignored(browser, video_sink: WebSink, feed: SceneVideoFeed) -> None:
    feed.describe(VideoDescription("avc1.42801f", (b"\x00\x00\x00\x01\x67sps", b"\x00\x00\x00\x01\x68pps")))
    session = open_page(browser, video_sink, init_script=STUB_DECODER_SCRIPT)
    try:
        page = session.page
        page.wait_for_function("() => window.__builds === 1", timeout=ELEMENT_TIMEOUT_MS)
        before = page.text_content("#video-status")

        page.evaluate("() => feedUnit(new ArrayBuffer(3))")
        page.wait_for_timeout(200)

        assert page.evaluate("window.__decoded") == []
        assert page.text_content("#video-status") == before
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_a_decoder_that_falls_behind_drops_to_a_keyframe_rather_than_growing_its_queue(browser, video_sink: WebSink, feed: SceneVideoFeed) -> None:
    feed.describe(VideoDescription("avc1.42801f", (b"\x00\x00\x00\x01\x67sps", b"\x00\x00\x00\x01\x68pps")))
    session = open_page(browser, video_sink, init_script=STUB_DECODER_SCRIPT)
    try:
        page = session.page
        page.wait_for_function("() => window.__builds === 1", timeout=ELEMENT_TIMEOUT_MS)
        feed.offer(_unit(0, keyframe=True))
        page.wait_for_function("() => window.__decoded.length === 1", timeout=ELEMENT_TIMEOUT_MS)

        page.evaluate("() => { window.__queueSize = 20; }")
        for index in range(1, 11):
            feed.offer(_unit(index, keyframe=False))
        page.wait_for_timeout(300)
        page.evaluate("() => { window.__queueSize = 0; }")
        feed.offer(_unit(11, keyframe=False))
        feed.offer(_unit(12, keyframe=True))
        page.wait_for_function("() => window.__decoded.length === 2", timeout=ELEMENT_TIMEOUT_MS)

        assert page.evaluate("window.__decoded") == ["key", "key"], "the ten deltas behind a full queue were dropped, and the delta before the keyframe too"
    finally:
        session.close()
    assert session.errors == [], session.errors


# *******************************************
# The two layouts
# *******************************************

CARDS_IN_PHONE_ORDER = ["heading-card", "video-card", "plan-card", "depth-card", "sound-card"]
# Every feature the page had before the design pass and gained in it, by the element that shows it.
FEATURE_ELEMENTS = [
    "arrow", "status", "link",
    "video-mode", "video-status", "live",
    "show-field", "show-path", "show-obstacles", "plan", "information", "legend",
    "picture-mode", "show-rings", "depth",
    "mode", "ears", "nc",
]
PHONE_VIEWPORT = (412, 915)
PHONE_LANDSCAPE_VIEWPORT = (915, 412)
DESKTOP_VIEWPORTS = [(1920, 1080), (1536, 960)]
BREAKPOINT = 900
# A finger needs about this much. Apple and Android both say 44 to 48 CSS pixels.
TOUCH_TARGET_PX = 44


def _box(page, element_id: str) -> dict:
    box = page.locator(f"#{element_id}").bounding_box()
    assert box is not None, f"#{element_id} has no box, so it is not laid out"
    return box


def _column_count(page) -> int:
    return page.evaluate("() => getComputedStyle(document.querySelector('main')).gridTemplateColumns.split(' ').length")


def _open_with_everything(browser, sink: WebSink, width: int, height: int, **options) -> PageSession:
    session = open_page(browser, sink, width=width, height=height, **options)
    _publish_everything(sink)
    session.page.wait_for_selector("#depth", state="visible", timeout=ELEMENT_TIMEOUT_MS)
    _wait_for_text(session.page, "information", "bits")
    return session


@pytest.mark.parametrize(("width", "height"), DESKTOP_VIEWPORTS)
def test_the_desktop_layout_puts_the_video_left_and_the_sound_bar_along_the_bottom(browser, sink: WebSink, width: int, height: int) -> None:
    session = _open_with_everything(browser, sink, width, height)
    try:
        page = session.page
        assert page.evaluate("() => getComputedStyle(document.querySelector('main')).display") == "grid"
        video, heading, plan, depth, sound = (_box(page, card) for card in ("video-card", "heading-card", "plan-card", "depth-card", "sound-card"))

        assert video["x"] + video["width"] <= plan["x"] + 1, "the video sits left of from above"
        assert video["y"] <= plan["y"] + 1 and video["y"] + video["height"] >= heading["y"], "the video spans from above's row and the next"
        assert heading["y"] >= plan["y"] + plan["height"] - 1 and depth["y"] >= plan["y"] + plan["height"] - 1, "heading and depth sit under from above"
        assert heading["x"] + heading["width"] <= depth["x"] + 1, "heading sits left of depth"
        assert sound["y"] >= max(card["y"] + card["height"] for card in (video, heading, plan, depth)) - 1, "the sound bar is along the bottom"
        snapshot(page, f"design_desktop_{width}")
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_the_phone_layout_is_one_column_in_the_stated_order(browser, sink: WebSink) -> None:
    session = _open_with_everything(browser, sink, *PHONE_VIEWPORT)
    try:
        page = session.page
        boxes = [_box(page, card) for card in CARDS_IN_PHONE_ORDER]

        assert _column_count(page) == 1
        for above, below in zip(boxes, boxes[1:]):
            assert below["y"] >= above["y"] + above["height"] - 1, "each card sits under the one before"
        snapshot(page, "design_phone_412")
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_the_phone_held_sideways_gets_the_desktop_layout(browser, sink: WebSink) -> None:
    session = _open_with_everything(browser, sink, *PHONE_LANDSCAPE_VIEWPORT)
    try:
        assert _column_count(session.page) > 1
        snapshot(session.page, "design_phone_landscape_915")
    finally:
        session.close()
    assert session.errors == [], session.errors


@pytest.mark.parametrize(("width", "height"), [DESKTOP_VIEWPORTS[0], PHONE_VIEWPORT])
def test_every_feature_is_visible_at_both_viewports(browser, sink: WebSink, width: int, height: int) -> None:
    session = _open_with_everything(browser, sink, width, height)
    try:
        page = session.page
        missing = [element_id for element_id in FEATURE_ELEMENTS if not page.is_visible(f"#{element_id}")]
        page.select_option("#mode", "cancel")
        if not page.is_visible("#music-label"):
            missing.append("music-label")

        assert missing == [], f"not visible at {width} wide: {missing}"
    finally:
        session.close()
    assert session.errors == [], session.errors


@pytest.mark.parametrize(("width", "height"), [DESKTOP_VIEWPORTS[0], PHONE_VIEWPORT])
def test_the_alarm_still_turns_the_whole_page_red(browser, sink: WebSink, width: int, height: int) -> None:
    session = open_page(browser, sink, width=width, height=height)
    try:
        page = session.page
        quiet = page.evaluate("() => getComputedStyle(document.body).backgroundColor")

        sink.publish(sample_path(alarm=True))
        page.wait_for_function("() => document.body.classList.contains('alarm')", timeout=ELEMENT_TIMEOUT_MS)

        assert page.evaluate("() => getComputedStyle(document.body).backgroundColor") != quiet
        snapshot(page, f"design_alarm_{width}")
    finally:
        session.close()
    assert session.errors == [], session.errors


@pytest.mark.parametrize(("width", "height"), [DESKTOP_VIEWPORTS[0], PHONE_VIEWPORT])
def test_nothing_scrolls_sideways_at_either_viewport(browser, sink: WebSink, width: int, height: int) -> None:
    session = _open_with_everything(browser, sink, width, height)
    try:
        page = session.page
        assert page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_the_breakpoint_switches_at_900(browser, sink: WebSink) -> None:
    narrow = open_page(browser, sink, width=BREAKPOINT - 1, height=900)
    wide = open_page(browser, sink, width=BREAKPOINT + 1, height=900)
    try:
        assert _column_count(narrow.page) == 1, "one column just under the breakpoint"
        assert _column_count(wide.page) > 1, "the grid just over it"
    finally:
        narrow.close()
        wide.close()
    assert narrow.errors == [] and wide.errors == []


@pytest.mark.parametrize("width", [PHONE_VIEWPORT[0], BREAKPOINT - 1, BREAKPOINT + 1, DESKTOP_VIEWPORTS[0][0]])
def test_the_plan_canvas_never_collapses(browser, sink: WebSink, width: int) -> None:
    # The knowledge-base trap: a percentage-sized child inside a shrink-to-fit parent ends up 0 high.
    session = _open_with_everything(browser, sink, width, 900)
    try:
        box = _box(session.page, "plan")
        assert box["height"] > 50 and box["width"] > 50, f"the plan canvas is {box['width']} by {box['height']} at {width} wide"
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_coarse_pointer_controls_are_at_least_44_px(browser, sink: WebSink) -> None:
    session = open_page(browser, sink, width=PHONE_VIEWPORT[0], height=PHONE_VIEWPORT[1], is_mobile=True, has_touch=True)
    try:
        page = session.page
        if not page.evaluate("() => matchMedia('(pointer: coarse)').matches"):
            pytest.skip("headless Chrome does not emulate a coarse pointer here, so the touch sizes cannot be checked")
        for control in ("mode", "video-mode", "picture-mode"):
            box = _box(page, control)
            assert box["height"] >= TOUCH_TARGET_PX, f"#{control} is {box['height']} px tall under a finger"
    finally:
        session.close()
    assert session.errors == [], session.errors


def test_the_live_status_says_the_stream_is_undescribed_until_the_laptop_describes_it(browser, video_sink: WebSink, feed: SceneVideoFeed) -> None:
    # A feed exists but has described nothing, which is what a stream without a sequence parameter
    # set leaves the sink with. The page must not claim to be waiting for a keyframe it cannot get.
    description, _ = _real_units(frame_count=1)
    session = open_page(browser, video_sink)
    try:
        page = session.page
        _wait_for_text(page, "video-status", "Waiting for the laptop to describe the stream")
        assert not page.evaluate("() => document.querySelector('#video-mode option[value=live]').disabled")

        feed.describe(description)

        _wait_for_text(page, "video-status", "Waiting for the next keyframe")
    finally:
        session.close()
    assert session.errors == [], session.errors
