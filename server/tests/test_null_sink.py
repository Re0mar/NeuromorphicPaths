"""Covers the sink that does nothing, which still has to do nothing correctly."""

# Third party imports
import numpy as np

# Local package imports
from nav.sinks.none import NullSink
from nav.types import PlannedPath


def test_the_loop_can_tell_a_plain_sink_from_a_debug_sink_at_runtime() -> None:
    # The loop dispatches on isinstance against the DebugSink protocol. Without runtime_checkable
    # that raises TypeError on the first publish, which a live run found and the suite had not.
    from nav.scene.config import SceneConfig
    from nav.sinks.config import DebugWindowConfig, PhoneAppConfig, WebConfig
    from nav.sinks.debug_window import DebugWindowSink
    from nav.sinks.phone_app import PhoneAppSink
    from nav.sinks.web import WebSink
    from nav.types import DebugSink

    assert not isinstance(NullSink(), DebugSink)
    assert not isinstance(PhoneAppSink(PhoneAppConfig(port=1)), DebugSink)
    assert isinstance(DebugWindowSink(DebugWindowConfig()), DebugSink)
    assert isinstance(WebSink(WebConfig(port=0), SceneConfig(), video_feed=None), DebugSink)


def test_publish_and_close_do_nothing_and_do_not_raise() -> None:
    sink = NullSink()
    path = PlannedPath(0.0, np.array([0.0]), np.array([0.0]), 0.0, False, 0.0, scene_information_bits=0.0, avoidance_surprise_bits=0.0)

    sink.publish(path)
    sink.publish(path)
    sink.close()
    sink.close()
