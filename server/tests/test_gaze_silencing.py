"""The run's gaze switch: its state, who hears of a change, and when they stop hearing."""

# Local package imports
from nav.runtime.gaze_silencing import GazeSilencingSwitch


def test_a_listener_hears_the_state_on_subscribing_and_on_each_change() -> None:
    switch = GazeSilencingSwitch(False)
    heard = []

    switch.subscribe(heard.append)
    switch.request(True)
    switch.request(False)

    assert heard == [False, True, False]
    assert switch.enabled is False


def test_asking_for_the_state_already_in_force_tells_nobody() -> None:
    switch = GazeSilencingSwitch(True)
    heard = []
    switch.subscribe(heard.append)

    switch.request(True)

    assert heard == [True]


def test_a_listener_that_unsubscribed_hears_nothing_more() -> None:
    switch = GazeSilencingSwitch(False)
    heard = []
    unsubscribe = switch.subscribe(heard.append)

    unsubscribe()
    switch.request(True)

    assert heard == [False]
    assert switch.enabled is True
