"""The run's switches: each one's state, who hears of a change, and when they stop hearing."""

# Local package imports
from nav.planner.config import GoalSwitches, PlannerConfig, RunSwitchKind
from nav.runtime.switches import RunSwitch, RunSwitches


def test_a_listener_hears_the_state_on_subscribing_and_on_each_change() -> None:
    switch = RunSwitch(RunSwitchKind.GAZE_SILENCING, False)
    heard = []

    switch.subscribe(heard.append)
    switch.request(True)
    switch.request(False)

    assert heard == [False, True, False]
    assert switch.enabled is False


def test_asking_for_the_state_already_in_force_tells_nobody() -> None:
    switch = RunSwitch(RunSwitchKind.GAZE_SILENCING, True)
    heard = []
    switch.subscribe(heard.append)

    switch.request(True)

    assert heard == [True]


def test_a_listener_that_unsubscribed_hears_nothing_more() -> None:
    switch = RunSwitch(RunSwitchKind.GAZE_SILENCING, False)
    heard = []
    unsubscribe = switch.subscribe(heard.append)

    unsubscribe()
    switch.request(True)

    assert heard == [False]
    assert switch.enabled is True


def test_every_switch_kind_starts_from_its_config_field() -> None:
    switches = RunSwitches(PlannerConfig(gaze_silences_alarm=True, gaze_goal=False, route_goal=True))
    assert switches[RunSwitchKind.GAZE_SILENCING].enabled is True
    assert switches[RunSwitchKind.GAZE_GOAL].enabled is False
    assert switches[RunSwitchKind.ROUTE_GOAL].enabled is True
    assert set(switches.as_mapping()) == set(RunSwitchKind)


def test_the_goal_switches_follow_their_two_switches() -> None:
    switches = RunSwitches(PlannerConfig())
    assert switches.goal_switches() == GoalSwitches()
    switches[RunSwitchKind.ROUTE_GOAL].request(True)
    assert switches.goal_switches() == GoalSwitches(route_goal=True)
    # The gaze silencing switch is not a goal source.
    switches[RunSwitchKind.GAZE_SILENCING].request(True)
    assert switches.goal_switches() == GoalSwitches(route_goal=True)
