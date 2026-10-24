"""Why-wrong reformulation: frames fire, genuine questions pass through."""

from school_index.index_mcp.reformulate import reformulate_query


def test_not_just_frame_yields_the_concept() -> None:
    assert reformulate_query("why is the hypotenuse not just a+b") == "hypotenuse"
    assert reformulate_query("why isn't the hypotenuse just a+b") == "hypotenuse"
    assert (
        reformulate_query("why is the hypotenuse not just a plus b")
        == "hypotenuse"
    )


def test_cant_just_frame_yields_the_trailing_concept() -> None:
    assert (
        reformulate_query("why can't you just add a and b to get the hypotenuse")
        == "hypotenuse"
    )
    assert (
        reformulate_query("why can't I just add the sides to get the hypotenuse?")
        == "hypotenuse"
    )


def test_wrong_frames_yield_the_concept() -> None:
    assert (
        reformulate_query(
            "why is my answer for the hypotenuse wrong when I added a and b"
        )
        == "hypotenuse"
    )
    assert reformulate_query("why is the hypotenuse wrong") == "hypotenuse"


def test_topicless_wrong_falls_back_to_the_original_query() -> None:
    assert reformulate_query("why is my answer wrong") == "why is my answer wrong"
    assert reformulate_query("why is it not just a+b") == "why is it not just a+b"


def test_genuine_why_and_plain_queries_pass_through_unchanged() -> None:
    # Benchmark q19: a genuine why question, NOT a misconception frame.
    assert (
        reformulate_query("why did Europeans explore and colonize America")
        == "why did Europeans explore and colonize America"
    )
    assert (
        reformulate_query("how do I find the hypotenuse of a right triangle")
        == "how do I find the hypotenuse of a right triangle"
    )
    assert reformulate_query("volume of 3d shapes") == "volume of 3d shapes"
    assert (
        reformulate_query("worked example: pythagorean theorem")
        == "worked example: pythagorean theorem"
    )
